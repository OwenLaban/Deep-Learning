import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class ModelConfig:
    model_name: str
    num_labels: int
    problem_type: str = "single_label_classification"
    max_length: int = 128
    
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    early_stopping_patience: int = 3
    metric_for_best_model: str = "accuracy"


PRETRAINED_MODELS = {
    'bert-base': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'tinybert': 'huawei-noah/TinyBERT_General_4L_312D',
    'roberta': 'roberta-base',
    'albert': 'albert-base-v2'
}


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def create_model(
    model_name: str,
    num_labels: int,
    problem_type: str = "single_label_classification"
) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type
    )
    return model


def create_training_args(
    config: ModelConfig,
    output_dir: str,
    logging_dir: str
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.eval_strategy,
        save_steps=config.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        
        logging_dir=logging_dir,
        logging_steps=100,
        report_to='none',
        
        save_total_limit=2,
    )


def compute_single_label_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
        'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0)
    }


def compute_multilabel_metrics(eval_pred, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import hamming_loss, jaccard_score
    
    predictions, labels = eval_pred
    
    predictions = 1 / (1 + np.exp(-predictions))
    predictions = (predictions > threshold).astype(int)
    labels = labels.astype(int)
    
    return {
        'f1_micro': f1_score(labels, predictions, average='micro', zero_division=0),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'f1_samples': f1_score(labels, predictions, average='samples', zero_division=0),
        'hamming_loss': hamming_loss(labels, predictions),
        'jaccard_micro': jaccard_score(labels, predictions, average='micro', zero_division=0),
        'precision_micro': precision_score(labels, predictions, average='micro', zero_division=0),
        'recall_micro': recall_score(labels, predictions, average='micro', zero_division=0)
    }


def train_model(
    model_key: str,
    model_name: str,
    train_dataset,
    eval_dataset,
    config: ModelConfig,
    output_dir: str,
    logging_dir: str,
    compute_metrics: Callable = None
) -> Dict:
    print(f"\n{'='*80}")
    print(f"🚀 Training: {model_key}")
    print(f"   Model: {model_name}")
    print(f"{'='*80}")
    
    model = create_model(model_name, config.num_labels, config.problem_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    params = count_parameters(model)
    print(f"\n📊 Model Parameters: {params['total']:,}")
    
    model_output_dir = os.path.join(output_dir, f'{model_key}')
    model_logging_dir = os.path.join(logging_dir, model_key)
    
    training_args = create_training_args(config, model_output_dir, model_logging_dir)
    
    if compute_metrics is None:
        if config.problem_type == "multi_label_classification":
            compute_metrics = compute_multilabel_metrics
        else:
            compute_metrics = compute_single_label_metrics
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    )
    
    print("\n🏋️ Starting training...")
    train_result = trainer.train()
    
    print("\n📊 Evaluating...")
    eval_result = trainer.evaluate()
    
    best_model_path = os.path.join(output_dir, f'{model_key}_best')
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    results = {
        'model_key': model_key,
        'model_name': model_name,
        'parameters': params,
        'train_metrics': train_result.metrics,
        'eval_metrics': eval_result,
        'best_model_path': best_model_path
    }
    
    history_path = os.path.join(logging_dir, f'{model_key}_history.json')
    with open(history_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n✅ {model_key} training complete!")
    print(f"   Model saved to: {best_model_path}")
    
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def train_all_models(
    train_dataset,
    eval_dataset,
    config: ModelConfig,
    output_dir: str,
    logging_dir: str,
    models: Dict[str, str] = None,
    compute_metrics: Callable = None
) -> Dict[str, Dict]:
    if models is None:
        models = {
            'bert-base': PRETRAINED_MODELS['bert-base'],
            'distilbert': PRETRAINED_MODELS['distilbert'],
            'tinybert': PRETRAINED_MODELS['tinybert']
        }
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    
    all_results = {}
    
    for model_key, model_name in models.items():
        try:
            results = train_model(
                model_key=model_key,
                model_name=model_name,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                config=config,
                output_dir=output_dir,
                logging_dir=logging_dir,
                compute_metrics=compute_metrics
            )
            all_results[model_key] = results
        except Exception as e:
            print(f"\n❌ Error training {model_key}: {e}")
            all_results[model_key] = {'error': str(e)}
    
    return all_results
