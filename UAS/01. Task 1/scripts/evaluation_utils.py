import os
import json
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    hamming_loss, jaccard_score
)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path: str, device: torch.device = None):
    if device is None:
        device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer


@torch.no_grad()
def get_predictions(
    model,
    dataset,
    batch_size: int = 32,
    device: torch.device = None
) -> Dict[str, np.ndarray]:
    if device is None:
        device = get_device()
    
    model.eval()
    
    all_logits = []
    all_labels = []
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        if 'token_type_ids' in batch:
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        all_logits.extend(outputs.logits.cpu().numpy())
        
        if 'label' in batch:
            all_labels.extend(batch['label'].numpy())
        elif 'labels' in batch:
            all_labels.extend(batch['labels'].numpy())
    
    logits = np.array(all_logits)
    labels = np.array(all_labels) if all_labels else None
    
    if logits.shape[1] > 1:
        predictions = np.argmax(logits, axis=-1)
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    else:
        probabilities = 1 / (1 + np.exp(-logits))
        predictions = (probabilities > 0.5).astype(int)
    
    return {
        'predictions': predictions,
        'logits': logits,
        'probabilities': probabilities,
        'labels': labels
    }


def compute_single_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    y_pred = (y_prob > threshold).astype(int)
    y_true = y_true.astype(int)
    
    return {
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_samples': f1_score(y_true, y_pred, average='samples', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(y_true, y_pred),
        'exact_match': accuracy_score(y_true, y_pred),
        'jaccard_micro': jaccard_score(y_true, y_pred, average='micro', zero_division=0),
        'jaccard_samples': jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1_micro',
    thresholds: np.ndarray = None
) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (y_prob > thresh).astype(int)
        
        if metric == 'f1_micro':
            score = f1_score(y_true, y_pred, average='micro', zero_division=0)
        elif metric == 'f1_macro':
            score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'jaccard':
            score = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
        else:
            score = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> Dict[str, Dict[str, float]]:
    per_class = {}
    
    if len(y_true.shape) == 1:
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, label in enumerate(label_names):
            per_class[label] = {
                'f1': f1_per_class[i],
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'support': int((y_true == i).sum())
            }
    
    else:
        for i, label in enumerate(label_names):
            per_class[label] = {
                'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'support': int(y_true[:, i].sum())
            }
    
    return per_class


@torch.no_grad()
def benchmark_inference(
    model,
    dataset,
    num_samples: int = 500,
    warmup_runs: int = 10,
    benchmark_runs: int = 50,
    device: torch.device = None
) -> Dict[str, float]:
    if device is None:
        device = get_device()
    
    model.eval()
    
    sample_data = dataset.select(range(min(num_samples, len(dataset))))
    input_ids = torch.tensor(sample_data['input_ids']).to(device)
    attention_mask = torch.tensor(sample_data['attention_mask']).to(device)
    
    has_token_types = 'token_type_ids' in sample_data.features
    if has_token_types:
        token_type_ids = torch.tensor(sample_data['token_type_ids']).to(device)
    
    for _ in range(warmup_runs):
        if has_token_types:
            _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        
        if has_token_types:
            _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        times.append(time.perf_counter() - start)
    
    times = np.array(times)
    
    return {
        'mean_time_ms': times.mean() * 1000,
        'std_time_ms': times.std() * 1000,
        'min_time_ms': times.min() * 1000,
        'max_time_ms': times.max() * 1000,
        'samples_per_second': num_samples / times.mean(),
        'ms_per_sample': (times.mean() * 1000) / num_samples
    }


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> str:
    return classification_report(y_true, y_pred, target_names=label_names, digits=4)


def save_evaluation_report(
    report: Dict,
    output_path: str
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    
    print(f"✅ Report saved to: {output_path}")


def evaluate_all_models(
    models_config: Dict[str, str],
    test_dataset,
    label_names: List[str],
    model_dir: str,
    output_dir: str,
    is_multilabel: bool = False
) -> Dict[str, Dict]:
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for model_key, model_name in models_config.items():
        print(f"\n📊 Evaluating {model_key}...")
        
        try:
            model_path = os.path.join(model_dir, model_name)
            model, tokenizer = load_model(model_path, device)
            
            results = get_predictions(model, test_dataset, device=device)
            
            if is_multilabel:
                probs = 1 / (1 + np.exp(-results['logits']))
                threshold, _ = find_optimal_threshold(results['labels'], probs)
                metrics = compute_multilabel_metrics(results['labels'], probs, threshold)
                metrics['optimal_threshold'] = threshold
            else:
                metrics = compute_single_label_metrics(results['labels'], results['predictions'])
            
            benchmark = benchmark_inference(model, test_dataset, device=device)
            
            all_results[model_key] = {
                'metrics': metrics,
                'benchmark': benchmark,
                'predictions': results['predictions'].tolist(),
                'probabilities': results['probabilities'].tolist()
            }
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"   ✅ {model_key}: Accuracy/F1 = {metrics.get('accuracy', metrics.get('f1_micro', 0)):.4f}")
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_key}: {e}")
            all_results[model_key] = {'error': str(e)}
    
    return all_results
