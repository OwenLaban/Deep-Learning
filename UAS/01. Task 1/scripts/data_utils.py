import os
import json
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer


def load_hf_dataset(
    dataset_name: str,
    subset: str = None,
    cache_dir: str = None
) -> DatasetDict:
    if subset:
        dataset = load_dataset(dataset_name, subset, cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    return dataset


def analyze_label_distribution(
    dataset: Dataset,
    label_column: str = 'label',
    label_names: List[str] = None
) -> Dict:
    labels = dataset[label_column]
    
    if isinstance(labels[0], list):
        all_labels = []
        for label_list in labels:
            all_labels.extend(label_list)
        label_counts = Counter(all_labels)
    else:
        label_counts = Counter(labels)
    
    total = len(labels)
    
    distribution = {}
    for label_id, count in sorted(label_counts.items()):
        label_name = label_names[label_id] if label_names else str(label_id)
        distribution[label_name] = {
            'count': count,
            'percentage': count / total * 100
        }
    
    return {
        'total_samples': total,
        'distribution': distribution,
        'num_classes': len(label_counts)
    }


def analyze_text_lengths(
    dataset: Dataset,
    text_column: str = 'text',
    sample_size: int = 10000
) -> Dict:
    if len(dataset) > sample_size:
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        texts = [dataset[int(i)][text_column] for i in indices]
    else:
        texts = dataset[text_column]
    
    word_lengths = [len(text.split()) for text in texts]
    
    char_lengths = [len(text) for text in texts]
    
    return {
        'word_length': {
            'mean': np.mean(word_lengths),
            'median': np.median(word_lengths),
            'std': np.std(word_lengths),
            'min': min(word_lengths),
            'max': max(word_lengths),
            'percentile_95': np.percentile(word_lengths, 95)
        },
        'char_length': {
            'mean': np.mean(char_lengths),
            'median': np.median(char_lengths),
            'std': np.std(char_lengths),
            'min': min(char_lengths),
            'max': max(char_lengths),
            'percentile_95': np.percentile(char_lengths, 95)
        }
    }


def create_tokenization_function(
    tokenizer: AutoTokenizer,
    text_column: str = 'text',
    max_length: int = 128,
    padding: str = 'max_length',
    truncation: bool = True
):
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=truncation,
            max_length=max_length,
            padding=padding
        )
    
    return tokenize_function


def create_sentence_pair_tokenization_function(
    tokenizer: AutoTokenizer,
    text_column_1: str = 'premise',
    text_column_2: str = 'hypothesis',
    max_length: int = 128,
    padding: str = 'max_length',
    truncation: bool = True
):
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_1],
            examples[text_column_2],
            truncation=truncation,
            max_length=max_length,
            padding=padding
        )
    
    return tokenize_function


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    tokenize_fn,
    remove_columns: List[str] = None,
    batched: bool = True
) -> DatasetDict:
    tokenized = dataset.map(
        tokenize_fn,
        batched=batched,
        remove_columns=remove_columns,
        desc="Tokenizing"
    )
    
    return tokenized


def prepare_multilabel_targets(
    dataset: Dataset,
    label_column: str = 'labels',
    num_labels: int = None
) -> Dataset:
    def encode_labels(example):
        labels = example[label_column]
        multi_hot = [0] * num_labels
        for label in labels:
            multi_hot[label] = 1
        return {label_column: multi_hot}
    
    return dataset.map(encode_labels)


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    strategy: str = 'balanced'
) -> np.ndarray:
    label_counts = Counter(labels)
    total = len(labels)
    
    weights = np.ones(num_classes)
    
    for label_id in range(num_classes):
        count = label_counts.get(label_id, 0)
        if count > 0:
            if strategy == 'balanced':
                weights[label_id] = total / (num_classes * count)
            elif strategy == 'inverse':
                weights[label_id] = total / count
            elif strategy == 'sqrt_inverse':
                weights[label_id] = np.sqrt(total / count)
    
    return weights


def compute_multilabel_pos_weights(
    labels: np.ndarray,
    num_labels: int
) -> np.ndarray:
    labels = np.array(labels)
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    
    pos_counts = np.maximum(pos_counts, 1)
    
    pos_weights = neg_counts / pos_counts
    
    return pos_weights


def save_processed_dataset(
    dataset: DatasetDict,
    output_dir: str,
    dataset_name: str
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, f'{dataset_name}_tokenized')
    dataset.save_to_disk(dataset_path)
    
    print(f"✅ Dataset saved to: {dataset_path}")
    return dataset_path


def save_data_config(
    config: Dict,
    output_dir: str,
    config_name: str
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f'{config_name}_config.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=float)
    
    print(f"✅ Config saved to: {config_path}")
    return config_path


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42
) -> DatasetDict:
    train_test = dataset.train_test_split(test_size=test_size + val_size, seed=seed)
    
    val_test = train_test['test'].train_test_split(
        test_size=test_size / (test_size + val_size),
        seed=seed
    )
    
    return DatasetDict({
        'train': train_test['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })


def subset_dataset(
    dataset: Dataset,
    max_samples: int,
    seed: int = 42
) -> Dataset:
    if len(dataset) <= max_samples:
        return dataset
    
    return dataset.shuffle(seed=seed).select(range(max_samples))
