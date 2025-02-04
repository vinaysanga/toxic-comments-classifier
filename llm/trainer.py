"""
This script implements the training pipeline for a multi-label toxic comment classifier using transformers.


Key Features:
- Uses HuggingFace's Transformers library for model architecture
- Implements custom training with Binary Cross Entropy loss
- Provides comprehensive evaluation metrics including:
  * ROC AUC score
  * F1 scores (micro, macro, per-label)
  * Precision and Recall
  * Accuracy
- Supports mixed precision training
- Integrates with MLflow for experiment tracking

Typical usage:
    python trainer.py

Note: Requires pre-processed dataset in HuggingFace's datasets format.
"""

from torch import nn
from datasets import load_from_disk
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch
import numpy as np


# Configuration
CONFIG = {
    'checkpoint': 'v2-bert-31/checkpoint-1350',
    'labels': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
    'data': "datasets-bert-31",
    'run-name': "bert-31"
}


# Labels
labels = CONFIG['labels']
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# Load dataset and tokenizer
checkpoint = CONFIG['checkpoint']
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_datasets = load_from_disk(CONFIG['data'])

def multi_label_metrics(predictions, labels, threshold=0.5):
    """
    Calculate comprehensive evaluation metrics for multi-label classification.
    
    This function computes various metrics including ROC AUC, F1 scores,
    precision, recall, and accuracy for both overall and per-label evaluation.
    
    Args:
        predictions (numpy.ndarray or torch.Tensor): Raw model predictions (logits)
        labels (numpy.ndarray or torch.Tensor): Ground truth labels
        threshold (float, optional): Classification threshold for converting
            probabilities to binary predictions. Defaults to 0.5.
    
    Returns:
        dict: A dictionary containing the following metrics:
            - roc_auc: Area under ROC curve (micro-averaged)
            - f1_micro: Micro-averaged F1 score
            - f1_macro: Macro-averaged F1 score
            - precision_micro: Micro-averaged precision
            - precision_macro: Macro-averaged precision
            - recall_micro: Micro-averaged recall
            - recall_macro: Macro-averaged recall
            - accuracy: Overall accuracy
            - f1_label_X: Individual F1 scores for each label
    """
    # Ensure predictions and labels are NumPy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Apply sigmoid to convert logits to probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(predictions)).numpy()

    # Convert probabilities to binary predictions using the threshold
    y_pred = (probs >= threshold).astype(int)
    y_true = labels

    # Initialize metrics dictionary
    metrics = {}

    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, probs, average='micro', multi_class='ovr')
    except ValueError:
        # Handle cases where one label is missing positive/negative samples
        roc_auc = np.nan
    metrics['roc_auc'] = roc_auc

    # Calculate overall metrics
    metrics.update({
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    })

    # Add per-label F1 scores
    metrics.update({
        f'f1_label_{i}': f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        for i in range(y_true.shape[1])
    })

    return metrics

def compute_metrics_v2(p: EvalPrediction, threshold=0.5):
    """
    Computes metrics for Hugging Face Trainer using multi-label classification.
    Args:
        p (EvalPrediction): Contains predictions and labels.
        threshold (float): Threshold for converting probabilities to binary predictions.
    Returns:
        dict: A dictionary of evaluation metrics.
    """
    # Extract predictions (logits) and labels
    if isinstance(p.predictions, tuple):
        preds = p.predictions[0]  # Handle models returning (logits, hidden_states, attentions)
    else:
        preds = p.predictions
    
    return multi_label_metrics(
        predictions=preds,
        labels=p.label_ids,
        threshold=threshold
    )

class CustomTrainer(Trainer):
    """
    Custom implementation of HuggingFace's Trainer for multi-label toxic comment classification.
    
    This trainer extends the base Trainer class to implement custom loss computation
    using Binary Cross Entropy with Logits Loss, which is specifically suited for
    multi-label classification tasks.
    
    The trainer handles:
    - Custom loss computation
    - Multi-label training dynamics
    - Integration with evaluation metrics
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the Binary Cross Entropy loss for multi-label classification.
        
        Args:
            model: The transformer model being trained
            inputs (dict): Input dictionary containing:
                - input_ids: Tokenized input text
                - attention_mask: Attention mask for padding
                - labels: Ground truth labels
            return_outputs (bool, optional): If True, returns model outputs along
                with the loss. Defaults to False.
            **kwargs: Additional keyword arguments
            
        Returns:
            torch.Tensor or tuple: If return_outputs is False, returns only the loss.
                                 If True, returns a tuple of (loss, outputs).
        """
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')


        # Compute custom loss with class weights
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())  # Convert labels to float for BCEWithLogitsLoss
        
        return (loss, outputs) if return_outputs else loss


# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
# Configure the training process with optimal hyperparameters and settings
training_args = TrainingArguments(
    # Output and checkpoint configuration
    output_dir="v2-distilbert-31-5ep",  # Directory to save model checkpoints
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,  # Keep only the 2 best checkpoints to save disk space
    
    # Logging configuration
    logging_strategy="steps",
    logging_steps=200,
    logging_first_step=True,
    report_to="mlflow",  # Track experiments with MLflow
    
    # Evaluation configuration
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_micro",  # Optimize for micro-averaged F1 score
    
    # Training hyperparameters
    learning_rate=2e-5,  # Standard learning rate for fine-tuning transformers
    num_train_epochs=5,
    weight_decay=0.01,  # L2 regularization to prevent overfitting
    
    # Batch size and memory optimization
    per_device_train_batch_size=16,  # Adjust based on available GPU memory
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=3,  # Simulate larger batch size of 48
    fp16=False,  # Mixed precision training (disabled)
)

# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_v2,
)

# Train the model
trainer.train()