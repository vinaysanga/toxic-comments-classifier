## Standalone script to train a model

from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import evaluate
import numpy as np

# Labels
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# Load dataset and tokenizer
checkpoint = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_datasets = load_from_disk("datasets")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    problem_type="multi_label_classification",
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define evaluation metrics
clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = sigmoid(logits)
    predictions = (predictions > 0.5).astype(int)

    # Flatten predictions and labels to 1D arrays
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Ensure labels are integers
    labels = labels.astype(int)

    return clf_metrics.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="toxic-trainer",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    per_device_train_batch_size=16,  # Adjust based on memory usage
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=3,  # Simulate effective batch size of 48
    fp16=False,  # Mixed precision
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="mlflow",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
