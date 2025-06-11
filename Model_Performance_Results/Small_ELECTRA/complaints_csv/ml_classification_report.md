# Load pre-trained ELECTRA SMALL Discriminator model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=5)
 
# Prepare data collator for padding sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
    # push_to_hub=True
)

# Define Trainer object for training the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    # tokenizer=tokenizer,
    data_collator=data_collator #
   # compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('model')




complaints.csv






# ML Classification Performance Report

## Model Overview
- **Total Samples**: 70,687
- **Overall Accuracy**: 90%

## Class-Level Performance

| Class | Precision | Recall | F1-Score | Support | Performance Summary |
|-------|-----------|--------|----------|---------|-------------------|
| **Incorrect information on your report** | 94% | 94% | 94% | 45,861 | Excellent - Best performing class |
| **Struggling to pay mortgage** | 94% | 93% | 94% | 3,475 | Excellent - High precision & recall |
| **Fraud or scam** | 93% | 88% | 91% | 2,469 | Very Good - Strong precision |
| **Communication tactics** | 86% | 81% | 83% | 4,249 | Good - Room for improvement |
| **Attempts to collect debt not owed** | 78% | 78% | 78% | 14,633 | Fair - Needs attention |

## Key Insights

### Strengths
- **Dominant Class Excellence**: "Incorrect information on your report" (65% of data) performs exceptionally well with 94% across all metrics
- **Mortgage Issues**: Despite smaller sample size, "Struggling to pay mortgage" achieves excellent performance
- **Fraud Detection**: Strong precision (93%) for fraud cases, though recall could be improved

### Areas for Improvement
- **Debt Collection Cases**: "Attempts to collect debt not owed" shows the weakest performance (78% F1-score) despite being the second-largest class
- **Communication Tactics**: Moderate performance with room for enhancement in both precision and recall

### Model Statistics
- **Macro Average**: 89% precision, 87% recall, 88% F1-score
- **Weighted Average**: 90% precision, 90% recall, 90% F1-score

## Recommendations
1. **Focus on debt collection classification** - largest opportunity for improvement
2. **Investigate class imbalance** - consider techniques for smaller classes
3. **Review misclassified fraud cases** - improve recall while maintaining high precision