# Load pre-trained DistilBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(le.classes_)) #, id2label=id2label, label2id=label2id)

# Prepare data collator for padding sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
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
    data_collator=data_collator
  # compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('model')




# ML Classification Report - Consumer Financial Complaints

## Overall Model Performance
- **Overall Accuracy**: 71%
- **Total Samples**: 182,315

## Summary Statistics
| Metric | Macro Average | Weighted Average |
|--------|---------------|------------------|
| Precision | 68% | 71% |
| Recall | 67% | 71% |
| F1-Score | 68% | 71% |

## Performance by Category

### Top Performing Categories (F1-Score ≥ 80%)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Improper use of your report** | 83% | 85% | **84%** | 20,402 |
| **Managing an account** | 82% | 79% | **81%** | 7,388 |

### Good Performance (F1-Score 70-79%)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Problem with a purchase shown on your statement** | 79% | 79% | **79%** | 4,611 |
| **Trouble during payment process** | 74% | 80% | **77%** | 6,730 |
| **Incorrect information on your report** | 72% | 80% | **76%** | 45,861 |
| **Unable to get your credit report or credit score** | 75% | 76% | **76%** | 1,742 |
| **Applying for a mortgage or refinancing** | 83% | 67% | **74%** | 2,092 |
| **Fees or interest** | 69% | 79% | **74%** | 2,023 |
| **Fraud or scam** | 79% | 69% | **74%** | 2,469 |
| **Closing an account** | 66% | 76% | **71%** | 1,905 |
| **Communication tactics** | 68% | 72% | **70%** | 4,249 |
| **Managing the loan or lease** | 67% | 73% | **70%** | 2,476 |
| **Problem with credit reporting investigation** | 77% | 64% | **70%** | 28,786 |

### Moderate Performance (F1-Score 60-69%)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Struggling to pay mortgage** | 65% | 73% | **69%** | 3,475 |
| **Loan modification, collection, foreclosure** | 70% | 65% | **67%** | 2,155 |
| **Loan servicing, payments, escrow account** | 70% | 64% | **67%** | 2,943 |
| **Other features, terms, or problems** | 67% | 58% | **62%** | 2,274 |
| **Incorrect information on credit report** | 62% | 60% | **61%** | 4,032 |
| **Attempts to collect debt not owed** | 57% | 63% | **60%** | 14,633 |

### Areas for Improvement (F1-Score < 60%)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Problem when making payments** | 55% | 57% | **56%** | 1,996 |
| **Written notification about debt** | 54% | 57% | **55%** | 5,903 |
| **Continued attempts collect debt not owed** | 58% | 47% | **52%** | 3,366 |
| **Took or threatened negative/legal action** | 50% | 41% | **45%** | 3,468 |
| **False statements or representation** | 44% | 31% | **36%** | 4,081 |

## Key Insights

### Strengths
- **High-volume categories** like "Improper use of your report" (20,402 samples) and "Incorrect information on your report" (45,861 samples) show strong performance
- **Credit reporting issues** generally well-classified
- **Account management** categories perform consistently well

### Areas Needing Attention
- **Debt collection categories** show lower performance, particularly "False statements or representation" (36% F1) and "Took or threatened negative/legal action" (45% F1)
- **Payment-related issues** have moderate performance and may benefit from additional training data
- Categories with **lower sample sizes** tend to have more variable performance

### Recommendations
1. **Collect more training data** for underperforming categories, especially debt collection issues
2. **Feature engineering** around debt collection language patterns
3. **Consider class balancing** techniques for categories with low support
4. **Review misclassifications** in the 36-52% F1-score range for pattern identification