# Load pre-trained DistilBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(le.classes_)) #, id2label=id2label, label2id=label2id)

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
    data_collator=data_collator
  # compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('model')






# ML Classification Report - Financial Complaint Categories

## Overall Performance Summary
- **Total Samples**: 182,315
- **Overall Accuracy**: 71%
- **Macro Average**: Precision 69%, Recall 66%, F1-Score 67%
- **Weighted Average**: Precision 70%, Recall 71%, F1-Score 70%

## Performance by Category

### High Performing Categories (F1-Score ≥ 0.80)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Dealing with your lender or servicer | 85% | 87% | **86%** | 3,255 |
| Improper use of your report | 81% | 85% | **83%** | 20,402 |
| Managing an account | 80% | 83% | **81%** | 7,388 |

### Good Performing Categories (F1-Score 0.70-0.79)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Problem with a purchase shown on your statement | 78% | 80% | **79%** | 4,611 |
| Trouble during payment process | 76% | 78% | **77%** | 6,730 |
| Unable to get your credit report or credit score | 76% | 74% | **75%** | 1,742 |
| Applying for a mortgage or refinancing | 81% | 68% | **74%** | 2,092 |
| Fees or interest | 71% | 77% | **74%** | 2,023 |
| Fraud or scam | 77% | 71% | **74%** | 2,469 |
| Incorrect information on your report | 72% | 77% | **74%** | 45,861 |
| Closing an account | 73% | 70% | **71%** | 1,905 |
| Managing the loan or lease | 74% | 68% | **71%** | 2,476 |
| Communication tactics | 68% | 71% | **70%** | 4,249 |

### Moderate Performing Categories (F1-Score 0.60-0.69)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Struggling to pay mortgage | 67% | 71% | **69%** | 3,475 |
| Problem with credit reporting investigation | 72% | 67% | **69%** | 28,786 |
| Loan modification, collection, foreclosure | 65% | 71% | **68%** | 2,155 |
| Loan servicing, payments, escrow account | 67% | 66% | **67%** | 2,943 |
| Other features, terms, or problems | 68% | 58% | **63%** | 2,274 |
| Attempts to collect debt not owed | 56% | 64% | **60%** | 14,633 |
| Incorrect information on credit report | 66% | 53% | **58%** | 4,032 |

### Low Performing Categories (F1-Score < 0.60)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Problem when making payments | 57% | 56% | **57%** | 1,996 |
| Written notification about debt | 58% | 51% | **54%** | 5,903 |
| Continued attempts collect debt not owed | 56% | 49% | **52%** | 3,366 |
| Took or threatened negative/legal action | 52% | 38% | **44%** | 3,468 |
| False statements or representation | 47% | 28% | **35%** | 4,081 |

## Key Insights

### Strengths
- **Account Management**: Categories related to account management and credit reporting perform well
- **Large Sample Categories**: High-volume categories like "Incorrect information on your report" (45K samples) maintain good performance
- **Specific Issues**: Well-defined problems like mortgage dealings and purchase disputes show strong classification

### Areas for Improvement
- **Debt Collection**: Multiple debt-related categories show poor performance, suggesting difficulty distinguishing between similar complaint types
- **Legal Actions**: "Took or threatened negative/legal action" has the second-lowest F1-score (44%)
- **False Statements**: Lowest performing category (35% F1-score) indicates challenges in identifying deceptive practices

### Recommendations
1. **Focus on Debt Categories**: Improve feature engineering for debt-related complaints to better distinguish between subcategories
2. **Address Class Imbalance**: Some categories have very different sample sizes, which may affect performance
3. **Feature Enhancement**: Consider adding domain-specific features for poorly performing categories
4. **Data Quality**: Review labeling consistency for categories with low precision or recall