# Load pre-trained ELECTRA SMALL Discriminator model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=len(le.classes_))
 
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






# ML Classification Results Summary

## Overall Performance
- **Accuracy**: 67%
- **Total Samples**: 182,315

## Performance by Category

### High Performing Categories (F1-Score ≥ 0.75)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Dealing with your lender or servicer | 84% | 85% | **84%** | 3,255 |
| Improper use of your report | 78% | 84% | **81%** | 20,402 |
| Managing an account | 78% | 80% | **79%** | 7,388 |
| Problem with a purchase shown on your statement | 78% | 78% | **78%** | 4,611 |
| Trouble during payment process | 74% | 75% | **75%** | 6,730 |

### Moderate Performing Categories (F1-Score 0.60-0.74)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Applying for a mortgage or refinancing | 80% | 66% | **72%** | 2,092 |
| Incorrect information on your report | 68% | 76% | **72%** | 45,861 |
| Fraud or scam | 76% | 67% | **71%** | 2,469 |
| Fees or interest | 64% | 77% | **70%** | 2,023 |
| Unable to get your credit report or credit score | 69% | 71% | **70%** | 1,742 |
| Closing an account | 66% | 72% | **69%** | 1,905 |
| Communication tactics | 67% | 69% | **68%** | 4,249 |
| Managing the loan or lease | 71% | 64% | **68%** | 2,476 |
| Problem with credit reporting investigation | 71% | 60% | **65%** | 28,786 |
| Struggling to pay mortgage | 62% | 69% | **65%** | 3,475 |
| Loan modification, collection, foreclosure | 62% | 66% | **63%** | 2,155 |
| Loan servicing, payments, escrow account | 65% | 60% | **62%** | 2,943 |

### Lower Performing Categories (F1-Score < 0.60)
| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Other features, terms, or problems | 70% | 51% | **59%** | 2,274 |
| Attempts to collect debt not owed | 53% | 61% | **57%** | 14,633 |
| Incorrect information on credit report | 60% | 46% | **52%** | 4,032 |
| Written notification about debt | 50% | 53% | **52%** | 5,903 |
| Problem when making payments | 51% | 51% | **51%** | 1,996 |
| Cont'd attempts collect debt not owed | 49% | 42% | **45%** | 3,366 |
| Took or threatened negative/legal action | 45% | 36% | **40%** | 3,468 |
| False statements or representation | 42% | 15% | **22%** | 4,081 |

## Key Insights

### Strengths
- **Best performing category**: "Dealing with your lender or servicer" (84% F1-score)
- **High-volume success**: "Improper use of your report" performs well with 20,402 samples
- **Consistent performers**: Categories related to account management and payment processing

### Areas for Improvement
- **Poorest performer**: "False statements or representation" (22% F1-score) - very low recall (15%)
- **Debt collection categories**: Both "Attempts to collect debt not owed" and "Cont'd attempts collect debt not owed" show moderate to poor performance
- **Legal action detection**: "Took or threatened negative/legal action" has low precision and recall

### Model Balance
- **Macro Average**: Precision 65%, Recall 63%, F1-Score 64%
- **Weighted Average**: Precision 67%, Recall 67%, F1-Score 67%
- The weighted averages being higher suggests better performance on high-volume categories