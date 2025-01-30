# Model Performance Comparison

## Overview
This document presents a performance comparison of multiple classification models trained on the given dataset. The models are evaluated based on their **F1-Score**, **Overall Accuracy**, and **Balanced Accuracy**. The primary metric used for ranking is **F1-Score**, which balances precision and recall.

## Ranking Criteria
1. **F1-Score**: The harmonic mean of precision and recall, used to evaluate the overall effectiveness of the model.
2. **Overall Accuracy**: The percentage of correctly classified instances.
3. **Balanced Accuracy**: The average of recall scores for each class, useful for imbalanced datasets.

## Model Rankings
Average accuracy according on the data produced by `qinh2-survivalData_pipeline_example.Rmd`

| Rank | Model               | Class Accuracy | Balanced Accuracy  | Overall Accuracy | Precision | F1-Score |
|------|---------------------|----------------|--------------------|------------------|-----------|----------|
| 1    | Decision Tree       | 76%            | 71%                | 71%              | 69%       | 68%      |
| 2    | Logistic Regression | 74%            | 74%                | 74%              | 62%       | 67%      |
| 3    | Elastic Net         | 74%            | 73%                | 74%              | 62%       | 67%      |
| 4    | XGBoost             | 69%            | 64%                | 65%              | 60%       | 59%      |
| 5    | Naive Bayes         | 64%            | 72%                | 67%              | 33%       | 47%      |
| 6    | GBM                 | 64%            | 77%                | 68%              | 29%       | 44%      |

## Key Observations
- **Decision Tree** achieved the highest **F1-Score (68%)**, making it the best-performing model in this comparison.
- **Logistic Regression and Elastic Net** both scored **67% F1-Score**, making them closely tied for the second-best spot.
- **XGBoost** performed moderately with a **59% F1-Score**, showing potential with hyperparameter tuning.
- **Naive Bayes** had a relatively lower F1-Score (**47%**), likely due to its assumptions about feature independence, which may not hold in this dataset.
- **GBM** scored **44% F1-Score**, indicating it needs further optimization.

## Conclusion
Based on the ranking, the **Decision Tree model is currently the best performing**, followed by **Logistic Regression and Elastic Net**. XGBoost is promising but may require additional tuning, and Naive Bayes and GBM perform weaker in this scenario.