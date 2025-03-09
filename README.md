# Machine Learning Flight Delay Prediction

This project predicts flight delays using various machine learning models implemented in R. Our workflow includes data cleaning, feature engineering, handling class imbalance through upsampling, and model training/evaluation using Logistic Regression, Neural Networks, Decision Trees, Random Forest, and XGBoost (including weighted versions).

## Overview

- **Data:**  
  We use a cleaned dataset (`data_for_modeling_clean.csv`) containing flight information such as month, day of week, departure time, distance, number of seats, weather conditions, and more. The target variable, `DEP_DEL15`, indicates whether a flight is delayed (1) or on time (0).

- **Preprocessing:**  
  - **Duplicates & Missing Values:** Duplicate rows were removed and missing values were handled.  
  - **Feature Transformation:** Categorical features (e.g., MONTH, DAY_OF_WEEK, DEP_TIME_START) were converted to factors.  
  - **Upsampling:** Due to class imbalance (52302 on-time vs. 12288 delays), we upsampled the minority class so that both classes have equal representation in the training data.

- **Modeling:**  
  We split the data into training (80%) and test (20%) sets using stratified sampling to preserve class distribution. Models are tuned using stratified 5-fold cross-validation. Our evaluation focuses on metrics robust for imbalanced data (ROC AUC, Precision, Recall, F1-score).  
  - **Logistic Regression** was used as a baseline and to assist with feature selection.  
  - **Neural Networks** and **Decision Trees** were experimented with to capture non-linear relationships.  
  - **Random Forest** and **XGBoost** (including a weighted XGBoost variant to further address imbalance) achieved competitive performance.

- **Probability Correction:**  
  Since training is done on upsampled data, we apply a correction to the predicted probabilities (using Bayes’ formula) to adjust for the original population imbalance.

## How to Run

1. **Requirements:**  
   Install the following R packages:
   - caret
   - nnet
   - rpart
   - MLmetrics
   - gbm
   - randomForest
   - xgboost
   - dplyr
   - DT
   - pROC

2. **Execution:**  
   Open the provided R Notebook or run the scripts in R. The code is organized into sections:
   - Data loading, cleaning, and feature transformation
   - Handling class imbalance (upsampling)
   - Train-Test splitting (with stratification)
   - Model training (including cross-validation and hyperparameter tuning)
   - Performance evaluation and probability correction

## Results

- **Logistic Regression:** Achieved an overall test accuracy of ~81% with high AUC, but low recall for the minority class.
- **Neural Network & Decision Tree:** Provided insights into non-linear effects but struggled with minority class recall.
- **Random Forest & XGBoost:** The weighted XGBoost model showed strong performance in distinguishing delays, with cross-validated ROC AUC close to 0.93.  
  *Note:* Although unweighted models achieved higher accuracy on the balanced data, probability correction is required to reflect true class distributions, impacting metrics like recall and F1-score.

## Conclusion

This project demonstrates a comprehensive workflow for flight delay prediction. By addressing class imbalance through upsampling and applying probability correction, we ensure that our final models accurately reflect the original data distribution. Our experiments highlight the importance of balancing overall accuracy with metrics such as Recall and F1-score, especially in imbalanced classification scenarios.

## Contact

Feel free to open issues or pull requests for further improvements.

