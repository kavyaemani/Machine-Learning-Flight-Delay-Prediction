# Flight Delay Prediction

This project explores various machine learning methods to predict flight delays using a dataset of flight records. Our goal is to handle class imbalance (delays vs. non-delays) by **assigning higher weights** to the minority class rather than performing up/down sampling.

## 1. Overview

- **Dataset:**  
  - The dataset contains numeric and categorical features (e.g., `MONTH`, `DAY_OF_WEEK`, `DISTANCE_GROUP`, `PLANE_AGE`, `DEP_TIME_START`, etc.).
  - The target variable is `DEP_DEL15` (0 = on-time, 1 = delayed).

- **Imbalance Handling:**  
  - Instead of upsampling or downsampling, we apply **class weighting** in Random Forest (`classwt`) and XGBoost (`scale_pos_weight`) to penalize misclassifications of the minority class more heavily.

- **Modeling Techniques:**
  1. **Logistic Regression** – Baseline model.
  2. **Neural Network (nnet)** – Explores non-linear relationships.
  3. **K Nearest Neighbor**
  4. **Generalized Additive Models(GAM)**
  5. **Boosted Tree**
  6. **Decision Tree (rpart)** – Simple, interpretable model with potential for bias on imbalanced data.
  7. **Random Forest (randomForest)** – Ensemble approach with built-in weighting (`classwt`) to handle imbalance.
  8. **XGBoost** – Gradient boosting method, uses `scale_pos_weight` to adjust for class imbalance.

## 2. Key Steps in the Code

1. **Data Loading & Cleaning:**
   - Read `data_for_modeling_clean.csv`.
   - Remove duplicates (`df <- df[!duplicated(df), ]`).
   - Convert relevant columns to factors (e.g., `DEP_DEL15`, `MONTH`, etc.).
   - Split into **train** and **test** sets using stratified sampling (`createDataPartition`).

2. **Feature Scaling (Optional):**
   - Standardize numeric variables in the training set.
   - Apply the same scaling to the test set to prevent data leakage.

3. **Model Training:**
   - **Logistic Regression:** Fit a baseline GLM model, evaluate on test data.
   - **Neural Network:** Perform cross-validation to tune size (hidden neurons) and decay, then train a final model.
   - **Decision Tree:** Use `rpart` with cross-validation to find the best complexity parameter (cp). Optionally, apply `loss` matrix for weighting.
   - **Random Forest:** Train with `randomForest(..., classwt = c("0"=1, "1"=2))` or other ratios to handle imbalance.
   - **XGBoost:** Convert data to matrix form (`model.matrix`), set parameters (like `scale_pos_weight = 2`), run `xgb.cv` for best iteration, and train a final model.

4. **Evaluation Metrics:**
   - **Accuracy** – Overall correctness of predictions.
   - **Precision** – Of flights predicted delayed, how many are actually delayed?
   - **Recall** – Of flights that are truly delayed, how many does the model catch?
   - **F1-Score** – Harmonic mean of precision and recall, balances both.
   - **ConfusionMatrix (caret)** – Summarizes predictions vs. actual classes.

5. **Model Comparison:**
   - Results (Accuracy, Precision, Recall, F1) are stored and compared across models.
   - Weighted approaches often improve recall at the expense of accuracy or precision.

## 3. How to Run

1. **Install Packages** (if needed):
   ```r
   install.packages(c("caret", "nnet", "rpart", "MLmetrics", "gbm", "randomForest", "xgboost"))
   ```
2. **Execute the R Notebook or Script** that contains the above code.  
3. **Inspect Model Outputs**:
   - Compare Accuracy, Precision, Recall, and F1-Score to decide the best approach for your flight delay use case.

## 4. Insights

- **High Accuracy Doesn’t Always Help**: With imbalanced data, focusing on accuracy can result in extremely low recall for the minority class.  
- **Class Weighting**: By penalizing misclassifications of delayed flights, we improve recall but often reduce overall accuracy.  
- **Threshold Tuning**: Adjusting the 0.5 decision threshold can further balance precision and recall.

## 5. License & Contributions

- Feel free to open issues or PRs if you have suggestions or improvements.
- Licensed under [MIT License](LICENSE).

---

**Why No Upsampling?**  
This code demonstrates using **class weights** instead of up/down sampling. Class weighting modifies the learning process to pay extra attention to the minority class. This often simplifies the workflow, avoids artificially replicating data, and can yield better recall for delayed flights.
