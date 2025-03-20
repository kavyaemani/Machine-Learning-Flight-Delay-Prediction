# Airlines Delay Prediction

## Project Overview
This project aims to develop a predictive model for airline flight delays. Initially, we attempted modeling using all available features, but the results showed no significant improvement. To address this, we utilized **logistic regression for feature selection** and then applied various **non-linear models** to improve prediction accuracy. Among these, **XGBoost**, which was not introduced in class, showed promising results in enhancing model performance.

## Approach

### 1. Data Collection
- The dataset was sourced from the **Bureau of Transportation Statistics** and **NOAA**, as published on [Kaggle](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations/data).
- It consists of **6 million rows and 24 columns**, including airline information, weather conditions, and flight schedules.

### 2. Feature Selection
- We used **logistic regression** to identify the most relevant features and remove redundant ones.
- A **Variance Inflation Factor (VIF) analysis** and **stepwise selection** helped refine the feature set.

### 3. Model Training
We experimented with several machine learning models, including:
- **Logistic Regression** (baseline)
- **Decision Trees**
- **Random Forest**
- **Gradient Boosting**
- **Generalized Additive Model (GAM)**
- **Neural Networks**
- **K-Nearest Neighbors (KNN)**
- **Extreme Gradient Boosting (XGBoost)**

To handle class imbalance (since only **18.9%** of flights are delayed), we applied **class weighting techniques** and evaluated models based on:
- **Precision**
- **Recall**
- **F1-score** (a priority metric due to class imbalance)

### 4. Model Performance & Results
- **Baseline Logistic Regression** had poor recall, failing to detect delayed flights.
- **Tree-based models** improved performance, but overfitting was a concern.
- **XGBoost with class weighting (4:1)** provided the best balance of recall and precision, making it the most robust choice.

| Model                     | Accuracy | Precision | Recall | F1-score |
|---------------------------|----------|-----------|--------|----------|
| Logistic Regression       | 80.92%   | 43.14%    | 0.9%  | 1.75%   |
| Decision Tree (Weighted)  | 65.06%   | 28.47%    | 55.33% | 37.6%    |
| Neural Network  | 81.16%   | 56.45%    | 4.27% | 7.94%    |
| KNN  | 76.9%   | 30.14%    | 16.27% | 21.14%    |
| GAM  | 81.06%   | 56.32%    | 1.99% | 3.85%    |
| Boosted Tree (Weighted)  | 65.85%   | 29.81%    | 58.67% | 39.53%    |
| Random Forest (Weighted)  | 81.12%   | 52.38%    | 8.5%   | 14.63%   |
| XGBoost (Weighted)    | **66.09%** | **29.95%** | **58.46%** | **39.61%** |

## Key Findings
- **Departure time** is the strongest predictor of delays.
- **Weather conditions** (precipitation, snow, wind speed) are crucial factors.
- **Class weighting (4:1 ratio)** significantly improves recall for delayed flights.
- **XGBoost outperformed other models**, providing the best trade-off between detecting delays and maintaining precision.

## Repository Information
- **Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations/data)
- **Codebase:** [GitHub Repository](https://github.com/glenyslion/airlines_delay_prediction)
