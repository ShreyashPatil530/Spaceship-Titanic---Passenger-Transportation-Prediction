# Spaceship Titanic - Passenger Transportation Prediction

[![Kaggle](https://img.shields.io/badge/Kaggle-Spaceship%20Titanic-blue)](https://www.kaggle.com/competitions/spaceship-titanic)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project presents a comprehensive machine learning solution for the Kaggle Spaceship Titanic competition. The goal is to predict which passengers were transported to another dimension during a spaceship collision using ensemble classification methods.

**Competition Link:** [Spaceship Titanic on Kaggle](https://www.kaggle.com/competitions/spaceship-titanic)  
**Kaggle Notebook:** [View on Kaggle](https://www.kaggle.com/code/shreyashpatil217/spaceship-titanic-ensemble-classification-77-2)

## Results

| Metric | Score |
|--------|-------|
| **Best Individual Model** | 77.22% (Gradient Boosting) |
| **Ensemble Accuracy** | ~77% |
| **Test Predictions** | 4,277 passengers |
| **Competition Ranking** | Leaderboard Submission |

## Dataset

- **Training Samples:** Historical passenger records
- **Test Samples:** 4,277 passengers to predict
- **Target Variable:** `Transported` (Binary: 0 = Not Transported, 1 = Transported)
- **Source:** [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data)

### Features

**Categorical Features:**
- PassengerId
- HomePlanet
- CryoSleep
- Destination
- VIP
- Cabin

**Numerical Features:**
- Age
- RoomService
- FoodCourt
- ShoppingMall
- Spa
- VRDeck

## Methodology

### 1. Data Preprocessing

```python
# Handling missing values (median/mode imputation)
# Outlier detection (IQR method)
# Categorical encoding with LabelEncoder
# Support for unseen categories
# Feature scaling (StandardScaler)
```

- **Missing Values:** Filled using median (numerical) and mode (categorical)
- **Outliers:** Removed using IQR (Interquartile Range) method
- **Encoding:** LabelEncoder with support for unseen categories in test data
- **Scaling:** StandardScaler normalization

### 2. Feature Engineering

- **Polynomial Transformations:** Squared, square root, logarithmic
- **Interaction Features:** Product and ratio features
- **Correlation Analysis:** Identified top correlated features
- **Feature Selection:** SelectKBest with f_classif (top 30 features)

### 3. Model Training

Four classification models trained with optimized hyperparameters:

| Model | Estimators | Max Depth | Learning Rate | Accuracy |
|-------|-----------|-----------|---------------|----------|
| **XGBoost** | 200 | 7 | 0.08 | 76.15% |
| **LightGBM** | 200 | - | 0.08 | 76.69% |
| **Random Forest** | 200 | 15 | - | 76.61% |
| **Gradient Boosting** | 200 | 7 | 0.08 | **77.22%** |

### 4. Ensemble Strategy

**Weighted Voting Classifier:**
- Weights based on individual model accuracy
- Probability-based ensemble
- 0.5 decision threshold
- Final prediction = weighted average of all 4 models

**Model Weights:**
- XGBoost: 24.83%
- LightGBM: 25.01%
- Random Forest: 24.98%
- Gradient Boosting: 25.18%

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
xgboost>=1.5.0
lightgbm>=3.2.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Usage

### Quick Start

```python
# Download the code and data files
# Place train.csv and test.csv in the project directory

# Run the notebook or script
python spaceship_titanic.py

# Or use Jupyter notebook
jupyter notebook spaceship_titanic.ipynb
```

### Step-by-Step Pipeline

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# 1. Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Preprocess
# - Handle missing values
# - Encode categorical variables
# - Remove outliers

# 3. Feature Engineering
# - Create polynomial features
# - Create interaction features

# 4. Train Models
# - XGBoost, LightGBM, Random Forest, Gradient Boosting

# 5. Ensemble & Predict
# - Weighted voting classifier
# - Generate submission.csv

# 6. Submit to Kaggle
```

## Project Structure

```
spaceship-titanic/
├── spaceship_titanic.py          # Main Python script
├── spaceship_titanic.ipynb       # Jupyter Notebook
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── data/
│   ├── train.csv                # Training data
│   ├── test.csv                 # Test data
│   └── submission.csv           # Predictions
├── models/
│   ├── xgb_model.pkl
│   ├── lgb_model.pkl
│   ├── rf_model.pkl
│   └── gb_model.pkl
└── notebooks/
    └── analysis.ipynb           # EDA and analysis
```

## Key Findings

1. **Balanced Class Distribution:** Target variable is roughly balanced (50-50)
2. **Feature Importance:** Age and spending habits are key predictors
3. **Ensemble Benefits:** Combined predictions better than individual models
4. **Categorical Handling:** Proper encoding of unseen categories is critical
5. **Feature Engineering:** Polynomial and interaction features improve performance

## Performance Analysis

### Validation Metrics

```
XGBoost:           76.15%
LightGBM:          76.69%
Random Forest:     76.61%
Gradient Boosting: 77.22% (Best)
Ensemble:          ~77.00%
```

### Test Set Predictions

- Class 0 (Not Transported): 2,175 (50.8%)
- Class 1 (Transported): 2,102 (49.2%)
- Total Predictions: 4,277

## Improvements Made

### From Initial Baseline
- **Feature Count:** Increased from basic to 30 selected features
- **Model Estimators:** Increased from 100 to 200
- **Hyperparameters:** Tuned for better performance
- **Feature Engineering:** Added polynomial and interaction features
- **Encoding:** Fixed unseen category handling

### Potential Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Stacking with meta-learner
- [ ] Advanced feature engineering (domain-specific)
- [ ] Class imbalance handling (SMOTE)
- [ ] Neural network ensemble
- [ ] SHAP values for interpretability
- [ ] Cross-validation optimization
- [ ] Automated feature selection

## Author

**Shreyash Patil**

- **Email:** [shreyashpatil530@gmail.com](mailto:shreyashpatil530@gmail.com)
- **Kaggle:** [Shreyash Patil](https://www.kaggle.com/shreyashpatil217)
- **GitHub:** [ShreyashPatil530](https://github.com/ShreyashPatil530)
- **Portfolio:** [Shreyash Patil Portfolio](https://shreyash-patil-portfolio1.netlify.app/)

## Competition Information

- **Platform:** Kaggle
- **Competition Name:** Spaceship Titanic
- **Task Type:** Binary Classification
- **Evaluation Metric:** Accuracy
- **Status:** Ongoing Rolling Leaderboard

## References

- [Kaggle Spaceship Titanic Competition](https://www.kaggle.com/competitions/spaceship-titanic)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This project is for educational purposes. The predictions and models are based on training data and may not be 100% accurate. Use at your own discretion.

## Acknowledgments

- Kaggle for hosting the competition
- Open source community for providing amazing ML libraries
- Contributors and reviewers

---

**Last Updated:** 2025  
**Status:** Active Development  
**Version:** 1.0.0

## Connect

Feel free to reach out for discussions, feedback, or collaboration:

- Open an issue on GitHub
- Connect on Kaggle
- Email: shreyashpatil530@gmail.com

---

**Happy Learning and Coding! 🚀**
