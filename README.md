# House Price Prediction — Overfitting Analysis with Dropout & Regularization

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-King%20County%20House%20Sales-lightgrey)
![Task](https://img.shields.io/badge/Task-Regression-blue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A systematic study of overfitting in deep neural networks using the King County house sales dataset. Three MLP architectures are designed, trained, and compared - each adding a regularization technique - to demonstrate how dropout and L2 weight regularization improve generalization on unseen data.

---

## What's Implemented

| Model | Architecture | Regularization | Goal |
|-------|-------------|---------------|------|
| **Basic MLP** | 5 Dense layers (80→160→160→80→40) | None | Baseline |
| **MLP + Dropout** | Same + Dropout(0.2–0.5) after each layer | Dropout | Lower val loss than baseline |
| **MLP + L2** | Smaller network with L2(0.01) per layer | L2 weight decay | Lower val loss than both above |

---

## Dataset

- **Source:** [King County House Sales — Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction)
- **Size:** ~21,000 records, 19 features
- **Target:** `price` (house sale price in USD)
- **Split:** 70% train / 30% validation
- **Period:** May 2014 – May 2015, King County (Seattle area)

**Selected features:** bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, condition, grade, yr_built, zipcode, lat, long, and more.

---

## Exploratory Data Analysis

- Pairplot of key features colored by bedroom count
- Boxplots: price vs bedrooms, floors, bathrooms
- Feature distribution histograms (25 bins)
- Pearson correlation heatmap across all numeric features

---

## Data Preprocessing

- Date column parsed into `sale_yr`, `sale_month`, `sale_day`
- Z-score normalization using **training set statistics only** (applied to both train and validation to prevent data leakage)

```
x_normalized = (x - mean_train) / std_train
```

---

## Model Architectures

### Basic MLP
```
Input(19) → Dense(80, tanh) → Dense(160, relu) → Dense(160, relu)
          → Dense(80, relu) → Dense(40, relu)  → Dense(1)
```

### MLP + Dropout
```
Input(19) → Dense(80, tanh) → Dropout(0.2)
          → Dense(160, relu) → Dropout(0.5)
          → Dense(160, relu) → Dropout(0.5)
          → Dense(80, relu)  → Dropout(0.5)
          → Dense(40, relu)  → Dropout(0.5) → Dense(1)
```

### MLP + L2 Regularization
```
Input(19) → Dense(6,  tanh, L2=0.01)
          → Dense(4,  relu, L2=0.01)
          → Dense(11, relu, L2=0.01)
          → Dense(6,  relu, L2=0.01)
          → Dense(8,  relu, L2=0.01) → Dense(1)
```

---

## Training Setup

| Hyperparameter | Value |
|---------------|-------|
| Loss | Mean Squared Error (MSE) |
| Metric | Mean Absolute Error (MAE) |
| Optimizer | Adam |
| Batch size | 128 |
| Epochs | 800 |
| LR schedule | Step decay (factor=0.75, every 200 epochs) |

---

## Results

| Model | Train MAE | Val MAE | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| Basic MLP | — | — | — | — |
| + Dropout | — | — | — | — |
| + L2 | — | — | — | — |


---

## Project Structure

```
House-Price-Prediction-Overfitting-Study/
├── notebook/
│   └── house_price_regression.ipynb   # EDA → preprocessing → 3 model comparison
├── results/
│   ├── Basic MLP - Train vs Validation Loss.png            
│   ├── MLP + Dropout - Train vs Validation Loss.png             
│   ├── MLP + L2 Regularization - Train vs Validation Loss.png            
│   └── Train vs Validation Loss - Model Comparison.png                   
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/FatinIshraq/House-Price-Prediction-Overfitting-Study
cd House-Price-Prediction-Overfitting-Study
pip install -r requirements.txt
jupyter notebook notebook/house_price_regression.ipynb
```

The notebook downloads the dataset automatically on first run.

---

## Key Concepts Demonstrated

- Overfitting diagnosis via train/validation loss divergence
- Z-score normalization with training-only statistics (no data leakage)
- Dropout as stochastic regularization during training
- L2 weight decay penalizing large weight coefficients
- Learning rate scheduling with step decay
- Side-by-side model comparison on identical train/val splits

---

## Roadmap

- [x] Basic MLP baseline
- [x] Dropout regularization
- [x] L2 weight regularization
- [ ] Early stopping integration
- [ ] Batch normalization comparison
- [ ] Feature importance analysis

---

## Author

**Fatin Ishraq** 

[![GitHub](https://img.shields.io/badge/GitHub-FatinIshraq-black?logo=github)](https://github.com/FatinIshraq)
