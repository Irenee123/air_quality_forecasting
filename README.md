# Air Quality Forecasting Project

A machine learning project using LSTM neural networks to predict PM2.5 air pollution levels in Beijing.

## Project Overview

This project develops a deep learning solution for forecasting air quality 24 hours in advance. Using historical meteorological and air quality data, I built LSTM models that achieve excellent prediction accuracy for PM2.5 concentrations.

**Best Model Performance:** RMSE 71.98 (validation) | 74.23 (test)

## Repository Structure

```
air_quality_forecasting/
├── notebooks/
│   └── air_quality_forecasting_starter_code.ipynb    # Main analysis and experiments
├── models/
│   ├── best_model_architecture.keras                 # Top performing model
│   ├── baseline_model.keras                         # Simple baseline
│   ├── complex_model.keras                          # Deep architecture variant
│   └── model_metadata.json                          # Model configurations
├── submission/
│   ├── submission.csv                               # Kaggle predictions
│   ├── submission_corrected.csv                     # Improved predictions
│   └── submission_improved.csv                      # Final submission
└── results_experiment_table.csv                     # 15 experiments summary
```

## Dataset

- **Training samples:** 30,676 records
- **Features:** 17 (meteorological + air quality parameters)
- **Target:** PM2.5 concentration levels
- **Time window:** 24-hour sequences for pattern recognition

## Model Architecture

**Optimal LSTM Configuration:**
- 2 LSTM layers (64 units each)
- 35% dropout rate for regularization
- 24-hour input sequences
- Adam optimizer (lr=0.0008)
- Training time: 7.5 minutes

## Key Results

| Experiment | Architecture | Validation RMSE | Notes |
|------------|-------------|-----------------|-------|
| EXP_007 | 64×2 layers, 0.35 dropout | **71.98** | Best performance |
| EXP_010 | Complex 3-layer | 78.45 | Good but slower |
| EXP_006 | Low learning rate | 85.12 | Stable training |
| EXP_002 | Medium baseline | 98.45 | Standard comparison |

Complete results: 15 experiments testing different architectures, hyperparameters, and training strategies.

## Running the Code

1. **Setup Environment:**
   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn
   ```

2. **Run Main Notebook:**
   Open `notebooks/air_quality_forecasting_starter_code.ipynb` in Jupyter

3. **Load Best Model:**
   ```python
   model = tf.keras.models.load_model('models/best_model_architecture.keras')
   ```

## Technical Approach

- **Data preprocessing:** Missing value handling, sequence creation, temporal splitting
- **Architecture design:** LSTM layers with dropout regularization
- **Optimization:** Systematic hyperparameter tuning across 15 experiments
- **Evaluation:** Temporal validation split to prevent data leakage

## Applications

The model enables:
- 24-hour air quality forecasting for public health warnings
- Data-driven traffic and industrial emission management
- Healthcare resource planning for respiratory emergencies
- Real-time environmental monitoring integration

## Files Description

- `air_quality_forecasting_starter_code.ipynb` - Complete analysis with data exploration, model training, and evaluation
- `results_experiment_table.csv` - Systematic comparison of 15 different model configurations
- `models/` - Saved model variants for different use cases
- `submission/` - Kaggle competition prediction files

---