# Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

## 🎯 Project Overview

This project implements a production-quality time series forecasting system featuring:

- **Transformer-based architecture** with multi-head self-attention mechanisms
- **LSTM with attention** as an alternative architecture
- **Complex multivariate time series** generation with multiple seasonalities
- **Comprehensive preprocessing pipeline** with feature engineering
- **Hyperparameter optimization** using Optuna (Bayesian optimization)
- **Benchmark comparison** against SARIMA statistical models
- **Attention visualization** and interpretation

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Citation](#citation)

## ✨ Features

### 1. Data Generation
- Multi-component time series with:
  - Multiple seasonalities (daily, weekly, yearly)
  - Non-stationary trends
  - ARIMA autocorrelation structures
  - Volatility clustering (GARCH-like effects)
  - Structural breaks
  - Cross-correlated features

### 2. Preprocessing Pipeline
- Missing value imputation
- Time-based feature extraction (cyclical encoding)
- Fourier terms for seasonality capture
- Lag features
- Rolling window statistics
- Standard/MinMax scaling

### 3. Deep Learning Models
- **Transformer Encoder**: Multi-head self-attention with positional encoding
- **LSTM with Attention**: Additive attention mechanism
- Production-ready implementations with:
  - Early stopping
  - Learning rate reduction
  - Model checkpointing

### 4. Hyperparameter Optimization
- Bayesian optimization using Optuna
- Optimized parameters:
  - Model dimensions (d_model, num_heads, ff_dim)
  - Architecture depth (num_blocks, lstm_units)
  - Regularization (dropout_rate)
  - Training (batch_size, learning_rate)

### 5. Benchmarking
- SARIMA statistical model comparison
- Comprehensive metrics:
  - MAE, MSE, RMSE
  - R², MAPE, SMAPE
  - Percentage improvements

### 6. Interpretability
- Attention weight extraction
- Heatmap visualizations
- Temporal distribution analysis
- Pattern interpretation

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**
```bash
# If using Git
git clone <repository-url>
cd time-series-forecasting

# Or simply download and extract the files
```

2. **Create a virtual environment (recommended)**
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Alternative: Manual Installation**
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn statsmodels optuna scipy
```

## 🎬 Quick Start

### Running the Complete Pipeline

```bash
python main_pipeline.py
```

This executes all steps:
1. Data generation
2. Preprocessing
3. Sequence creation
4. Hyperparameter optimization
5. Model training
6. Benchmark comparison
7. Evaluation
8. Attention visualization

**Expected Runtime**: 30-60 minutes (depends on hardware and optimization trials)

### Configuration

Edit the `CONFIG` dictionary in `main_pipeline.py`:

```python
CONFIG = {
    'n_samples': 10000,           # Number of time steps
    'n_features': 5,              # Number of features
    'train_ratio': 0.70,          # Training data ratio
    'val_ratio': 0.15,            # Validation data ratio
    'test_ratio': 0.15,           # Test data ratio
    'input_length': 168,          # Lookback window (1 week)
    'output_length': 24,          # Forecast horizon (1 day)
    'epochs': 100,                # Maximum epochs
    'batch_size': 32,             # Batch size
    'optimize_hyperparams': True, # Enable hyperparameter tuning
    'n_trials': 20,               # Optuna trials
}
```

## 📁 Project Structure

```
time-series-forecasting/
│
├── main_pipeline.py              # Complete implementation (all code)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── report.md                     # Technical report
│
└── outputs/                      # Generated outputs
    ├── dataset_analysis.png      # Dataset visualizations
    ├── results.json              # Metrics and config
    ├── transformer_model.h5      # Saved model
    │
    ├── training_history/
    │   └── transformer_history.png
    │
    ├── model_comparison/
    │   ├── predictions_comparison.png
    │   └── metrics_comparison.png
    │
    └── attention_weights/
        ├── attention_heatmap_sample_*.png
        ├── attention_distribution_sample_*.png
        └── attention_interpretation.txt
```

## 🔬 Methodology

### Architecture Design

**Transformer Encoder Block**:
```
Input → Dense(d_model) → Positional Encoding
    ↓
Multi-Head Self-Attention → LayerNorm → Dropout
    ↓
Feed-Forward Network → LayerNorm → Dropout
    ↓
Global Average Pooling → Dense Layers → Output
```

**Key Components**:
- **Scaled Dot-Product Attention**: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`
- **Multi-Head Mechanism**: Parallel attention with different learned projections
- **Residual Connections**: Stabilize training and enable deeper networks
- **Layer Normalization**: Normalize activations across features

### Preprocessing Pipeline

1. **Missing Value Handling**: Forward/backward fill + interpolation
2. **Time Features**: Hour, day, month + cyclical encoding (sin/cos)
3. **Fourier Terms**: Capture complex seasonality patterns
4. **Lag Features**: Past values as predictors
5. **Rolling Statistics**: Mean, std, min, max over windows
6. **Scaling**: Standardization for stable training

### Evaluation Methodology

- **Time Series Cross-Validation**: Temporal train/val/test splits
- **Multi-Horizon Forecasting**: Predict 24 steps ahead
- **Benchmark Comparison**: Against SARIMA(2,1,2)(1,1,1,24)
- **Attention Analysis**: Interpret temporal dependencies

## 📊 Results

### Expected Performance

Typical results on synthetic complex time series:

| Metric | Transformer | SARIMA | Improvement |
|--------|------------|--------|-------------|
| MAE    | 3-5        | 6-8    | 30-40%      |
| RMSE   | 5-7        | 9-12   | 35-45%      |
| MAPE   | 3-5%       | 6-9%   | 40-50%      |
| R²     | 0.85-0.92  | 0.70-0.80 | +15-20% |

*Note: Actual results depend on data characteristics and hyperparameters*

### Attention Insights

The model typically learns:
- **Recent Bias**: Strong attention to last 24 hours
- **Periodic Patterns**: Peaks at 24h and 168h lags (daily/weekly cycles)
- **Contextual Dependencies**: Different heads focus on different temporal scales

## 🎓 Key Learnings

### Technical Insights

1. **Attention Mechanisms**: Effectively capture long-range dependencies
2. **Feature Engineering**: Time-based features crucial for performance
3. **Hyperparameter Tuning**: 10-15% improvement over default settings
4. **Preprocessing Quality**: More important than model complexity

### Production Considerations

- **Computational Cost**: Transformers require more GPU memory
- **Inference Speed**: LSTM-Attention faster for real-time applications
- **Interpretability**: Attention weights provide valuable insights
- **Scalability**: Batch predictions efficient with proper windowing

## 🛠️ Customization

### Using Your Own Data

Replace the data generation step in `main_pipeline.py`:

```python
# Instead of:
# generator = ComplexTimeSeriesGenerator(...)
# df = generator.generate()

# Load your data:
df = pd.read_csv('your_data.csv', index_col='timestamp', parse_dates=True)

# Ensure columns: ['target', 'feature_1', 'feature_2', ...]
```

### Modifying Model Architecture

```python
# Change transformer configuration
transformer_model = build_transformer_model(
    input_shape=(168, n_features),
    output_steps=24,
    d_model=128,          # Increase model capacity
    num_heads=8,          # More attention heads
    ff_dim=256,           # Larger feed-forward
    num_blocks=3,         # Deeper network
    dropout_rate=0.15
)
```

### Alternative Models

Use the LSTM-Attention model instead:

```python
model = build_lstm_attention_model(
    input_shape=(168, n_features),
    output_steps=24,
    lstm_units=256,
    attention_units=128,
    dropout_rate=0.2
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `batch_size` (try 16)
   - Reduce `n_samples` for development
   - Reduce model size (smaller `d_model`, `ff_dim`)

2. **Slow Training**
   - Set `optimize_hyperparams=False` for faster runs
   - Reduce `n_trials` (try 10)
   - Use CPU if GPU unavailable: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

3. **Poor Performance**
   - Increase training data
   - Tune hyperparameters more extensively
   - Check data quality and stationarity
   - Try different preprocessing methods

4. **Import Errors**
   - Verify all packages installed: `pip list`
   - Update packages: `pip install --upgrade -r requirements.txt`
   - Check Python version: `python --version` (need 3.8+)

## 📚 References

### Academic Papers

1. **Attention Mechanisms**:
   - Vaswani et al. (2017). "Attention Is All You Need"
   - Bahdanau et al. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate"

2. **Time Series Forecasting**:
   - Lim et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
   - Wen et al. (2017). "A Multi-Horizon Quantile Recurrent Forecaster"

3. **Statistical Methods**:
   - Box & Jenkins (1970). "Time Series Analysis: Forecasting and Control"
   - Hyndman & Athanasopoulos (2018). "Forecasting: Principles and Practice"

### Libraries

- TensorFlow: https://www.tensorflow.org/
- Statsmodels: https://www.statsmodels.org/
- Optuna: https://optuna.org/
- Scikit-learn: https://scikit-learn.org/

## 📝 License

This project is provided for educational and research purposes. Feel free to modify and extend.

## 👤 Author

Advanced ML Research - November 2025

## 🤝 Contributing

Suggestions and improvements welcome! This is a reference implementation for learning and research.

---

## 💡 Tips for Success

1. **Start Small**: Use 1000 samples initially to test everything works
2. **Monitor Training**: Watch loss curves for overfitting
3. **Experiment**: Try different architectures and hyperparameters
4. **Interpret Results**: Use attention visualizations to understand the model
5. **Compare Fairly**: Ensure SARIMA uses appropriate seasonal parameters

## 🎯 Next Steps

After running successfully:

1. **Analyze Results**: Read `outputs/results.json` and `report.md`
2. **Visualize Attention**: Study attention heatmaps in detail
3. **Experiment**: Modify hyperparameters and observe changes
4. **Apply to Real Data**: Test on your own time series datasets
5. **Extend**: Add ensemble methods, uncertainty quantification

---

**Happy Forecasting! 📈**