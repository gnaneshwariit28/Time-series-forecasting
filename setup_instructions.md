# Setup and Execution Instructions

## 📦 Complete Project Package

This document provides step-by-step instructions to set up and run the Advanced Time Series Forecasting project.

---

## 🗂️ Project Files

Your project folder should contain:

```
time-series-forecasting/
├── main_pipeline.py          ⭐ Main implementation (RUN THIS)
├── requirements.txt          📋 Dependencies
├── README.md                 📖 Project overview
├── report.md                 📊 Technical report
├── SETUP_INSTRUCTIONS.md     🛠️ This file
└── outputs/                  📁 (Created automatically)
```

---

## ⚙️ Installation Steps

### Step 1: Verify Python Installation

**Check your Python version**:
```bash
python --version
```

**Required**: Python 3.8 or higher

If you don't have Python installed:
- **Windows**: Download from https://www.python.org/downloads/
- **macOS**: `brew install python@3.11`
- **Linux**: `sudo apt-get install python3.11`

### Step 2: Create Virtual Environment (Recommended)

**Why?** Isolates project dependencies from your system Python.

**On Windows**:
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

**On macOS/Linux**:
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

**You'll see** `(venv)` prefix in your terminal when activated.

### Step 3: Install Dependencies

**Method 1: Using requirements.txt** (Recommended):
```bash
pip install -r requirements.txt
```

**Method 2: Manual installation**:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn statsmodels optuna scipy
```

**Verify installation**:
```bash
pip list
```

You should see all packages listed with versions.

### Step 4: Handle Common Installation Issues

**Issue: TensorFlow installation fails**

**Solution**:
```bash
# For CPU-only (works on all systems)
pip install tensorflow-cpu

# For GPU (requires CUDA)
pip install tensorflow[and-cuda]
```

**Issue: "No module named 'xyz'"**

**Solution**:
```bash
pip install xyz --upgrade
```

**Issue: Permission denied**

**Solution**:
```bash
pip install --user -r requirements.txt
```

---

## 🚀 Running the Project

### Quick Start (Default Configuration)

```bash
python main_pipeline.py
```

**What happens**:
1. ✅ Generates synthetic time series (10,000 samples)
2. ✅ Preprocesses data with feature engineering
3. ✅ Creates sequences for deep learning
4. ✅ Optimizes hyperparameters (20 trials)
5. ✅ Trains Transformer model
6. ✅ Trains SARIMA benchmark
7. ✅ Evaluates and compares models
8. ✅ Visualizes attention mechanisms
9. ✅ Saves results to `outputs/` folder

**Expected Runtime**: 30-60 minutes (varies by hardware)

### Output Directory Structure

After successful run:

```
outputs/
├── dataset_analysis.png              # Data visualizations
├── results.json                      # Performance metrics
├── transformer_model.h5              # Saved model
├── transformer_best.h5               # Best checkpoint
├── training_history/
│   └── transformer_history.png       # Training curves
├── model_comparison/
│   ├── predictions_comparison.png    # Forecast plots
│   └── metrics_comparison.png        # Bar chart comparison
└── attention_weights/
    ├── attention_heatmap_sample_1.png
    ├── attention_heatmap_sample_2.png
    ├── attention_heatmap_sample_3.png
    ├── attention_distribution_sample_1.png
    ├── attention_distribution_sample_2.png
    ├── attention_distribution_sample_3.png
    └── attention_interpretation.txt   # Written analysis
```

---

## 🎛️ Configuration Options

### Modify Settings

Open `main_pipeline.py` and find the `CONFIG` dictionary:

```python
CONFIG = {
    'n_samples': 10000,           # Total time steps
    'n_features': 5,              # Number of features
    'train_ratio': 0.70,          # 70% training data
    'val_ratio': 0.15,            # 15% validation
    'test_ratio': 0.15,           # 15% test
    'input_length': 168,          # 1 week lookback
    'output_length': 24,          # 1 day forecast
    'epochs': 100,                # Max training epochs
    'batch_size': 32,             # Batch size
    'optimize_hyperparams': True, # Enable tuning
    'n_trials': 20,               # Optimization trials
}
```

### Quick Testing (Faster Run)

For development/testing, use smaller configuration:

```python
CONFIG = {
    'n_samples': 2000,            # Reduced dataset
    'n_features': 3,              # Fewer features
    'epochs': 30,                 # Fewer epochs
    'optimize_hyperparams': False,# Skip optimization
    'n_trials': 5,                # Fewer trials if enabled
}
```

**Runtime**: ~10-15 minutes

### Production Settings (Maximum Quality)

For best performance:

```python
CONFIG = {
    'n_samples': 20000,           # More data
    'n_features': 8,              # Richer features
    'epochs': 150,                # More training
    'optimize_hyperparams': True,
    'n_trials': 50,               # Extensive search
}
```

**Runtime**: 2-3 hours

---

## 🖥️ Hardware Considerations

### CPU-Only Setup

If you don't have a GPU:

**1. Adjust batch size** (in CONFIG):
```python
'batch_size': 16,  # Smaller for CPU
```

**2. Reduce model size** (in code, line ~800):
```python
transformer_model = build_transformer_model(
    d_model=32,      # Reduced from 64
    num_heads=2,     # Reduced from 4
    ff_dim=64,       # Reduced from 128
)
```

**Expected Runtime**: 60-90 minutes

### GPU Setup

**Prerequisites**:
- NVIDIA GPU (CUDA compatible)
- CUDA Toolkit installed
- cuDNN installed

**Verify GPU**:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Expected Output**: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

**If GPU not detected**:
```bash
# Install GPU version
pip install tensorflow[and-cuda]
```

---

## 📊 Viewing Results

### 1. Visualizations

**Open PNG files** in `outputs/` folder:
- Use any image viewer
- Or open in browser: `file:///path/to/outputs/dataset_analysis.png`

### 2. Metrics (JSON)

**View results.json**:
```bash
# Pretty print
python -c "import json; print(json.dumps(json.load(open('outputs/results.json')), indent=2))"
```

### 3. Attention Interpretation

**Read the analysis**:
```bash
cat outputs/attention_weights/attention_interpretation.txt
```

Or open in text editor.

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution**:
```bash
pip install tensorflow
```

### Problem: "Out of Memory Error"

**Solutions**:
1. Reduce batch size: `'batch_size': 16`
2. Reduce dataset: `'n_samples': 5000`
3. Reduce model size: `d_model=32`

### Problem: "Training is very slow"

**Solutions**:
1. Set `'optimize_hyperparams': False` (skips 20-trial search)
2. Reduce epochs: `'epochs': 50`
3. Use GPU if available

### Problem: "No such file or directory: outputs/"

**Solution**: Directory is created automatically, but if missing:
```bash
mkdir -p outputs/attention_weights outputs/model_comparison outputs/training_history
```

### Problem: Script exits with error

**Debug mode**:
```bash
python -u main_pipeline.py 2>&1 | tee run.log
```

This saves all output to `run.log` for inspection.

---

## ✅ Verification Checklist

After successful run, verify:

- [ ] `outputs/results.json` exists and contains metrics
- [ ] `outputs/transformer_model.h5` exists (model saved)
- [ ] At least 10 PNG files in `outputs/` subdirectories
- [ ] `attention_interpretation.txt` contains text analysis
- [ ] Console shows "PROJECT EXECUTION COMPLETED SUCCESSFULLY"

---

## 🔄 Running Multiple Experiments

### Experiment 1: Compare configurations

**Run 1** (Baseline):
```python
CONFIG = {'optimize_hyperparams': False}
```

**Run 2** (Optimized):
```python
CONFIG = {'optimize_hyperparams': True, 'n_trials': 30}
```

**Compare**: Check `outputs/results.json` for both runs.

### Experiment 2: Different architectures

**Modify** `main_pipeline.py` line ~850:

**Option A: Transformer** (default)
```python
transformer_model = build_transformer_model(...)
```

**Option B: LSTM-Attention**
```python
transformer_model = build_lstm_attention_model(...)
```

---

## 📚 Next Steps

### 1. Review Outputs

- **Technical Report**: Read `report.md` for detailed analysis
- **Visualizations**: Examine attention heatmaps
- **Metrics**: Compare Transformer vs. SARIMA performance

### 2. Experiment

- Modify hyperparameters in CONFIG
- Try different architectures
- Add more features
- Test on your own data

### 3. Use Your Data

**Replace data generation** in `main_pipeline.py` (around line 700):

```python
# Comment out synthetic generation
# generator = ComplexTimeSeriesGenerator(...)
# df = generator.generate()

# Load your data
df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# Ensure columns: 'target', 'feature_1', 'feature_2', ...
```

---

## 🆘 Getting Help

### Common Questions

**Q: Can I run this on Google Colab?**  
A: Yes! Upload `main_pipeline.py` and run in a notebook cell:
```python
!pip install -r requirements.txt
!python main_pipeline.py
```

**Q: How do I use the trained model for new predictions?**  
A: Load the model:
```python
from tensorflow import keras
model = keras.models.load_model('outputs/transformer_model.h5')
predictions = model.predict(new_data)
```

**Q: What if I want to forecast a different horizon?**  
A: Change in CONFIG:
```python
'output_length': 48,  # 2 days instead of 1
```

---

## 📞 Support

For issues:
1. Check `TROUBLESHOOTING` section above
2. Review error messages carefully
3. Verify all dependencies installed
4. Check Python version (3.8+)

---

## ✨ Success Indicators

You know it worked when:

✅ Console shows step-by-step progress  
✅ No red error messages  
✅ `outputs/` folder populated with files  
✅ Final message: "PROJECT EXECUTION COMPLETED SUCCESSFULLY"  
✅ Metrics show Transformer > SARIMA performance  

---

## 🎉 You're Ready!

**To start**:
```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Run the pipeline
python main_pipeline.py

# Wait for completion (~30-60 min)

# Review outputs in outputs/ folder
```

**Happy Forecasting! 📈**

---

**Document Version**: 1.0  
**Last Updated**: November 2025