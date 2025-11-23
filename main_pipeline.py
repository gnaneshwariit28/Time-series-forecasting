"""
Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms
=============================================================================

Main Pipeline Implementation - Complete Production-Quality Code

Author: Advanced ML Research
Date: November 2025
Python Version: 3.8+

This module implements a comprehensive time series forecasting system featuring:
- Multivariate time series generation with complex patterns
- Robust preprocessing pipeline with feature engineering
- Transformer and LSTM-Attention models
- Hyperparameter optimization using Optuna
- Benchmark comparison with SARIMA
- Attention visualization and interpretation

Installation Requirements:
--------------------------
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn statsmodels optuna scipy

Usage:
------
python main_pipeline.py

Project Structure:
------------------
├── main_pipeline.py          (This file - complete pipeline)
├── requirements.txt          (Dependencies)
├── README.md                 (Project documentation)
├── report.md                 (Technical report)
└── outputs/                  (Generated plots and results)
    ├── attention_weights/
    ├── model_comparison/
    └── training_history/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Any
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Statistical Models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Preprocessing and Optimization
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from scipy import signal

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Create output directories
os.makedirs('outputs/attention_weights', exist_ok=True)
os.makedirs('outputs/model_comparison', exist_ok=True)
os.makedirs('outputs/training_history', exist_ok=True)


# ============================================================================
# SECTION 1: DATA GENERATION WITH COMPLEX PATTERNS
# ============================================================================

class ComplexTimeSeriesGenerator:
    """
    Generates synthetic multivariate time series with realistic complexity including:
    - Multiple seasonalities (daily, weekly, yearly)
    - Non-stationary trends
    - Volatility clustering
    - Cross-correlation between features
    - Structural breaks
    """
    
    def __init__(self, 
                 n_samples: int = 10000,
                 n_features: int = 5,
                 freq: str = 'H'):
        """
        Initialize the complex time series generator.
        
        Args:
            n_samples: Number of time steps (default: 10000 hourly observations)
            n_features: Number of features/variates in the time series
            freq: Frequency string for pandas DatetimeIndex ('H' for hourly)
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.freq = freq
        
    def _generate_arima_component(self, 
                                  ar_params: List[float],
                                  ma_params: List[float],
                                  n_samples: int) -> np.ndarray:
        """
        Generate ARIMA process component.
        
        Args:
            ar_params: AR coefficients
            ma_params: MA coefficients
            n_samples: Number of samples to generate
            
        Returns:
            ARIMA process array
        """
        # Generate white noise
        noise = np.random.normal(0, 1, n_samples + 100)
        
        # Apply MA process
        ma_series = np.convolve(noise, [1] + ma_params, mode='valid')[:n_samples]
        
        # Apply AR process
        ar_series = np.zeros(n_samples)
        for i in range(len(ar_params), n_samples):
            ar_component = sum(ar_params[j] * ar_series[i-j-1] for j in range(len(ar_params)))
            ar_series[i] = ar_component + ma_series[i]
        
        return ar_series
    
    def _add_multiple_seasonalities(self, 
                                   t: np.ndarray,
                                   daily_strength: float = 10.0,
                                   weekly_strength: float = 15.0,
                                   yearly_strength: float = 20.0) -> np.ndarray:
        """
        Add multiple seasonal components.
        
        Args:
            t: Time index array
            daily_strength: Amplitude of daily seasonality
            weekly_strength: Amplitude of weekly seasonality
            yearly_strength: Amplitude of yearly seasonality
            
        Returns:
            Combined seasonal component
        """
        # Daily seasonality (24 hours)
        daily = daily_strength * np.sin(2 * np.pi * t / 24)
        
        # Weekly seasonality (168 hours)
        weekly = weekly_strength * np.sin(2 * np.pi * t / 168)
        weekly += 5 * np.cos(4 * np.pi * t / 168)  # Add harmonic
        
        # Yearly seasonality (8760 hours)
        yearly = yearly_strength * np.sin(2 * np.pi * t / 8760)
        
        return daily + weekly + yearly
    
    def _add_volatility_clustering(self, 
                                   t: np.ndarray,
                                   base_volatility: float = 5.0) -> np.ndarray:
        """
        Add time-varying volatility (GARCH-like effect).
        
        Args:
            t: Time index array
            base_volatility: Base volatility level
            
        Returns:
            Noise with volatility clustering
        """
        # Volatility changes over time
        volatility = base_volatility * (1 + 0.5 * np.sin(2 * np.pi * t / 1000))
        volatility += 0.3 * np.abs(np.sin(2 * np.pi * t / 500))
        
        # Generate clustered noise
        noise = np.random.normal(0, 1, len(t))
        return noise * volatility
    
    def _add_structural_break(self, 
                            series: np.ndarray,
                            break_point: int,
                            shift: float = 20.0) -> np.ndarray:
        """
        Add a structural break (level shift) to the series.
        
        Args:
            series: Input series
            break_point: Time index for the break
            shift: Magnitude of the level shift
            
        Returns:
            Series with structural break
        """
        series_with_break = series.copy()
        series_with_break[break_point:] += shift
        return series_with_break
    
    def generate(self) -> pd.DataFrame:
        """
        Generate the complete complex multivariate time series.
        
        Returns:
            DataFrame with datetime index and multiple feature columns
        """
        print("\n" + "="*80)
        print("GENERATING COMPLEX MULTIVARIATE TIME SERIES")
        print("="*80)
        
        # Create datetime index
        start_date = '2022-01-01'
        date_range = pd.date_range(start=start_date, periods=self.n_samples, freq=self.freq)
        
        # Time components
        t = np.arange(self.n_samples)
        
        # Generate primary target variable
        print("\nGenerating target variable with multiple components...")
        
        # 1. Non-linear trend
        trend = 100 + 0.01 * t + 0.00001 * (t ** 1.5)
        print(f"  ✓ Added non-linear trend")
        
        # 2. Multiple seasonalities
        seasonality = self._add_multiple_seasonalities(t)
        print(f"  ✓ Added daily, weekly, and yearly seasonalities")
        
        # 3. ARIMA component for autocorrelation
        arima_component = self._generate_arima_component(
            ar_params=[0.7, -0.3],
            ma_params=[0.5],
            n_samples=self.n_samples
        ) * 3
        print(f"  ✓ Added ARIMA(2,0,1) autocorrelation structure")
        
        # 4. Volatility clustering
        volatility_noise = self._add_volatility_clustering(t)
        print(f"  ✓ Added time-varying volatility (GARCH-like)")
        
        # 5. Combine components
        target = trend + seasonality + arima_component + volatility_noise
        
        # 6. Add structural break
        break_point = self.n_samples // 2
        target = self._add_structural_break(target, break_point, shift=15)
        print(f"  ✓ Added structural break at sample {break_point}")
        
        # Create DataFrame
        data = {'target': target}
        
        # Generate correlated features
        print(f"\nGenerating {self.n_features - 1} correlated features...")
        
        for i in range(1, self.n_features):
            # Correlation with target
            correlation_strength = np.random.uniform(0.5, 0.85)
            
            # Feature-specific patterns
            feature_seasonality = 8 * np.sin(2 * np.pi * t / (24 * (i + 1)))
            feature_seasonality += 4 * np.cos(2 * np.pi * t / (168 * (i + 1)))
            
            # ARIMA component
            feature_arima = self._generate_arima_component(
                ar_params=[np.random.uniform(0.3, 0.7)],
                ma_params=[np.random.uniform(0.2, 0.5)],
                n_samples=self.n_samples
            ) * 2
            
            # Noise
            feature_noise = np.random.normal(0, 3, self.n_samples)
            
            # Combine with correlation to target
            feature = (correlation_strength * target + 
                      (1 - correlation_strength) * 50 +
                      feature_seasonality + 
                      feature_arima +
                      feature_noise)
            
            data[f'feature_{i}'] = feature
            print(f"  ✓ Generated feature_{i} (correlation with target: {correlation_strength:.2f})")
        
        # Create DataFrame
        df = pd.DataFrame(data, index=date_range)
        
        # Add realistic missing values (3% of data)
        print(f"\nAdding missing values...")
        missing_mask = np.random.random(df.shape) < 0.03
        df = df.mask(missing_mask)
        missing_count = df.isnull().sum().sum()
        print(f"  ✓ Added {missing_count} missing values ({missing_count/df.size*100:.2f}%)")
        
        print(f"\nDataset Summary:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Frequency: {self.freq}")
        print(f"  Features: {df.columns.tolist()}")
        
        return df


# ============================================================================
# SECTION 2: COMPREHENSIVE DATA PREPROCESSING PIPELINE
# ============================================================================

class RobustTimeSeriesPreprocessor:
    """
    Production-quality preprocessing pipeline with:
    - Missing value imputation
    - Feature scaling (Standard/MinMax)
    - Time-based feature engineering
    - Fourier terms for seasonality
    - Lag features
    - Rolling statistics
    """
    
    def __init__(self,
                 scaling_method: str = 'standard',
                 add_fourier: bool = True,
                 fourier_orders: List[int] = None,
                 add_lags: bool = True,
                 lag_periods: List[int] = None,
                 add_rolling: bool = True,
                 rolling_windows: List[int] = None):
        """
        Initialize the preprocessor.
        
        Args:
            scaling_method: 'standard' or 'minmax'
            add_fourier: Whether to add Fourier features
            fourier_orders: Orders for Fourier terms [daily, weekly]
            add_lags: Whether to add lag features
            lag_periods: List of lag periods
            add_rolling: Whether to add rolling statistics
            rolling_windows: List of window sizes for rolling stats
        """
        self.scaling_method = scaling_method
        self.add_fourier = add_fourier
        self.fourier_orders = fourier_orders or [24, 168]  # Daily and weekly
        self.add_lags = add_lags
        self.lag_periods = lag_periods or [1, 24, 168]
        self.add_rolling = add_rolling
        self.rolling_windows = rolling_windows or [24, 168]
        
        self.scalers = {}
        self.feature_names = []
        self.target_scaler = None
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using multiple strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with imputed missing values
        """
        df_filled = df.copy()
        
        # Forward fill then backward fill
        df_filled = df_filled.fillna(method='ffill', limit=3)
        df_filled = df_filled.fillna(method='bfill', limit=3)
        
        # If still missing, use interpolation
        df_filled = df_filled.interpolate(method='time', limit_direction='both')
        
        # Final fallback: use column mean
        df_filled = df_filled.fillna(df_filled.mean())
        
        return df_filled
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive time-based features from datetime index.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Basic time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding (important for maintaining continuity)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fourier term features for capturing complex seasonality.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with Fourier features
        """
        df = df.copy()
        t = np.arange(len(df))
        
        for i, period in enumerate(self.fourier_orders):
            # Multiple orders for each period
            for order in range(1, 4):  # 3 orders per period
                df[f'fourier_sin_{period}_{order}'] = np.sin(2 * np.pi * order * t / period)
                df[f'fourier_cos_{period}_{order}'] = np.cos(2 * np.pi * order * t / period)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """
        Add lag features for the target variable.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in self.lag_periods:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """
        Add rolling window statistics.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using the specified method.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit scalers (True for training data)
            
        Returns:
            Scaled DataFrame
        """
        df_scaled = df.copy()
        
        if fit:
            self.feature_names = df.columns.tolist()
        
        for col in self.feature_names:
            if col not in df_scaled.columns:
                continue
            
            if fit:
                if self.scaling_method == 'standard':
                    scaler = StandardScaler()
                elif self.scaling_method == 'minmax':
                    scaler = MinMaxScaler(feature_range=(0, 1))
                else:
                    raise ValueError(f"Unknown scaling method: {self.scaling_method}")
                
                df_scaled[[col]] = scaler.fit_transform(df_scaled[[col]])
                self.scalers[col] = scaler
                
                # Separate scaler for target
                if col == 'target':
                    self.target_scaler = scaler
            else:
                if col in self.scalers:
                    df_scaled[[col]] = self.scalers[col].transform(df_scaled[[col]])
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("\n" + "="*80)
        print("PREPROCESSING PIPELINE")
        print("="*80)
        
        print(f"\nOriginal shape: {df.shape}")
        
        # Handle missing values
        print("\n1. Handling missing values...")
        df = self.handle_missing_values(df)
        print(f"   ✓ Missing values imputed")
        
        # Extract time features
        print("\n2. Extracting time-based features...")
        df = self.extract_time_features(df)
        print(f"   ✓ Added {len([c for c in df.columns if c not in ['target'] + [f'feature_{i}' for i in range(1, 10)]])} time features")
        
        # Add Fourier features
        if self.add_fourier:
            print("\n3. Adding Fourier term features...")
            df = self.add_fourier_features(df)
            fourier_count = len([c for c in df.columns if 'fourier' in c])
            print(f"   ✓ Added {fourier_count} Fourier features")
        
        # Add lag features
        if self.add_lags:
            print("\n4. Adding lag features...")
            df = self.add_lag_features(df)
            print(f"   ✓ Added lags: {self.lag_periods}")
        
        # Add rolling features
        if self.add_rolling:
            print("\n5. Adding rolling window statistics...")
            df = self.add_rolling_features(df)
            rolling_count = len([c for c in df.columns if 'rolling' in c])
            print(f"   ✓ Added {rolling_count} rolling features")
        
        # Remove rows with NaN from lagging/rolling
        print("\n6. Removing rows with NaN from feature engineering...")
        initial_len = len(df)
        df = df.dropna()
        print(f"   ✓ Removed {initial_len - len(df)} rows")
        
        # Scale features
        print("\n7. Scaling features...")
        df = self.scale_features(df, fit=True)
        print(f"   ✓ Applied {self.scaling_method} scaling")
        
        print(f"\nFinal shape: {df.shape}")
        print(f"Total features: {df.shape[1]}")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Preprocessed DataFrame
        """
        df = self.handle_missing_values(df)
        df = self.extract_time_features(df)
        
        if self.add_fourier:
            df = self.add_fourier_features(df)
        
        if self.add_lags:
            df = self.add_lag_features(df)
        
        if self.add_rolling:
            df = self.add_rolling_features(df)
        
        df = df.dropna()
        df = self.scale_features(df, fit=False)
        
        return df
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform the target variable.
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Original scale target values
        """
        if self.target_scaler is None:
            return y_scaled
        
        # Reshape if needed
        original_shape = y_scaled.shape
        if len(y_scaled.shape) == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        elif len(y_scaled.shape) == 2 and y_scaled.shape[1] > 1:
            # Multiple time steps - flatten, transform, reshape
            y_scaled = y_scaled.reshape(-1, 1)
        
        y_original = self.target_scaler.inverse_transform(y_scaled)
        
        # Restore original shape
        if len(original_shape) == 1:
            return y_original.flatten()
        else:
            return y_original.reshape(original_shape)


# ============================================================================
# SECTION 3: SEQUENCE GENERATION FOR DEEP LEARNING
# ============================================================================

class SequenceGenerator:
    """
    Creates input-output sequences for time series forecasting.
    """
    
    def __init__(self,
                 input_length: int = 168,
                 output_length: int = 24,
                 stride: int = 1):
        """
        Initialize sequence generator.
        
        Args:
            input_length: Number of past time steps (lookback window)
            output_length: Number of future time steps to predict (forecast horizon)
            stride: Step size for sliding window
        """
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        
    def create_sequences(self,
                        data: np.ndarray,
                        target_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences from time series data.
        
        Args:
            data: Input array of shape (n_samples, n_features)
            target_idx: Index of the target variable in the feature array
            
        Returns:
            Tuple of (X, y):
                X: Input sequences of shape (n_sequences, input_length, n_features)
                y: Target sequences of shape (n_sequences, output_length)
        """
        X, y = [], []
        
        for i in range(0, len(data) - self.input_length - self.output_length + 1, self.stride):
            # Input sequence: all features
            X.append(data[i:i + self.input_length])
            
            # Output sequence: only target variable
            y.append(data[i + self.input_length:i + self.input_length + self.output_length, target_idx])
        
        return np.array(X), np.array(y)


# ============================================================================
# SECTION 4: ATTENTION MECHANISMS AND DEEP LEARNING MODELS
# ============================================================================

class MultiHeadSelfAttention(layers.Layer):
    """
    Multi-head self-attention mechanism for time series.
    Implements scaled dot-product attention with multiple heads.
    """
    
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        """
        Initialize multi-head attention layer.
        
        Args:
            d_model: Dimension of the model (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        # Linear projections
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor,
             mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        batch_size = tf.shape(query)[0]
        
        # Linear projections
        query = self.wq(query)  # (batch_size, seq_len, d_model)
        key = self.wk(key)
        value = self.wv(value)
        
        # Split into multiple heads
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, depth)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        
        # Scale
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention weights to values
        attention_output = tf.matmul(attention_weights, value)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(concat_attention)
        
        return output, attention_weights


class TransformerEncoderBlock(layers.Layer):
    """
    Transformer encoder block with multi-head attention and feed-forward network.
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout_rate: float = 0.1,
                 **kwargs):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate
        """
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass of transformer block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)
        
        return output, attn_weights


def build_transformer_model(input_shape: Tuple[int, int],
                           output_steps: int,
                           d_model: int = 64,
                           num_heads: int = 4,
                           ff_dim: int = 128,
                           num_blocks: int = 2,
                           dropout_rate: float = 0.1) -> Model:
    """
    Build a Transformer-based time series forecasting model.
    
    Args:
        input_shape: Shape of input (seq_length, n_features)
        output_steps: Number of future steps to predict
        d_model: Model dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        num_blocks: Number of transformer blocks
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Project input to d_model dimension
    x = layers.Dense(d_model)(inputs)
    
    # Positional encoding (simple learned encoding)
    pos_encoding = layers.Embedding(input_dim=input_shape[0], output_dim=d_model)(
        tf.range(start=0, limit=input_shape[0], delta=1)
    )
    x = x + pos_encoding
    
    # Stack transformer blocks
    attention_weights_list = []
    for _ in range(num_blocks):
        x, attn_weights = TransformerEncoderBlock(
            d_model, num_heads, ff_dim, dropout_rate
        )(x)
        attention_weights_list.append(attn_weights)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(output_steps)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TransformerForecaster')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


class LSTMAttentionLayer(layers.Layer):
    """
    Custom attention layer for LSTM outputs.
    """
    
    def __init__(self, attention_units: int, **kwargs):
        """
        Initialize attention layer.
        
        Args:
            attention_units: Number of attention units
        """
        super(LSTMAttentionLayer, self).__init__(**kwargs)
        self.attention_units = attention_units
        
        self.W = layers.Dense(attention_units, use_bias=False)
        self.U = layers.Dense(attention_units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)
    
    def call(self, hidden_states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute attention-weighted context vector.
        
        Args:
            hidden_states: LSTM hidden states of shape (batch, time_steps, lstm_units)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        # Additive attention (Bahdanau attention)
        # score = V * tanh(W * h)
        score = self.V(tf.nn.tanh(self.W(hidden_states)))
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector
        context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)
        
        return context_vector, attention_weights


def build_lstm_attention_model(input_shape: Tuple[int, int],
                               output_steps: int,
                               lstm_units: int = 128,
                               attention_units: int = 64,
                               dropout_rate: float = 0.2) -> Model:
    """
    Build an LSTM model with explicit attention mechanism.
    
    Args:
        input_shape: Shape of input (seq_length, n_features)
        output_steps: Number of future steps to predict
        lstm_units: Number of LSTM units
        attention_units: Number of attention units
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # First LSTM layer
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    lstm_out = layers.Dropout(dropout_rate)(lstm_out)
    
    # Second LSTM layer
    lstm_out = layers.LSTM(lstm_units // 2, return_sequences=True)(lstm_out)
    lstm_out = layers.Dropout(dropout_rate)(lstm_out)
    
    # Attention layer
    context_vector, attention_weights = LSTMAttentionLayer(attention_units)(lstm_out)
    
    # Output layers
    x = layers.Dense(128, activation='relu')(context_vector)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(output_steps)(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LSTMAttentionForecaster')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


# ============================================================================
# SECTION 5: MODEL TRAINING UTILITIES
# ============================================================================

class ModelTrainer:
    """
    Handles model training with callbacks and monitoring.
    """
    
    def __init__(self, model: Model, model_name: str, patience: int = 15):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            model_name: Name for saving checkpoints
            patience: Patience for early stopping
        """
        self.model = model
        self.model_name = model_name
        self.patience = patience
        self.history = None
        
    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'outputs/{self.model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        return callbacks
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 100,
             batch_size: int = 32) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target sequences
            X_val: Validation input sequences
            y_val: Validation target sequences
            epochs: Maximum number of epochs
            batch_size: Batch size
            
        Returns:
            Training history dictionary
        """
        print(f"\nTraining {self.model_name}...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max epochs: {epochs}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        return self.history.history
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save the figure
        """
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title(f'{self.model_name} - Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(history['mae'], label='Training MAE')
        axes[1].plot(history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title(f'{self.model_name} - Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# ============================================================================
# SECTION 6: EVALUATION METRICS
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # MAPE (avoiding division by zero)
    mask = y_true_flat != 0
    mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    
    # SMAPE (Symmetric MAPE)
    smape = np.mean(2.0 * np.abs(y_pred_flat - y_true_flat) / 
                    (np.abs(y_true_flat) + np.abs(y_pred_flat))) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'SMAPE': smape
    }


def print_metrics(metrics: Dict[str, float], model_name: str):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{model_name} Performance Metrics:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"  {metric:10s}: {value:10.4f}")
    print("=" * 50)


# ============================================================================
# SECTION 7: BENCHMARK MODEL (SARIMA)
# ============================================================================

class SARIMABenchmark:
    """
    SARIMA model for benchmarking.
    """
    
    def __init__(self,
                 order: Tuple[int, int, int] = (2, 1, 2),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24)):
        """
        Initialize SARIMA model.
        
        Args:
            order: (p, d, q) order of the model
            seasonal_order: (P, D, Q, s) seasonal order
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def fit(self, train_data: pd.Series):
        """
        Fit SARIMA model.
        
        Args:
            train_data: Training time series
        """
        print(f"\nFitting SARIMA{self.order}x{self.seasonal_order}...")
        
        self.model = SARIMAX(
            train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(disp=False, maxiter=200)
        print(f"  ✓ SARIMA model fitted successfully")
        
    def forecast(self, steps: int) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecast array
        """
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values
    
    def predict_sequences(self, n_sequences: int, steps_per_sequence: int) -> np.ndarray:
        """
        Generate multiple forecast sequences.
        
        Args:
            n_sequences: Number of sequences to generate
            steps_per_sequence: Steps per sequence
            
        Returns:
            Array of predictions
        """
        predictions = []
        total_steps = n_sequences * steps_per_sequence
        forecast = self.forecast(total_steps)
        
        for i in range(n_sequences):
            start_idx = i * steps_per_sequence
            end_idx = start_idx + steps_per_sequence
            predictions.append(forecast[start_idx:end_idx])
        
        return np.array(predictions)


# ============================================================================
# SECTION 8: HYPERPARAMETER OPTIMIZATION
# ============================================================================

class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    """
    
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 model_type: str = 'transformer'):
        """
        Initialize optimizer.
        
        Args:
            X_train: Training input
            y_train: Training target
            X_val: Validation input
            y_val: Validation target
            model_type: 'transformer' or 'lstm_attention'
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_type = model_type
        self.best_params = None
        
    def objective_transformer(self, trial: optuna.Trial) -> float:
        """
        Objective function for transformer optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss
        """
        # Suggest hyperparameters
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        ff_dim = trial.suggest_categorical('ff_dim', [64, 128, 256])
        num_blocks = trial.suggest_int('num_blocks', 1, 3)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        
        # Build model
        model = build_transformer_model(
            input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
            output_steps=self.y_train.shape[1],
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate
        )
        
        # Update learning rate
        model.optimizer.learning_rate.assign(learning_rate)
        
        # Train
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=50,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Return best validation loss
        return min(history.history['val_loss'])
    
    def objective_lstm_attention(self, trial: optuna.Trial) -> float:
        """
        Objective function for LSTM-attention optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss
        """
        # Suggest hyperparameters
        lstm_units = trial.suggest_categorical('lstm_units', [64, 128, 256])
        attention_units = trial.suggest_categorical('attention_units', [32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        
        # Build model
        model = build_lstm_attention_model(
            input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
            output_steps=self.y_train.shape[1],
            lstm_units=lstm_units,
            attention_units=attention_units,
            dropout_rate=dropout_rate
        )
        
        # Update learning rate
        model.optimizer.learning_rate.assign(learning_rate)
        
        # Train
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=50,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        return min(history.history['val_loss'])
    
    def optimize(self, n_trials: int = 30) -> Dict:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials
            
        Returns:
            Best hyperparameters
        """
        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER OPTIMIZATION - {self.model_type.upper()}")
        print(f"{'='*80}")
        print(f"Number of trials: {n_trials}")
        
        # Create study
        study = optuna.create_study(direction='minimize')
        
        # Run optimization
        if self.model_type == 'transformer':
            study.optimize(self.objective_transformer, n_trials=n_trials, show_progress_bar=True)
        else:
            study.optimize(self.objective_lstm_attention, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        print(f"\nOptimization completed!")
        print(f"Best validation loss: {study.best_value:.6f}")
        print(f"\nBest hyperparameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        return self.best_params


# ============================================================================
# SECTION 9: ATTENTION VISUALIZATION
# ============================================================================

class AttentionVisualizer:
    """
    Extract and visualize attention weights.
    """
    
    def __init__(self, model: Model):
        """
        Initialize visualizer.
        
        Args:
            model: Trained model with attention
        """
        self.model = model
        
    def extract_attention_weights(self, X_sample: np.ndarray) -> List[np.ndarray]:
        """
        Extract attention weights from model.
        
        Args:
            X_sample: Input sample (single sequence)
            
        Returns:
            List of attention weight arrays
        """
        # Ensure input is batched
        if len(X_sample.shape) == 2:
            X_sample = np.expand_dims(X_sample, axis=0)
        
        # Create attention extraction model
        attention_layers = []
        for layer in self.model.layers:
            if isinstance(layer, (TransformerEncoderBlock, LSTMAttentionLayer)):
                attention_layers.append(layer)
        
        if not attention_layers:
            print("No attention layers found in model")
            return []
        
        # Build functional model to extract attention
        attention_outputs = []
        x = self.model.input
        
        for layer in self.model.layers:
            if isinstance(layer, TransformerEncoderBlock):
                x, attn = layer(x)
                attention_outputs.append(attn)
            else:
                if hasattr(layer, 'call'):
                    try:
                        x = layer(x)
                    except:
                        pass
        
        if attention_outputs:
            attention_model = Model(inputs=self.model.input, outputs=attention_outputs)
            weights = attention_model.predict(X_sample, verbose=0)
            return weights if isinstance(weights, list) else [weights]
        
        return []
    
    def plot_attention_heatmap(self,
                              attention_weights: np.ndarray,
                              sample_idx: int = 0,
                              head_idx: int = 0,
                              save_path: Optional[str] = None):
        """
        Plot attention weights as heatmap.
        
        Args:
            attention_weights: Attention weight tensor
            sample_idx: Sample index to visualize
            head_idx: Attention head index
            save_path: Optional save path
        """
        # Extract weights for specific sample and head
        if len(attention_weights.shape) == 4:
            # (batch, heads, seq, seq)
            weights = attention_weights[sample_idx, head_idx, :, :]
        elif len(attention_weights.shape) == 3:
            # (batch, seq, seq)
            weights = attention_weights[sample_idx, :, :]
        else:
            weights = attention_weights
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(weights, cmap='YlOrRd', cbar=True, square=True)
        plt.xlabel('Key Position (Past Time Steps)', fontsize=12)
        plt.ylabel('Query Position (Time Steps)', fontsize=12)
        plt.title(f'Attention Weights Heatmap - Head {head_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attention_over_time(self,
                                attention_weights: np.ndarray,
                                time_steps: Optional[List[int]] = None,
                                save_path: Optional[str] = None):
        """
        Plot how attention distributes over different time steps.
        
        Args:
            attention_weights: Attention weights
            time_steps: Optional list of time step labels
            save_path: Optional save path
        """
        if len(attention_weights.shape) == 4:
            # Average over batch and heads
            weights = attention_weights.mean(axis=(0, 1))
        elif len(attention_weights.shape) == 3:
            weights = attention_weights.mean(axis=0)
        else:
            weights = attention_weights
        
        # Average attention received by each position
        attention_per_position = weights.mean(axis=0)
        
        plt.figure(figsize=(14, 6))
        
        if time_steps is None:
            time_steps = list(range(len(attention_per_position)))
        
        plt.bar(time_steps, attention_per_position, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Time Step (Lookback Position)', fontsize=12)
        plt.ylabel('Average Attention Weight', fontsize=12)
        plt.title('Average Attention Distribution Over Past Time Steps', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def interpret_attention_patterns(self,
                                    attention_weights: np.ndarray,
                                    lookback_window: int,
                                    forecast_horizon: int) -> str:
        """
        Generate textual interpretation of attention patterns.
        
        Args:
            attention_weights: Attention weights
            lookback_window: Input sequence length
            forecast_horizon: Output sequence length
            
        Returns:
            Interpretation string
        """
        # Average over batches and heads if needed
        if len(attention_weights.shape) == 4:
            weights = attention_weights.mean(axis=(0, 1))
        elif len(attention_weights.shape) == 3:
            weights = attention_weights.mean(axis=0)
        else:
            weights = attention_weights
        
        # Analyze attention distribution
        attention_per_position = weights.mean(axis=0)
        
        # Find most attended positions
        top_k = min(5, len(attention_per_position))
        top_indices = np.argsort(attention_per_position)[-top_k:][::-1]
        
        interpretation = []
        interpretation.append(f"\n{'='*80}")
        interpretation.append("ATTENTION PATTERN INTERPRETATION")
        interpretation.append(f"{'='*80}\n")
        
        interpretation.append(f"Lookback window: {lookback_window} time steps")
        interpretation.append(f"Forecast horizon: {forecast_horizon} time steps\n")
        
        interpretation.append(f"Top {top_k} most attended past time steps:")
        for i, idx in enumerate(top_indices, 1):
            attention_score = attention_per_position[idx]
            time_ago = lookback_window - idx
            interpretation.append(f"  {i}. Position {idx} ({time_ago} steps ago): "
                                f"{attention_score:.4f} attention weight")
        
        # Analyze temporal patterns
        recent_attention = attention_per_position[-24:].mean() if len(attention_per_position) >= 24 else 0
        distant_attention = attention_per_position[:-24].mean() if len(attention_per_position) >= 24 else 0
        
        interpretation.append(f"\nTemporal attention distribution:")
        interpretation.append(f"  Recent past (last 24 steps): {recent_attention:.4f}")
        interpretation.append(f"  Distant past (beyond 24 steps): {distant_attention:.4f}")
        
        if recent_attention > distant_attention * 1.2:
            interpretation.append(f"\n  → Model shows RECENCY BIAS: Recent observations heavily influence predictions")
        elif distant_attention > recent_attention * 1.2:
            interpretation.append(f"\n  → Model shows LONG-TERM DEPENDENCY: Distant past is more influential")
        else:
            interpretation.append(f"\n  → Model shows BALANCED attention across time")
        
        # Check for periodic patterns
        if len(attention_per_position) >= 24:
            daily_positions = attention_per_position[::24]
            if len(daily_positions) > 1:
                daily_attention = daily_positions.mean()
                interpretation.append(f"\nAttention to 24-hour cycles (daily patterns): {daily_attention:.4f}")
        
        if len(attention_per_position) >= 168:
            weekly_positions = attention_per_position[::168]
            if len(weekly_positions) > 1:
                weekly_attention = weekly_positions.mean()
                interpretation.append(f"Attention to 168-hour cycles (weekly patterns): {weekly_attention:.4f}")
        
        return '\n'.join(interpretation)


# ============================================================================
# SECTION 10: VISUALIZATION UTILITIES
# ============================================================================

def plot_predictions_comparison(y_true: np.ndarray,
                               y_pred_dl: np.ndarray,
                               y_pred_sarima: np.ndarray,
                               n_samples: int = 3,
                               save_path: Optional[str] = None):
    """
    Plot prediction comparisons for sample sequences.
    
    Args:
        y_true: True values
        y_pred_dl: Deep learning predictions
        y_pred_sarima: SARIMA predictions
        n_samples: Number of samples to plot
        save_path: Optional save path
    """
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 4*n_samples))
    
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        axes[i].plot(y_true[i], 'o-', label='True', linewidth=2, markersize=6)
        axes[i].plot(y_pred_dl[i], 's--', label='Deep Learning', linewidth=2, markersize=6)
        axes[i].plot(y_pred_sarima[i], '^:', label='SARIMA', linewidth=2, markersize=6)
        axes[i].set_xlabel('Forecast Horizon (Steps)', fontsize=11)
        axes[i].set_ylabel('Value', fontsize=11)
        axes[i].set_title(f'Forecast Comparison - Sample {i+1}', fontsize=12, fontweight='bold')
        axes[i].legend(loc='best', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(metrics_dl: Dict[str, float],
                           metrics_sarima: Dict[str, float],
                           save_path: Optional[str] = None):
    """
    Plot bar chart comparing metrics between models.
    
    Args:
        metrics_dl: Deep learning model metrics
        metrics_sarima: SARIMA model metrics
        save_path: Optional save path
    """
    metrics_to_plot = ['MAE', 'RMSE', 'MAPE']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dl_values = [metrics_dl[m] for m in metrics_to_plot]
    sarima_values = [metrics_sarima[m] for m in metrics_to_plot]
    
    bars1 = ax.bar(x - width/2, dl_values, width, label='Deep Learning', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, sarima_values, width, label='SARIMA', color='coral', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_dataset_characteristics(df: pd.DataFrame, save_dir: str = 'outputs'):
    """
    Perform statistical analysis and visualization of the dataset.
    
    Args:
        df: Input DataFrame
        save_dir: Directory to save plots
    """
    print("\n" + "="*80)
    print("DATASET CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Stationarity test (ADF test)
    print("\nAugmented Dickey-Fuller Test (Stationarity):")
    adf_result = adfuller(df['target'].dropna())
    print(f"  ADF Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print("  → Series is STATIONARY (p < 0.05)")
    else:
        print("  → Series is NON-STATIONARY (p >= 0.05)")
    
    # Visualizations
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. Time series plot
    axes[0, 0].plot(df.index, df['target'], linewidth=0.8)
    axes[0, 0].set_title('Target Time Series', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution
    axes[0, 1].hist(df['target'].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Target Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Seasonal decomposition (first 1000 points for speed)
    decomposition = seasonal_decompose(df['target'][:1000].dropna(), model='additive', period=24)
    
    axes[1, 0].plot(decomposition.trend, linewidth=1)
    axes[1, 0].set_title('Trend Component', fontweight='bold')
    axes[1, 0].set_ylabel('Trend')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(decomposition.seasonal, linewidth=1)
    axes[1, 1].set_title('Seasonal Component (Period=24)', fontweight='bold')
    axes[1, 1].set_ylabel('Seasonal')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 4. ACF plot
    plot_acf(df['target'].dropna(), lags=72, ax=axes[2, 0])
    axes[2, 0].set_title('Autocorrelation Function', fontweight='bold')
    
    # 5. Correlation heatmap
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=axes[2, 1])
    axes[2, 1].set_title('Feature Correlation Matrix', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Analysis plots saved to {save_dir}/dataset_analysis.png")


# ============================================================================
# SECTION 11: MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """
    Main execution pipeline for the entire forecasting project.
    """
    print("\n" + "="*80)
    print(" " * 10 + "ADVANCED TIME SERIES FORECASTING WITH ATTENTION")
    print(" " * 20 + "Complete Production Pipeline")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'n_samples': 10000,
        'n_features': 5,
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'input_length': 168,  # 1 week
        'output_length': 24,  # 1 day
        'epochs': 100,
        'batch_size': 32,
        'optimize_hyperparams': True,
        'n_trials': 20,
    }
    
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # ========================================================================
    # STEP 1: DATA GENERATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: DATA GENERATION")
    print("="*80)
    
    generator = ComplexTimeSeriesGenerator(
        n_samples=CONFIG['n_samples'],
        n_features=CONFIG['n_features'],
        freq='H'
    )
    df = generator.generate()
    
    # Analyze dataset characteristics
    analyze_dataset_characteristics(df, save_dir='outputs')
    
    # ========================================================================
    # STEP 2: DATA PREPROCESSING
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80)
    
    # Split data
    train_size = int(len(df) * CONFIG['train_ratio'])
    val_size = int(len(df) * CONFIG['val_ratio'])
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()
    
    print(f"\nData splits:")
    print(f"  Training: {len(train_df)} samples ({CONFIG['train_ratio']*100:.0f}%)")
    print(f"  Validation: {len(val_df)} samples ({CONFIG['val_ratio']*100:.0f}%)")
    print(f"  Test: {len(test_df)} samples ({CONFIG['test_ratio']*100:.0f}%)")
    
    # Preprocess
    preprocessor = RobustTimeSeriesPreprocessor(
        scaling_method='standard',
        add_fourier=True,
        fourier_orders=[24, 168],
        add_lags=True,
        lag_periods=[1, 24, 168],
        add_rolling=True,
        rolling_windows=[24, 168]
    )
    
    train_processed = preprocessor.fit_transform(train_df)
    val_processed = preprocessor.transform(val_df)
    test_processed = preprocessor.transform(test_df)
    
    # ========================================================================
    # STEP 3: SEQUENCE GENERATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: SEQUENCE GENERATION")
    print("="*80)
    
    seq_gen = SequenceGenerator(
        input_length=CONFIG['input_length'],
        output_length=CONFIG['output_length'],
        stride=1
    )
    
    X_train, y_train = seq_gen.create_sequences(train_processed.values, target_idx=0)
    X_val, y_val = seq_gen.create_sequences(val_processed.values, target_idx=0)
    X_test, y_test = seq_gen.create_sequences(test_processed.values, target_idx=0)
    
    print(f"\nSequence shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # ========================================================================
    # STEP 4: HYPERPARAMETER OPTIMIZATION (Optional)
    # ========================================================================
    
    best_params_transformer = None
    
    if CONFIG['optimize_hyperparams']:
        print("\n" + "="*80)
        print("STEP 4: HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        optimizer = HyperparameterOptimizer(
            X_train[:1000], y_train[:1000],  # Use subset for speed
            X_val[:200], y_val[:200],
            model_type='transformer'
        )
        
        best_params_transformer = optimizer.optimize(n_trials=CONFIG['n_trials'])
    else:
        print("\n" + "="*80)
        print("STEP 4: HYPERPARAMETER OPTIMIZATION (SKIPPED)")
        print("="*80)
        print("Using default hyperparameters")
        
        best_params_transformer = {
            'd_model': 64,
            'num_heads': 4,
            'ff_dim': 128,
            'num_blocks': 2,
            'dropout_rate': 0.2,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    
    # ========================================================================
    # STEP 5: BUILD AND TRAIN DEEP LEARNING MODEL
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 5: DEEP LEARNING MODEL TRAINING")
    print("="*80)
    
    # Build Transformer model with optimized hyperparameters
    transformer_model = build_transformer_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_steps=y_train.shape[1],
        d_model=best_params_transformer['d_model'],
        num_heads=best_params_transformer['num_heads'],
        ff_dim=best_params_transformer['ff_dim'],
        num_blocks=best_params_transformer['num_blocks'],
        dropout_rate=best_params_transformer['dropout_rate']
    )
    
    # Update learning rate
    transformer_model.optimizer.learning_rate.assign(best_params_transformer['learning_rate'])
    
    print("\nTransformer Model Architecture:")
    transformer_model.summary()
    
    # Train
    trainer = ModelTrainer(transformer_model, 'transformer', patience=20)
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=CONFIG['epochs'],
        batch_size=best_params_transformer['batch_size']
    )
    
    # Plot training history
    trainer.plot_training_history(save_path='outputs/training_history/transformer_history.png')
    
    # ========================================================================
    # STEP 6: BUILD AND TRAIN BENCHMARK MODEL (SARIMA)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 6: BENCHMARK MODEL (SARIMA)")
    print("="*80)
    
    # Use original unscaled data for SARIMA
    sarima = SARIMABenchmark(order=(2, 1, 2), seasonal_order=(1, 1, 1, 24))
    sarima.fit(train_df['target'])
    
    # ========================================================================
    # STEP 7: EVALUATION AND COMPARISON
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 7: MODEL EVALUATION")
    print("="*80)
    
    # Deep Learning predictions
    y_pred_dl_scaled = transformer_model.predict(X_test, verbose=0)
    
    # Inverse transform to original scale
    y_pred_dl = preprocessor.inverse_transform_target(y_pred_dl_scaled)
    y_test_original = preprocessor.inverse_transform_target(y_test)
    
    # SARIMA predictions
    y_pred_sarima = sarima.predict_sequences(len(y_test), CONFIG['output_length'])
    
    # Compute metrics
    metrics_dl = compute_metrics(y_test_original, y_pred_dl)
    metrics_sarima = compute_metrics(y_test_original, y_pred_sarima)
    
    # Print metrics
    print_metrics(metrics_dl, "TRANSFORMER MODEL")
    print_metrics(metrics_sarima, "SARIMA BENCHMARK")
    
    # Calculate improvement
    print(f"\n{'='*80}")
    print("PERFORMANCE IMPROVEMENT")
    print(f"{'='*80}")
    
    for metric in ['MAE', 'RMSE', 'MAPE']:
        improvement = ((metrics_sarima[metric] - metrics_dl[metric]) / metrics_sarima[metric]) * 100
        print(f"{metric} improvement: {improvement:+.2f}%")
    
    # Visualize comparisons
    plot_predictions_comparison(
        y_test_original, y_pred_dl, y_pred_sarima,
        n_samples=3,
        save_path='outputs/model_comparison/predictions_comparison.png'
    )
    
    plot_metrics_comparison(
        metrics_dl, metrics_sarima,
        save_path='outputs/model_comparison/metrics_comparison.png'
    )
    
    # ========================================================================
    # STEP 8: ATTENTION VISUALIZATION AND INTERPRETATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 8: ATTENTION VISUALIZATION")
    print("="*80)
    
    visualizer = AttentionVisualizer(transformer_model)
    
    # Extract attention for a few test samples
    sample_indices = [0, len(X_test)//2, -1]
    
    for i, idx in enumerate(sample_indices):
        print(f"\nAnalyzing sample {i+1}/{len(sample_indices)}...")
        
        attention_weights = visualizer.extract_attention_weights(X_test[idx:idx+1])
        
        if attention_weights:
            # Plot heatmap
            visualizer.plot_attention_heatmap(
                attention_weights[0],
                sample_idx=0,
                head_idx=0,
                save_path=f'outputs/attention_weights/attention_heatmap_sample_{i+1}.png'
            )
            
            # Plot attention distribution
            visualizer.plot_attention_over_time(
                attention_weights[0],
                save_path=f'outputs/attention_weights/attention_distribution_sample_{i+1}.png'
            )
    
    # Generate interpretation
    if attention_weights:
        interpretation = visualizer.interpret_attention_patterns(
            attention_weights[0],
            lookback_window=CONFIG['input_length'],
            forecast_horizon=CONFIG['output_length']
        )
        print(interpretation)
        
        # Save interpretation
        with open('outputs/attention_weights/attention_interpretation.txt', 'w') as f:
            f.write(interpretation)
    
    # ========================================================================
    # STEP 9: SAVE RESULTS AND MODELS
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 9: SAVING RESULTS")
    print("="*80)
    
    # Save model
    transformer_model.save('outputs/transformer_model.h5')
    print("✓ Model saved to outputs/transformer_model.h5")
    
    # Save metrics
    results = {
        'config': CONFIG,
        'best_hyperparameters': best_params_transformer,
        'transformer_metrics': metrics_dl,
        'sarima_metrics': metrics_sarima,
        'training_history': {
            'loss': [float(x) for x in history['loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'mae': [float(x) for x in history['mae']],
            'val_mae': [float(x) for x in history['val_mae']]
        }
    }
    
    with open('outputs/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved to outputs/results.json")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print("\nGenerated Outputs:")
    print("  📁 outputs/")
    print("    ├── dataset_analysis.png")
    print("    ├── transformer_model.h5")
    print("    ├── results.json")
    print("    ├── training_history/")
    print("    │   └── transformer_history.png")
    print("    ├── model_comparison/")
    print("    │   ├── predictions_comparison.png")
    print("    │   └── metrics_comparison.png")
    print("    └── attention_weights/")
    print("        ├── attention_heatmap_sample_*.png")
    print("        ├── attention_distribution_sample_*.png")
    print("        └── attention_interpretation.txt")
    
    print("\n" + "="*80)
    print("All deliverables have been generated successfully!")
    print("="*80 + "\n")
    
    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    results = main()
    
    print("\n" + "="*80)
    print("To view the technical report and documentation, please check:")
    print("  - README.md (Project overview and setup instructions)")
    print("  - report.md (Detailed technical report)")
    print("="*80)
