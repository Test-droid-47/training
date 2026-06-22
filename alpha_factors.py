import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

# Optional numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(func, nopython=False):
        return func

class AlphaFactorEngine:
    
    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.window = self.cfg.get('hurst_window', 100)
        self.er_windows = self.cfg.get('efficiency_windows', [10, 20, 50])
        self.zscore_windows = self.cfg.get('zscore_windows', [10, 20, 50])

    # ------------------------------------------------------------------
    # Optimized Hurst with Numba JIT and correct slice
    # ------------------------------------------------------------------
    @staticmethod
    @jit(nopython=True)  # Compile to machine code for speed
    def _hurst_rs_vectorized(price_series: np.ndarray, window: int) -> np.ndarray:
        n = len(price_series)
        hurst = np.full(n, 0.5, dtype=np.float64)
        if n < window + 10:
            return hurst
        
        log_prices = np.log(price_series)
        
        for i in range(window, n):
            # Slice includes current bar (i)
            segment = log_prices[i-window+1:i+1]
            mean_centered = segment - np.mean(segment)
            cumsum = np.cumsum(mean_centered)
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(segment, ddof=1)
            if s > 1e-10:
                rs = r / s
                h = np.log(rs) / np.log(window)
                hurst[i] = max(0.0, min(1.0, h))
        return hurst

    def add_hurst_fractal(self, df: pd.DataFrame) -> pd.DataFrame:
        close_arr = df['close'].values.astype(float)
        n = len(close_arr)
        window = min(self.window, n // 2)
        if window < 10:
            df['hurst_exp'] = 0.5
            df['fractal_dim'] = 1.5
            df['market_memory'] = 0.0
            return df
        
        hurst = self._hurst_rs_vectorized(close_arr, window)
        
        df['hurst_exp'] = hurst.astype(np.float32)
        df['fractal_dim'] = (2.0 - hurst).astype(np.float32)
        df['market_memory'] = (hurst - 0.5).astype(np.float32)
        
        h_series = pd.Series(hurst)
        df['hurst_mean_20'] = h_series.rolling(20).mean().values.astype(np.float32)
        df['hurst_std_20'] = h_series.rolling(20).std().values.astype(np.float32)
        df['hurst_mean_50'] = h_series.rolling(50).mean().values.astype(np.float32)
        df['hurst_std_50'] = h_series.rolling(50).std().values.astype(np.float32)
        
        df['trending_mkt'] = (hurst > 0.6).astype(np.int8)
        df['reverting_mkt'] = (hurst < 0.4).astype(np.int8)
        df['random_walk'] = ((hurst >= 0.45) & (hurst <= 0.55)).astype(np.int8)
        return df

    # ------------------------------------------------------------------
    # Fully vectorized Efficiency Ratio (no Python loops)
    # ------------------------------------------------------------------
    def add_efficiency_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close']
        for window in self.er_windows:
            direction = (close - close.shift(window)).abs()
            volatility = close.diff().abs().rolling(window).sum()
            er = direction / (volatility + 1e-10)
            df[f'efficiency_ratio_{window}'] = er.fillna(0).astype(np.float32)
        
        df['er_smooth_20'] = df['efficiency_ratio_20'].rolling(5).mean().astype(np.float32)
        df['trend_efficiency'] = (df['efficiency_ratio_20'] > 0.5).astype(np.int8)
        df['choppy_efficiency'] = (df['efficiency_ratio_20'] < 0.3).astype(np.int8)
        return df

    # ------------------------------------------------------------------
    # Volatility Regimes using rolling rank (fast)
    # ------------------------------------------------------------------
    def add_volatility_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'realized_vol_20' not in df.columns:
            log_ret = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
            df['realized_vol_20'] = log_ret.rolling(20).std() * np.sqrt(20)
        
        # Using built-in rank(pct=True) instead of scipy's percentileofscore
        df['vol_percentile_20'] = df['realized_vol_20'].rolling(100).rank(pct=True).fillna(0.5).astype(np.float32)
        
        df['vol_regime_low'] = (df['vol_percentile_20'] < 0.3).astype(np.int8)
        df['vol_regime_medium'] = ((df['vol_percentile_20'] >= 0.3) & (df['vol_percentile_20'] < 0.7)).astype(np.int8)
        df['vol_regime_high'] = (df['vol_percentile_20'] >= 0.7).astype(np.int8)
        df['vol_regime_score'] = (df['vol_percentile_20'] * 2 - 1).astype(np.float32)
        return df

    # ------------------------------------------------------------------
    # Skew & Kurtosis (fast, pandas)
    # ------------------------------------------------------------------
    def add_skew_kurtosis_features(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        returns = df['close'].pct_change()
        df['ret_skew_20'] = returns.rolling(20).skew().astype(np.float32)
        df['ret_kurt_20'] = returns.rolling(20).kurt().astype(np.float32)
        df['ret_skew_50'] = returns.rolling(50).skew().astype(np.float32)
        df['ret_kurt_50'] = returns.rolling(50).kurt().astype(np.float32)
        df['positive_skew'] = (df['ret_skew_20'] > 0.5).astype(np.int8)
        df['negative_skew'] = (df['ret_skew_20'] < -0.5).astype(np.int8)
        df['fat_tails'] = (df['ret_kurt_20'] > 3).astype(np.int8)
        return df

    # ------------------------------------------------------------------
    # Log Returns (vectorized)
    # ------------------------------------------------------------------
    def add_ln_returns(self, df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        close = df['close'].values
        for period in periods:
            if period >= len(close):
                df[f'ln_ret_{period}'] = 0.0
            else:
                ln_ret = np.log(close[period:] / (close[:-period] + 1e-10))
                ln_ret_full = np.full(len(close), np.nan)
                ln_ret_full[period:] = ln_ret
                df[f'ln_ret_{period}'] = ln_ret_full.astype(np.float32)
        
        df['ln_ret_accum_20'] = df['ln_ret_1'].rolling(20).sum().astype(np.float32)
        df['ln_ret_accum_50'] = df['ln_ret_1'].rolling(50).sum().astype(np.float32)
        return df

    # ------------------------------------------------------------------
    # Correlation Features (safe)
    # ------------------------------------------------------------------
    def add_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'volume' in df.columns:
            df['price_vol_corr_20'] = df['close'].rolling(20).corr(df['volume']).fillna(0).astype(np.float32)
        if 'rsi' in df.columns and 'macd' in df.columns:
            df['rsi_macd_corr_20'] = df['rsi'].rolling(20).corr(df['macd']).fillna(0).astype(np.float32)
        if 'atr' in df.columns:
            df['price_atr_corr_20'] = df['close'].rolling(20).corr(df['atr']).fillna(0).astype(np.float32)
        return df

    # ------------------------------------------------------------------
    # Z-Scores (vectorized)
    # ------------------------------------------------------------------
    def add_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close'].values
        for window in self.zscore_windows:
            if window >= len(close):
                df[f'zscore_{window}'] = 0.0
                continue
            rolling_mean = pd.Series(close).rolling(window).mean().values
            rolling_std = pd.Series(close).rolling(window).std().values
            zscore = (close - rolling_mean) / (rolling_std + 1e-10)
            df[f'zscore_{window}'] = zscore.astype(np.float32)
        
        df['zscore_divergence'] = (df['zscore_10'] - df['zscore_50']).astype(np.float32)
        df['zscore_extreme'] = (np.abs(df['zscore_20']) > 2).astype(np.int8)
        return df

    # ------------------------------------------------------------------
    # Build all (without entropy and Hilbert – removed for speed)
    # ------------------------------------------------------------------
    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_hurst_fractal(df)
        df = self.add_efficiency_ratio(df)
        df = self.add_volatility_regimes(df)
        df = self.add_skew_kurtosis_features(df)
        df = self.add_ln_returns(df)
        df = self.add_correlation_features(df)
        df = self.add_zscore_features(df)
        return df