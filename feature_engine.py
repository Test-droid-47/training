import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    print("⚠️ pandas-ta not installed. Install with: pip install pandas-ta")
    TA_AVAILABLE = False

class FeatureEngine:

    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if not TA_AVAILABLE:
            return df

        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)

        ich_df, _ = ta.ichimoku(df['high'], df['low'], df['close'])
        if ich_df is not None and not ich_df.empty:
            df['ich_conversion'] = ich_df.get('ITS_9', np.nan)
            df['ich_base'] = ich_df.get('IKS_26', np.nan)
            df['ich_span_a'] = ich_df.get('ISA_9', np.nan)
            df['ich_span_b'] = ich_df.get('ISB_26', np.nan)
        else:
            df['ich_conversion'] = df['ich_base'] = df['ich_span_a'] = df['ich_span_b'] = np.nan
        df['ich_lagging'] = df['close'].shift(26)

        adx_df = ta.adx(df['high'], df['low'], df['close'])
        if adx_df is not None and not adx_df.empty:
            df['adx'] = adx_df['ADX_14']
            df['adx_p'] = adx_df['DMP_14']
            df['adx_n'] = adx_df['DMN_14']
        else:
            df['adx'] = df['adx_p'] = df['adx_n'] = 0.0

        vx = ta.vortex(df['high'], df['low'], df['close'])
        if vx is not None and not vx.empty:
            df['vortex_p'] = vx['VTXP_14']
            df['vortex_n'] = vx['VTXM_14']
        else:
            df['vortex_p'] = df['vortex_n'] = 0.0

        ar = ta.aroon(df['high'], df['low'])
        if ar is not None and not ar.empty:
            df['aroon_up'] = ar['AROONU_14']
            df['aroon_down'] = ar['AROOND_14']
        else:
            df['aroon_up'] = df['aroon_down'] = 0.0

        df['close_vs_ema_20'] = (df['close'] - df['ema_20']) / (df['ema_20'] + 1e-10)
        df['close_vs_ema_50'] = (df['close'] - df['ema_50']) / (df['ema_50'] + 1e-10)

        df['golden_cross'] = (df['ema_20'] > df['ema_50']).astype(np.int8)
        df['death_cross'] = (df['ema_20'] < df['ema_50']).astype(np.int8)

        df['ema_20_slope'] = df['ema_20'].diff(3) / (df['ema_20'].shift(3) + 1e-10)
        df['ema_50_slope'] = df['ema_50'].diff(3) / (df['ema_50'].shift(3) + 1e-10)

        return df

    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if not TA_AVAILABLE:
            return df

        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi'] = df['rsi_14']

        macd_df = ta.macd(df['close'])
        if macd_df is not None and not macd_df.empty:
            df['macd'] = macd_df.iloc[:, 0]
            df['macd_hist'] = macd_df.iloc[:, 1]
            df['macd_signal'] = macd_df.iloc[:, 2]
            df['macd_cross'] = np.sign(df['macd_hist']).diff().fillna(0).astype(np.int8)
        else:
            df['macd'] = df['macd_hist'] = df['macd_signal'] = 0.0
            df['macd_cross'] = 0

        stoch_df = ta.stoch(df['high'], df['low'], df['close'])
        if stoch_df is not None and not stoch_df.empty:
            df['stoch_k'] = stoch_df['STOCHk_14_3_3']
            df['stoch_d'] = stoch_df['STOCHd_14_3_3']
        else:
            df['stoch_k'] = df['stoch_d'] = 0.0

        df['cci'] = ta.cci(df['high'], df['low'], df['close'])
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
        df['roc_10'] = ta.roc(df['close'], length=10)
        df['momentum_10'] = ta.mom(df['close'], length=10)
        df['awesome_osc'] = ta.ao(df['high'], df['low'])

        trix_df = ta.trix(df['close'], length=15)
        if trix_df is not None and not trix_df.empty:
            df['trix'] = trix_df['TRIX_15_9']
        else:
            df['trix'] = 0.0

        dpo_series = ta.dpo(df['close'], lookahead=False)
        if dpo_series is not None:
            df['dpo'] = dpo_series
        else:
            df['dpo'] = 0.0

        ppo_df = ta.ppo(df['close'])
        if ppo_df is not None and not ppo_df.empty:
            df['ppo'] = ppo_df['PPO_12_26_9']
        else:
            df['ppo'] = 0.0

        df['rsi_overbought'] = (df['rsi'] > 70).astype(np.int8)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(np.int8)
        df['rsi_midline'] = (df['rsi'] > 50).astype(np.int8)

        return df

    @staticmethod
    def add_volatility_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if not TA_AVAILABLE:
            return df

        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr'] = df['atr_14']
        df['natr'] = ta.natr(df['high'], df['low'], df['close'], length=14)
        df['true_range'] = ta.true_range(df['high'], df['low'], df['close'])
        df['atr_vs_mean'] = df['atr'] / (df['atr'].rolling(50).mean() + 1e-10)
        df['high_vol_regime'] = (df['atr_vs_mean'] > 1.5).astype(np.int8)
        df['low_vol_regime'] = (df['atr_vs_mean'] < 0.7).astype(np.int8)

        bb = ta.bbands(df['close'], length=20, std=2.0)
        if bb is not None and not bb.empty:
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / (bb['BBM_20_2.0'] + 1e-10)
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)).astype(np.int8)
        else:
            df['bb_lower'] = df['bb_middle'] = df['bb_upper'] = df['bb_width'] = df['bb_pct'] = 0.0
            df['bb_squeeze'] = 0

        try:
            kc = ta.kc(df['high'], df['low'], df['close'])
            if kc is not None and not kc.empty:
                df['kc_lower'] = kc.iloc[:, 0]
                df['kc_middle'] = kc.iloc[:, 1]
                df['kc_upper'] = kc.iloc[:, 2]
            else:
                raise ValueError
        except Exception:
            df['kc_lower'] = df['kc_middle'] = df['kc_upper'] = 0.0

        try:
            dc = ta.donchian(df['high'], df['low'])
            if dc is not None and not dc.empty:
                df['dc_lower'] = dc.iloc[:, 0]
                df['dc_middle'] = dc.iloc[:, 1]
                df['dc_upper'] = dc.iloc[:, 2]
                df['dc_width'] = (df['dc_upper'] - df['dc_lower']) / (df['dc_middle'] + 1e-10)
            else:
                raise ValueError
        except Exception:
            df['dc_lower'] = df['dc_middle'] = df['dc_upper'] = df['dc_width'] = 0.0

        df['obv'] = ta.obv(df['close'], df['volume'])
        df['obv_ema'] = ta.ema(df['obv'], length=20)
        df['obv_divergence'] = (df['obv'] - df['obv_ema']) / (df['obv_ema'].abs() + 1e-10)
        df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        df['mfi_overbought'] = (df['mfi'] > 80).astype(np.int8)
        df['mfi_oversold'] = (df['mfi'] < 20).astype(np.int8)

        vwap_series = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        if vwap_series is not None:
            df['vwap'] = vwap_series
        else:
            df['vwap'] = df['close'].copy()
        df['close_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)

        df['ad_line'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
        df['adosc'] = ta.adosc(df['high'], df['low'], df['close'], df['volume'])
        df['vol_ema_20'] = ta.ema(df['volume'], length=20)
        df['vol_ratio'] = df['volume'] / (df['vol_ema_20'] + 1e-10)
        df['vol_spike'] = (df['vol_ratio'] > 2.0).astype(np.int8)
        df['vol_dry'] = (df['vol_ratio'] < 0.5).astype(np.int8)

        df['candle_delta'] = np.where(df['close'] >= df['open'], df['volume'], -df['volume'])
        df['cvd'] = df['candle_delta'].rolling(window=100, min_periods=20).sum()
        df['cvd_ema'] = ta.ema(df['cvd'], length=20)
        df['cvd_trend'] = (df['cvd'] > df['cvd_ema']).astype(np.int8)

        df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)

        return df

    @staticmethod
    def add_pivot_fibonacci(df: pd.DataFrame) -> pd.DataFrame:
        ph = df['high'].shift(1).rolling(20).max()
        pl = df['low'].shift(1).rolling(20).min()
        rng = ph - pl
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fib_names = ['fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_100']
        for name, ratio in zip(fib_names, fib_levels):
            df[name] = ph - rng * ratio
            df[f'{name}_dist'] = (df['close'] - df[name]) / (df['close'] + 1e-10)

        prev_h = df['high'].shift(1)
        prev_l = df['low'].shift(1)
        prev_c = df['close'].shift(1)
        df['pvt_std'] = (prev_h + prev_l + prev_c) / 3
        df['pvt_r1'] = 2 * df['pvt_std'] - prev_l
        df['pvt_s1'] = 2 * df['pvt_std'] - prev_h
        df['pvt_r2'] = df['pvt_std'] + (prev_h - prev_l)
        df['pvt_s2'] = df['pvt_std'] - (prev_h - prev_l)
        df['cam_r3'] = prev_c + (prev_h - prev_l) * 1.1 / 4
        df['cam_s3'] = prev_c - (prev_h - prev_l) * 1.1 / 4
        df['dist_to_r1'] = (df['pvt_r1'] - df['close']) / (df['close'] + 1e-10)
        df['dist_to_s1'] = (df['close'] - df['pvt_s1']) / (df['close'] + 1e-10)
        return df

    @staticmethod
    def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        body = np.abs(c - o)
        rng = h - l + 1e-10
        upper = h - np.maximum(o, c)
        lower = np.minimum(o, c) - l

        df['cdl_doji'] = (body <= rng * 0.05).astype(np.int8)
        df['cdl_dragonfly'] = ((body <= rng * 0.05) & (lower > 2*body)).astype(np.int8)
        df['cdl_gravestone'] = ((body <= rng * 0.05) & (upper > 2*body)).astype(np.int8)
        df['cdl_hammer'] = ((lower >= body*2) & (upper <= body*0.3) & (c > o)).astype(np.int8)
        df['cdl_hanging_man'] = ((lower >= body*2) & (upper <= body*0.3) & (c < o)).astype(np.int8)
        df['cdl_marubozu_bull'] = ((body >= rng*0.95) & (c > o)).astype(np.int8)
        df['cdl_marubozu_bear'] = ((body >= rng*0.95) & (c < o)).astype(np.int8)
        df['cdl_candle_bull'] = (c > o).astype(np.int8)
        df['cdl_candle_bear'] = (c < o).astype(np.int8)
        df['cdl_body_size'] = (body / rng).astype(np.float32)
        df['cdl_upper_wick'] = (upper / rng).astype(np.float32)
        df['cdl_lower_wick'] = (lower / rng).astype(np.float32)

        prev_c = df['close'].shift(1)
        prev_o = df['open'].shift(1)
        df['cdl_bull_engulf'] = ((c > o) & (prev_c < prev_o) & (c > prev_o) & (o < prev_c)).astype(np.int8)
        df['cdl_bear_engulf'] = ((c < o) & (prev_c > prev_o) & (c < prev_o) & (o > prev_c)).astype(np.int8)
        return df

    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        ts = df.index
        df['hour'] = ts.hour.astype(np.int8)
        df['day_of_week'] = ts.dayofweek.astype(np.int8)
        df['month'] = ts.month.astype(np.int8)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24).astype(np.float32)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7).astype(np.float32)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7).astype(np.float32)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).astype(np.float32)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).astype(np.float32)
        return df

    @staticmethod
    def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        df['close_mean_20'] = df['close'].rolling(20, min_periods=10).mean()
        df['close_std_20'] = df['close'].rolling(20, min_periods=10).std()
        df['close_zscore_20'] = ((df['close'] - df['close_mean_20']) / (df['close_std_20'] + 1e-10)).astype(np.float32)

        log_ret = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
        df['realized_vol_20'] = log_ret.rolling(20).std() * np.sqrt(20)

        df['autocorr_5'] = df['close'].rolling(20, min_periods=10).corr(df['close'].shift(5)).fillna(0)
        return df

    @staticmethod
    def add_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
        for lag in [1, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            if 'rsi' in df.columns:
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            if 'macd' in df.columns:
                df[f'macd_lag_{lag}'] = df['macd'].shift(lag)

        for period in [1, 5, 20]:
            df[f'ret_{period}'] = df['close'].pct_change(period).astype(np.float32)
            df[f'log_ret_{period}'] = np.log(df['close'] / (df['close'].shift(period) + 1e-10)).astype(np.float32)

        return df

    @staticmethod
    def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        if 'rsi' in df.columns and 'macd' in df.columns:
            df['rsi_macd_x'] = (df['rsi'] * df['macd']).astype(np.float32)
        if 'volume' in df.columns and 'rsi' in df.columns:
            df['vol_rsi_x'] = (df['volume'] * df['rsi']).astype(np.float32)
        if 'atr' in df.columns and 'ema_20' in df.columns:
            df['atr_ema_x'] = (df['atr'] * df['ema_20']).astype(np.float32)
        if 'bb_width' in df.columns and 'rsi' in df.columns:
            df['bb_rsi_x'] = (df['bb_width'] * df['rsi']).astype(np.float32)
        if 'adx' in df.columns and 'rsi' in df.columns:
            df['adx_rsi_x'] = (df['adx'] * df['rsi']).astype(np.float32)
        if 'obv' in df.columns and 'volume' in df.columns:
            df['obv_vol_x'] = (df['obv'] * df['volume']).astype(np.float32)
        if 'cmf' in df.columns and 'mfi' in df.columns:
            df['cmf_mfi_x'] = (df['cmf'] * df['mfi']).astype(np.float32)
        if 'adx' in df.columns and 'atr' in df.columns:
            df['adx_atr_x'] = (df['adx'] * df['atr']).astype(np.float32)
        if 'volume' in df.columns and 'atr' in df.columns:
            df['vol_atr_x'] = (df['volume'] * df['atr']).astype(np.float32)
        return df

    @classmethod
    def build_all(cls, df: pd.DataFrame, mtf_data: Optional[Dict] = None) -> pd.DataFrame:
        original_timestamp = None
        if 'timestamp' in df.columns:
            original_timestamp = df['timestamp']
            df = df.set_index(pd.to_datetime(df['timestamp'], utc=True)).copy()

        df = cls.add_trend_indicators(df)
        df = cls.add_momentum_indicators(df)
        df = cls.add_volatility_volume_indicators(df)
        df = cls.add_pivot_fibonacci(df)
        df = cls.add_candlestick_patterns(df)
        df = cls.add_time_features(df)
        df = cls.add_statistical_features(df)
        df = cls.add_lagged_returns(df)
        df = cls.add_interaction_features(df)

        if original_timestamp is not None:
            df['timestamp'] = original_timestamp.values

        df = df.ffill()
        df = df.dropna()
        df = df.reset_index(drop=True)

        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype(np.float32)

        return df.copy()