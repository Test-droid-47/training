import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import faulthandler
import signal

# Enable segfault handler to catch C-level crashes (just for debugging)
faulthandler.enable()

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(func, nopython=False):
        return func

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

class FeatureEngine:

    def __init__(self, cfg: Dict = None):
        print("[DEBUG] FeatureEngine.__init__() called")
        self.cfg = cfg or {}
        self.window = self.cfg.get('hurst_window', 100)
        self.ob_lookback = self.cfg.get('ob_lookback', 20)
        self.atr_period = self.cfg.get('atr_period', 14)
        self.fvg_atr_mult = self.cfg.get('fvg_atr_mult', 0.2)
        self.max_active_fvgs = self.cfg.get('max_active_fvgs', 500)
        self.weight_bos = self.cfg.get('smc_weight_bos', 10.0)
        self.weight_choch = self.cfg.get('smc_weight_choch', 15.0)
        self.weight_fvg = self.cfg.get('smc_weight_fvg', 8.0)
        self.weight_premium = self.cfg.get('smc_weight_premium', 5.0)
        self.weight_liq = self.cfg.get('smc_weight_liq', 12.0)
        print("[DEBUG] FeatureEngine.__init__() finished")

    @staticmethod
    @jit(nopython=True)
    def _hurst_rs_vectorized(price_series: np.ndarray, window: int) -> np.ndarray:
        print("[DEBUG] _hurst_rs_vectorized() started")
        n = len(price_series)
        hurst = np.full(n, 0.5, dtype=np.float64)
        if n < window + 10:
            print("[DEBUG] _hurst_rs_vectorized() skipped (insufficient data)")
            return hurst
        log_prices = np.log(price_series)
        for i in range(window, n):
            segment = log_prices[i-window+1:i+1]
            mean_centered = segment - np.mean(segment)
            cumsum = np.cumsum(mean_centered)
            r = np.max(cumsum) - np.min(cumsum)
            # Manual sample std (ddof=1) – Numba compatible
            s = np.sqrt(np.sum((segment - np.mean(segment))**2) / (len(segment) - 1))
            if s > 1e-10:
                rs = r / s
                h = np.log(rs) / np.log(window)
                hurst[i] = max(0.0, min(1.0, h))
        print("[DEBUG] _hurst_rs_vectorized() finished")
        return hurst

    @staticmethod
    @jit(nopython=True)
    def _compute_smc_core(highs, lows, closes, opens, atr, lookback, fvg_atr_mult):
        print("[DEBUG] _compute_smc_core() started")
        n = len(highs)
        bos_bull = np.zeros(n, np.int8)
        bos_bear = np.zeros(n, np.int8)
        choch_bull = np.zeros(n, np.int8)
        choch_bear = np.zeros(n, np.int8)
        ob_bull_level = np.zeros(n, np.float32)
        ob_bear_level = np.zeros(n, np.float32)
        ob_bull_strength = np.zeros(n, np.float32)
        ob_bear_strength = np.zeros(n, np.float32)
        fvg_bull_sz = np.zeros(n, np.float32)
        fvg_bear_sz = np.zeros(n, np.float32)
        fvg_bull_fill = np.zeros(n, np.float32)
        fvg_bear_fill = np.zeros(n, np.float32)
        liq_sweep_bull = np.zeros(n, np.int8)
        liq_sweep_bear = np.zeros(n, np.int8)
        
        max_fvgs = 500
        active_bull_l = np.zeros(max_fvgs, np.float32)
        active_bull_h = np.zeros(max_fvgs, np.float32)
        active_bull_sz = np.zeros(max_fvgs, np.float32)
        bull_count = 0
        
        active_bear_l = np.zeros(max_fvgs, np.float32)
        active_bear_h = np.zeros(max_fvgs, np.float32)
        active_bear_sz = np.zeros(max_fvgs, np.float32)
        bear_count = 0

        last_dir = 0
        last_bos_idx = -1
        active_bull_lvl = 0.0
        active_bear_lvl = 0.0
        active_bull_str = 0.0
        active_bear_str = 0.0

        print("[DEBUG] _compute_smc_core: entering main loop")
        for i in range(lookback, n):
            if i % 5000 == 0:
                print(f"[DEBUG] _compute_smc_core: processing index {i}/{n}")

            if last_bos_idx != -1 and (i - last_bos_idx) > (lookback * 2):
                last_dir = 0
            sh = np.max(highs[i-lookback:i])
            sl = np.min(lows[i-lookback:i])
            if closes[i] > sh:
                bos_bull[i] = 1
                last_bos_idx = i
                if last_dir == -1:
                    choch_bull[i] = 1
                last_dir = 1
                for j in range(i-1, max(i-lookback, 0), -1):
                    if closes[j] < opens[j]:
                        active_bull_lvl = lows[j]
                        active_bull_str = (highs[j] - lows[j]) / (atr[j] + 1e-10)
                        break
            elif closes[i] < sl:
                bos_bear[i] = 1
                last_bos_idx = i
                if last_dir == 1:
                    choch_bear[i] = 1
                last_dir = -1
                for j in range(i-1, max(i-lookback, 0), -1):
                    if closes[j] > opens[j]:
                        active_bear_lvl = highs[j]
                        active_bear_str = (highs[j] - lows[j]) / (atr[j] + 1e-10)
                        break
            if active_bull_lvl > 0.0 and closes[i] < active_bull_lvl:
                active_bull_lvl = 0.0
                active_bull_str = 0.0
            if active_bear_lvl > 0.0 and closes[i] > active_bear_lvl:
                active_bear_lvl = 0.0
                active_bear_str = 0.0
            ob_bull_level[i] = active_bull_lvl
            ob_bear_level[i] = active_bear_lvl
            ob_bull_strength[i] = active_bull_str
            ob_bear_strength[i] = active_bear_str

            w_bull = 0
            for k in range(bull_count):
                if closes[i] > active_bull_l[k]:
                    active_bull_l[w_bull] = active_bull_l[k]
                    active_bull_h[w_bull] = active_bull_h[k]
                    active_bull_sz[w_bull] = active_bull_sz[k]
                    w_bull += 1
            bull_count = w_bull

            w_bear = 0
            for k in range(bear_count):
                if closes[i] < active_bear_h[k]:
                    active_bear_l[w_bear] = active_bear_l[k]
                    active_bear_h[w_bear] = active_bear_h[k]
                    active_bear_sz[w_bear] = active_bear_sz[k]
                    w_bear += 1
            bear_count = w_bear

            min_gap = atr[i] * fvg_atr_mult
            if i >= 2:
                gap_bull = lows[i] - highs[i-2]
                if gap_bull > min_gap and bull_count < max_fvgs:
                    active_bull_l[bull_count] = highs[i-2]
                    active_bull_h[bull_count] = lows[i]
                    active_bull_sz[bull_count] = gap_bull
                    bull_count += 1
                gap_bear = lows[i-2] - highs[i]
                if gap_bear > min_gap and bear_count < max_fvgs:
                    active_bear_l[bear_count] = highs[i]
                    active_bear_h[bear_count] = lows[i-2]
                    active_bear_sz[bear_count] = gap_bear
                    bear_count += 1

            if bull_count > 0:
                tot_sz = 0.0
                tot_fill = 0.0
                for k in range(bull_count):
                    tot_sz += active_bull_sz[k]
                    denom = active_bull_h[k] - active_bull_l[k] + 1e-10
                    c_fill = (active_bull_h[k] - closes[i]) / denom
                    if c_fill < 0.0: c_fill = 0.0
                    if c_fill > 1.0: c_fill = 1.0
                    tot_fill += c_fill
                fvg_bull_sz[i] = tot_sz
                fvg_bull_fill[i] = tot_fill / bull_count

            if bear_count > 0:
                tot_sz = 0.0
                tot_fill = 0.0
                for k in range(bear_count):
                    tot_sz += active_bear_sz[k]
                    denom = active_bear_h[k] - active_bear_l[k] + 1e-10
                    c_fill = (closes[i] - active_bear_l[k]) / denom
                    if c_fill < 0.0: c_fill = 0.0
                    if c_fill > 1.0: c_fill = 1.0
                    tot_fill += c_fill
                fvg_bear_sz[i] = tot_sz
                fvg_bear_fill[i] = tot_fill / bear_count

            recent_high = np.max(highs[i-lookback:i])
            recent_low = np.min(lows[i-lookback:i])
            if lows[i] < recent_low and closes[i] > recent_low:
                liq_sweep_bull[i] = 1
            if highs[i] > recent_high and closes[i] < recent_high:
                liq_sweep_bear[i] = 1

        print("[DEBUG] _compute_smc_core() finished")
        return (bos_bull, bos_bear, choch_bull, choch_bear,
                ob_bull_level, ob_bear_level, ob_bull_strength, ob_bear_strength,
                fvg_bull_sz, fvg_bear_sz, fvg_bull_fill, fvg_bear_fill,
                liq_sweep_bull, liq_sweep_bear)

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[DEBUG] build_all() started")
        df = df.copy()
        original_timestamp = df['timestamp'].values if 'timestamp' in df.columns else None
        
        print("[DEBUG] Extracting OHLC arrays...")
        highs = df['high'].values.astype(np.float32)
        lows = df['low'].values.astype(np.float32)
        closes = df['close'].values.astype(np.float32)
        opens = df['open'].values.astype(np.float32)
        volumes = df['volume'].values.astype(np.float32)
        n = len(df)

        print("[DEBUG] Calculating Hurst...")
        window = min(self.window, n // 2)
        if window >= 10:
            df['hurst_exp'] = self._hurst_rs_vectorized(df['close'].values.astype(float), window).astype(np.float32)
        else:
            df['hurst_exp'] = 0.5
        df['market_memory'] = (df['hurst_exp'] - 0.5).astype(np.float32)
        print("[DEBUG] Hurst done.")

        print("[DEBUG] Calculating Efficiency Ratio...")
        direction = (df['close'] - df['close'].shift(20)).abs()
        volatility = df['close'].diff().abs().rolling(20).sum()
        df['efficiency_ratio_20'] = (direction / (volatility + 1e-10)).fillna(0).astype(np.float32)

        print("[DEBUG] Calculating Vol Regime...")
        log_ret = np.log(df['close'] / (df['close'].shift(1) + 1e-10))
        realized_vol = log_ret.rolling(20).std() * np.sqrt(20)
        vol_pct = realized_vol.rolling(100).rank(pct=True).fillna(0.5)
        df['vol_regime_score'] = (vol_pct * 2 - 1).astype(np.float32)

        print("[DEBUG] Calculating ATR...")
        tr = np.zeros(n, np.float32)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr[0] = highs[0] - lows[0]
        atr = pd.Series(tr).rolling(self.atr_period, min_periods=1).mean().values.astype(np.float32)
        print("[DEBUG] ATR done.")

        print("[DEBUG] Calling SMC core...")
        res = self._compute_smc_core(highs, lows, closes, opens, atr, self.ob_lookback, self.fvg_atr_mult)
        
        bos_bull, bos_bear, choch_bull, choch_bear = res[0], res[1], res[2], res[3]
        ob_bull_level, ob_bear_level = res[4], res[5]
        df['ob_bull_strength'], df['ob_bear_strength'] = res[6], res[7]
        df['smc_fvg_bull_size'], df['smc_fvg_bear_size'] = res[8], res[9]
        df['fvg_bull_fill'], df['fvg_bear_fill'] = res[10], res[11]
        liq_sweep_bull, liq_sweep_bear = res[12], res[13]

        df['smc_liq_sweep_bull'] = liq_sweep_bull.astype(np.int8)
        df['smc_liq_sweep_bear'] = liq_sweep_bear.astype(np.int8)

        df['close_vs_ob_bull'] = np.where(ob_bull_level > 0.0, (df['close'] - ob_bull_level) / (df['close'] + 1e-10), 0.0).astype(np.float32)
        df['close_vs_ob_bear'] = np.where(ob_bear_level > 0.0, (df['close'] - ob_bear_level) / (df['close'] + 1e-10), 0.0).astype(np.float32)

        print("[DEBUG] Calculating Equilibrium...")
        roll_h = df['high'].rolling(50).max().bfill()
        roll_l = df['low'].rolling(50).min().bfill()
        eq = ((roll_h + roll_l) / 2.0).astype(np.float32)
        smc_premium = (df['close'] > eq).astype(np.float32)
        smc_discount = (df['close'] < eq).astype(np.float32)

        print("[DEBUG] Calculating SMC Score...")
        trend_w = df['hurst_exp'].values
        range_w = 1.0 - trend_w

        score = np.zeros(n, np.float32)
        score += bos_bull * self.weight_bos * trend_w
        score -= bos_bear * self.weight_bos * trend_w
        score += choch_bull * self.weight_choch * trend_w
        score -= choch_bear * self.weight_choch * trend_w
        score += (df['smc_fvg_bull_size'].values > 0) * self.weight_fvg * range_w * (1.0 - df['fvg_bull_fill'].values)
        score -= (df['smc_fvg_bear_size'].values > 0) * self.weight_fvg * range_w * (1.0 - df['fvg_bear_fill'].values)
        score -= smc_premium * self.weight_premium
        score += smc_discount * self.weight_premium
        score += liq_sweep_bull * self.weight_liq * range_w
        score -= liq_sweep_bear * self.weight_liq * range_w
        df['smc_score'] = score.astype(np.float32)

        print("[DEBUG] Calculating Pressure & Ratios...")
        df['buying_pressure'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)).astype(np.float32)
        df['selling_pressure'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)).astype(np.float32)
        
        body = (df['close'] - df['open']).abs()
        rng = df['high'] - df['low'] + 1e-10
        df['displacement_ratio'] = (body / rng).astype(np.float32)

        body_ratio = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        delta = df['volume'] * body_ratio
        df['vw_delta'] = (delta.rolling(20).mean() / (df['volume'].rolling(20).mean() + 1e-10)).fillna(0.0).astype(np.float32)

        print("[DEBUG] Calculating HTF Levels...")
        # Safe rolling 4-period high/low (1h → 4h) — NO resample, NO segfault
        df['htf_high'] = df['high'].rolling(4).max().ffill()
        df['htf_low'] = df['low'].rolling(4).min().ffill()
        df['close_vs_htf'] = ((df['close'] - df['htf_low']) / (df['htf_high'] - df['htf_low'] + 1e-10)).fillna(0.5).astype(np.float32)

        print("[DEBUG] Calculating Z-Score...")
        z_10_m = df['close'].rolling(10).mean()
        z_10_s = df['close'].rolling(10).std()
        z_10 = (df['close'] - z_10_m) / (z_10_s + 1e-10)
        z_50_m = df['close'].rolling(50).mean()
        z_50_s = df['close'].rolling(50).std()
        z_50 = (df['close'] - z_50_m) / (z_50_s + 1e-10)
        df['zscore_divergence'] = (z_10 - z_50).fillna(0).astype(np.float32)

        print("[DEBUG] Calculating TA indicators...")
        if TA_AVAILABLE:
            try:
                ema_v = ta.ema(df['volume'], length=20)
                df['vol_ratio'] = (df['volume'] / (ema_v + 1e-10)).fillna(1.0).astype(np.float32)
                
                # 🔥 FIX: Manual VWAP (safe, no segfault)
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                vwap = (df['volume'] * typical_price).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
                df['close_vs_vwap'] = ((df['close'] - vwap) / (vwap + 1e-10)).fillna(0.0).astype(np.float32)
                
                candle_delta = np.where(df['close'] >= df['open'], df['volume'], -df['volume'])
                cvd = pd.Series(candle_delta).rolling(window=100, min_periods=20).sum()
                cvd_ema = ta.ema(cvd, length=20)
                df['cvd_trend'] = (cvd > cvd_ema).astype(np.int8)

                df['rsi_14'] = ta.rsi(df['close'], length=14).fillna(50).astype(np.float32)
                
                macd_df = ta.macd(df['close'])
                atr_14 = ta.atr(df['high'], df['low'], df['close'], length=14).fillna(1e-10)
                if macd_df is not None and not macd_df.empty:
                    df['macd_hist_norm'] = (macd_df.iloc[:, 1] / (atr_14 + 1e-10)).fillna(0).astype(np.float32)
                else:
                    df['macd_hist_norm'] = 0.0
                    
                bb = ta.bbands(df['close'], length=20, std=2.0)
                if bb is not None and not bb.empty:
                    df['bb_width'] = ((bb.iloc[:, 2] - bb.iloc[:, 0]) / (bb.iloc[:, 1] + 1e-10)).fillna(0).astype(np.float32)
                    df['bb_pct'] = ((df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0] + 1e-10)).fillna(0).astype(np.float32)
                else:
                    df['bb_width'] = 0.0
                    df['bb_pct'] = 0.5
                    
                df['natr'] = ta.natr(df['high'], df['low'], df['close'], length=14).fillna(0).astype(np.float32)
                
                ema_20 = ta.ema(df['close'], length=20)
                df['close_vs_ema_20'] = ((df['close'] - ema_20) / (ema_20 + 1e-10)).fillna(0).astype(np.float32)
            except Exception as e:
                print(f"[DEBUG] TA indicators failed: {e}")
                raise
        else:
            # fallback if pandas_ta not available
            df['vol_ratio'] = 1.0
            df['close_vs_vwap'] = 0.0
            df['cvd_trend'] = 0
            df['rsi_14'] = 50.0
            df['macd_hist_norm'] = 0.0
            df['bb_width'] = 0.0
            df['bb_pct'] = 0.5
            df['natr'] = 0.0
            df['close_vs_ema_20'] = 0.0

        print("[DEBUG] Building final output...")
        elite_features = [
            'hurst_exp', 'market_memory', 'efficiency_ratio_20', 'vol_regime_score',
            'smc_score', 'smc_fvg_bull_size', 'smc_fvg_bear_size', 'fvg_bull_fill', 'fvg_bear_fill',
            'smc_liq_sweep_bull', 'smc_liq_sweep_bear', 'close_vs_ob_bull', 'close_vs_ob_bear',
            'ob_bull_strength', 'ob_bear_strength', 'buying_pressure', 'selling_pressure',
            'displacement_ratio', 'vw_delta', 'vol_ratio', 'close_vs_vwap', 'rsi_14',
            'macd_hist_norm', 'bb_width', 'bb_pct', 'natr', 'close_vs_ema_20',
            'zscore_divergence', 'close_vs_htf'
        ]

        output_df = df[elite_features].copy()
        output_df = output_df.ffill().bfill().fillna(0.0)

        if original_timestamp is not None:
            output_df.insert(0, 'timestamp', original_timestamp)

        print("[DEBUG] build_all() finished successfully")
        return output_df
