import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class SmartMoneyEngine:
    
    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.ob_lookback = self.cfg.get('ob_lookback', 20)
        self.atr_period = self.cfg.get('atr_period', 14)
        self.fvg_atr_mult = self.cfg.get('fvg_atr_mult', 0.2)  # FVG must be at least 20% of ATR
        
        # Weights for SMC score
        self.weight_bos = self.cfg.get('smc_weight_bos', 10.0)
        self.weight_choch = self.cfg.get('smc_weight_choch', 15.0)
        self.weight_fvg = self.cfg.get('smc_weight_fvg', 8.0)
        self.weight_premium = self.cfg.get('smc_weight_premium', 5.0)
        self.weight_liq = self.cfg.get('smc_weight_liq', 12.0)

    def _calculate_atr_inline(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized ATR calculation to prevent external dependency issues."""
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        n = len(df)
        
        tr = np.zeros(n, np.float32)
        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)
            
        tr[0] = highs[0] - lows[0]
        atr = pd.Series(tr).rolling(self.atr_period, min_periods=1).mean().values
        return atr.astype(np.float32)

    def detect_bos_choch(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < self.ob_lookback + 1:
            df['smc_bos_bull'] = 0
            df['smc_bos_bear'] = 0
            df['smc_choch_bull'] = 0
            df['smc_choch_bear'] = 0
            return df

        highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
        n = len(df)
        bos_bull = np.zeros(n, np.int8)
        bos_bear = np.zeros(n, np.int8)
        choch_bull = np.zeros(n, np.int8)
        choch_bear = np.zeros(n, np.int8)
        
        window = self.ob_lookback
        decay_threshold = window * 2
        last_dir = 0
        last_bos_idx = -1
        
        for i in range(window, n):
            # CHoCH Decay / Sideways Reset Logic
            if last_bos_idx != -1 and (i - last_bos_idx) > decay_threshold:
                last_dir = 0  # Market went sideways for too long, clear structural memory
            
            sh = np.max(highs[i-window:i])
            sl = np.min(lows[i-window:i])
            
            if closes[i] > sh:
                bos_bull[i] = 1
                last_bos_idx = i
                if last_dir == -1:
                    choch_bull[i] = 1
                last_dir = 1
            elif closes[i] < sl:
                bos_bear[i] = 1
                last_bos_idx = i
                if last_dir == 1:
                    choch_bear[i] = 1
                last_dir = -1
        
        df['smc_bos_bull'] = bos_bull
        df['smc_bos_bear'] = bos_bear
        df['smc_choch_bull'] = choch_bull
        df['smc_choch_bear'] = choch_bear
        return df

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < self.ob_lookback + 1:
            df['smc_ob_bull_level'] = 0.0
            df['smc_ob_bear_level'] = 0.0
            df['smc_ob_bull_strength'] = 0.0
            df['smc_ob_bear_strength'] = 0.0
            return df

        opens, highs, lows, closes, volumes = df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values
        n = len(df)
        
        ob_bull_level = np.zeros(n, np.float32)
        ob_bear_level = np.zeros(n, np.float32)
        ob_bull_str = np.zeros(n, np.float32)
        ob_bear_str = np.zeros(n, np.float32)
        
        bos_bull = df['smc_bos_bull'].values if 'smc_bos_bull' in df.columns else np.zeros(n)
        bos_bear = df['smc_bos_bear'].values if 'smc_bos_bear' in df.columns else np.zeros(n)
        
        active_bull_level, active_bull_str = 0.0, 0.0
        active_bear_level, active_bear_str = 0.0, 0.0
        
        for i in range(self.ob_lookback, n):
            # Creation Bullish OB (Trace last down-candle's true structural Low)
            if bos_bull[i]:
                for j in range(i-1, max(i-self.ob_lookback, 0), -1):
                    if closes[j] < opens[j]:
                        active_bull_level = float(lows[j])  # Strictly mitigate at candle Low
                        active_bull_str = float(volumes[j])
                        break
            
            # Creation Bearish OB (Trace last up-candle's true structural High)
            if bos_bear[i]:
                for j in range(i-1, max(i-self.ob_lookback, 0), -1):
                    if closes[j] > opens[j]:
                        active_bear_level = float(highs[j])  # Strictly mitigate at candle High
                        active_bear_str = float(volumes[j])
                        break
            
            # Mitigation Check: True Structural Violations (Wick/Body crossing extreme levels)
            if active_bull_level > 0.0 and closes[i] < active_bull_level:
                active_bull_level, active_bull_str = 0.0, 0.0
            if active_bear_level > 0.0 and closes[i] > active_bear_level:
                active_bear_level, active_bear_str = 0.0, 0.0
                
            ob_bull_level[i] = active_bull_level
            ob_bull_str[i] = active_bull_str
            ob_bear_level[i] = active_bear_level
            ob_bear_str[i] = active_bear_str
        
        df['smc_ob_bull_level'] = ob_bull_level
        df['smc_ob_bear_level'] = ob_bear_level
        df['smc_ob_bull_strength'] = ob_bull_str
        df['smc_ob_bear_strength'] = ob_bear_str
        return df

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 3:
            df['smc_fvg_bull'] = 0
            df['smc_fvg_bear'] = 0
            df['smc_fvg_bull_size'] = 0.0
            df['smc_fvg_bear_size'] = 0.0
            return df

        highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
        atr = self._calculate_atr_inline(df)
        n = len(df)
        
        fvg_bull = np.zeros(n, np.int8)
        fvg_bear = np.zeros(n, np.int8)
        fvg_bull_sz = np.zeros(n, np.float32)
        fvg_bear_sz = np.zeros(n, np.float32)
        
        # State trackers for unfilled gaps: list of dicts {'boundary': level, 'size': size}
        active_bull_fvgs: List[Dict] = []
        active_bear_fvgs: List[Dict] = []
        
        for i in range(2, n):
            current_atr = atr[i]
            min_gap = current_atr * self.fvg_atr_mult
            
            # 1. Update/Mitigate existing gaps via State Memory
            active_bull_fvgs = [g for g in active_bull_fvgs if closes[i] >= g['boundary']]
            active_bear_fvgs = [g for g in active_bear_fvgs if closes[i] <= g['boundary']]
            
            # 2. Detect New Bullish FVG (Low of i > High of i-2)
            gap_bull = lows[i] - highs[i-2]
            if gap_bull > min_gap:
                active_bull_fvgs.append({'boundary': float(highs[i-2]), 'size': float(gap_bull)})
            
            # 3. Detect New Bearish FVG (High of i < Low of i-2)
            gap_bear = lows[i-2] - highs[i]
            if gap_bear > min_gap:
                active_bear_fvgs.append({'boundary': float(highs[i-2]), 'size': float(gap_bear)})
                
            # 4. Write dense cumulative features back to arrays
            if active_bull_fvgs:
                fvg_bull[i] = 1
                fvg_bull_sz[i] = sum(g['size'] for g in active_bull_fvgs)
            if active_bear_fvgs:
                fvg_bear[i] = 1
                fvg_bear_sz[i] = sum(g['size'] for g in active_bear_fvgs)
        
        df['smc_fvg_bull'] = fvg_bull
        df['smc_fvg_bear'] = fvg_bear
        df['smc_fvg_bull_size'] = fvg_bull_sz
        df['smc_fvg_bear_size'] = fvg_bear_sz
        return df

    def detect_premium_discount(self, df: pd.DataFrame, period: int = 50) -> pd.DataFrame:
        if len(df) < period:
            df['smc_premium'] = 0
            df['smc_discount'] = 0
            df['smc_eq_level'] = df['close']
            df['smc_eq_dist'] = 0.0
            return df

        # Using backfill (.bfill()) to handle cold-start boundaries safely
        roll_h = df['high'].rolling(period).max().bfill()
        roll_l = df['low'].rolling(period).min().bfill()
        eq = ((roll_h + roll_l) / 2.0).astype(np.float32)
        
        df['smc_premium'] = (df['close'] > eq).astype(np.int8)
        df['smc_discount'] = (df['close'] < eq).astype(np.int8)
        df['smc_eq_level'] = eq
        df['smc_eq_dist'] = ((df['close'] - eq) / (eq + 1e-10)).astype(np.float32)
        return df

    def detect_liquidity_sweeps(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        if len(df) < lookback + 1:
            df['smc_liq_sweep_bull'] = 0
            df['smc_liq_sweep_bear'] = 0
            return df

        highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
        n = len(df)
        liq_sweep_bull = np.zeros(n, np.int8)
        liq_sweep_bear = np.zeros(n, np.int8)
        
        for i in range(lookback, n):
            recent_high = np.max(highs[i-lookback:i])
            recent_low = np.min(lows[i-lookback:i])
            
            # Bullish Sweep: Low sweeps liquidity pool but closes back inside range
            if lows[i] < recent_low and closes[i] > recent_low:
                liq_sweep_bull[i] = 1
            
            # Bearish Sweep: High sweeps liquidity pool but closes back below range
            if highs[i] > recent_high and closes[i] < recent_high:
                liq_sweep_bear[i] = 1
        
        df['smc_liq_sweep_bull'] = liq_sweep_bull
        df['smc_liq_sweep_bear'] = liq_sweep_bear
        return df

    def detect_breaker_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 10:
            df['smc_breaker_bull'] = 0.0
            df['smc_breaker_bear'] = 0.0
            return df

        if 'smc_choch_bull' not in df.columns or 'smc_choch_bear' not in df.columns:
            df = self.detect_bos_choch(df)
        
        n = len(df)
        choch_bull = df['smc_choch_bull'].values
        choch_bear = df['smc_choch_bear'].values
        bos_bull = df['smc_bos_bull'].values
        bos_bear = df['smc_bos_bear'].values
        highs = df['high'].values
        lows = df['low'].values
        
        bb_bull = np.zeros(n, np.float32)
        bb_bear = np.zeros(n, np.float32)
        
        for i in range(1, n):
            if choch_bull[i] == 1:
                for j in range(max(0, i-10), i):
                    if bos_bear[j] == 1:
                        bb_bull[i] = float(lows[j])
                        break
            
            if choch_bear[i] == 1:
                for j in range(max(0, i-10), i):
                    if bos_bull[j] == 1:
                        bb_bear[i] = float(highs[j])
                        break
        
        df['smc_breaker_bull'] = bb_bull
        df['smc_breaker_bear'] = bb_bear
        return df

    def calculate_smc_score(self, df: pd.DataFrame) -> pd.DataFrame:
        # Initialized as pure float32 array to eliminate risk of overflow
        score = np.zeros(len(df), np.float32)
        
        if 'smc_bos_bull' in df.columns:
            score += df['smc_bos_bull'].values * self.weight_bos
            score -= df['smc_bos_bear'].values * self.weight_bos
        
        if 'smc_choch_bull' in df.columns:
            score += df['smc_choch_bull'].values * self.weight_choch
            score -= df['smc_choch_bear'].values * self.weight_choch
        
        if 'smc_fvg_bull' in df.columns:
            score += df['smc_fvg_bull'].values * self.weight_fvg
            score -= df['smc_fvg_bear'].values * self.weight_fvg
        
        if 'smc_premium' in df.columns:
            score -= df['smc_premium'].values * self.weight_premium
            score += df['smc_discount'].values * self.weight_premium
        
        if 'smc_liq_sweep_bull' in df.columns:
            score += df['smc_liq_sweep_bull'].values * self.weight_liq
            score -= df['smc_liq_sweep_bear'].values * self.weight_liq
        
        df['smc_score'] = score
        df['smc_signal_bull'] = (df['smc_score'] > 20.0).astype(np.int8)
        df['smc_signal_bear'] = (df['smc_score'] < -20.0).astype(np.int8)
        
        return df

    def build_all(self, df: pd.DataFrame) -> pd.DataFrame:
        # Defensive strategy against SettingWithCopyWarning
        df = df.copy()
        
        df = self.detect_bos_choch(df)
        df = self.detect_order_blocks(df)
        df = self.detect_fvg(df)
        df = self.detect_premium_discount(df)
        df = self.detect_liquidity_sweeps(df)
        df = self.detect_breaker_blocks(df)
        df = self.calculate_smc_score(df)
        return df
        