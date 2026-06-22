import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from BorutaShap import BorutaShap
import lightgbm as lgb

logger = logging.getLogger('BrainManager')
logger.setLevel(logging.INFO)

BASE_CATEGORY_FEATURES = {
    'trend': ['adx', 'ema_20', 'sma_20', 'aroon_up', 'vortex_p', 'ich_conversion'],
    'momentum': ['rsi_14', 'macd_hist', 'stoch_k', 'cci', 'williams_r', 'roc_10'],
    'volatility': ['atr_14', 'bb_width', 'realized_vol_20', 'natr'],
    'volume': ['obv_divergence', 'vol_ratio', 'mfi', 'cmf', 'vwap'],
    'price_structure': ['close_zscore_20', 'bb_pct', 'cdl_body_size', 'dist_to_r1'],
    'time': ['hour_sin', 'dow_cos', 'month_sin'],
    'statistical': ['close_mean_20', 'close_std_20', 'autocorr_5'],
    'lagged_returns': ['ret_1', 'ret_5', 'log_ret_1', 'log_ret_5']
}

class FeatureBrain:
    def __init__(self, cfg: Dict = None, feature_file="models/selected_features.json"):
        self.cfg = cfg or {}
        self.feature_file = feature_file
        self.selected_base_features = []
        self.last_update = None
        self.load_state()
        self.n_trials = self.cfg.get('boruta_trials', 50)

    def load_state(self):
        if os.path.exists(self.feature_file):
            try:
                with open(self.feature_file, 'r') as f:
                    data = json.load(f)
                self.selected_base_features = data.get('base_features', [])
                last_update_str = data.get('last_update')
                if last_update_str:
                    self.last_update = pd.to_datetime(last_update_str)
                logger.info(f"Brain: Loaded {len(self.selected_base_features)} cached base features")
            except Exception as e:
                logger.warning(f"Brain: Failed to load state: {e}")

    def save_state(self):
        data = {
            'base_features': self.selected_base_features,
            'last_update': pd.Timestamp.now(tz='UTC').isoformat()
        }
        try:
            dir_name = os.path.dirname(self.feature_file)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            with open(self.feature_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Brain: Saved {len(self.selected_base_features)} base features to {self.feature_file}")
        except Exception as e:
            logger.error(f"Brain: Failed to save state: {e}")

    def should_run_update(self):
        if self.last_update is None:
            return True
        now = pd.Timestamp.now(tz='UTC')
        days_since = (now - self.last_update).days
        is_sunday = now.dayofweek == 6
        return is_sunday and days_since >= 7

    def _enforce_categories(self, selected_list: List[str]) -> List[str]:
        final_set = set(selected_list)
        added = []
        for category, feats in BASE_CATEGORY_FEATURES.items():
            present = [f for f in feats if f in final_set]
            if not present and feats:
                candidate = feats[0]
                final_set.add(candidate)
                added.append(candidate)
                logger.info(f"  ➕ Enforced category '{category}' with feature '{candidate}'")
        if added:
            logger.info(f"Brain: Added {len(added)} enforced features")
        return list(final_set)

    def select_base_features(self, df_base: pd.DataFrame) -> List[str]:
        logger.info("🚀 Brain: Starting Boruta+SHAP feature selection...")

        if 'close' not in df_base.columns:
            logger.error("Brain: 'close' column missing. Cannot create target.")
            return self.selected_base_features

        if 'timestamp' in df_base.columns:
            df_base = df_base.sort_values('timestamp')

        # 1. Clean data while keeping raw columns intact for target creation
        X_full = df_base.select_dtypes(include=[np.number]).copy()
        X_full.dropna(axis=1, how='all', inplace=True)
        X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_full = X_full.ffill().dropna()

        # 2. Extract target explicitly using X_full['close'] BEFORE dropping anything
        next_close = X_full['close'].shift(-1)
        valid_mask = next_close.notna()

        # Align inputs and output (drops the last row cleanly)
        X_aligned = X_full[valid_mask]
        y = (next_close[valid_mask] > X_aligned['close']).astype(np.int8)

        # 3. Create ultimate feature pool X by removing raw columns from X_aligned
        raw_cols_to_drop = ['open', 'high', 'low', 'close', 'volume', 'candle_delta']
        raw_present = [c for c in raw_cols_to_drop if c in X_aligned.columns]
        X = X_aligned.drop(columns=raw_present)
        logger.info(f"Isolated feature pool. Dropped raw columns: {raw_present}")

        # 4. Fallback Check 1 (Safe & Solid Target Alignment)
        if len(X) < 200:
            logger.warning(f"Only {len(X)} rows, using fallback correlation selection")
            corr_target = X.corrwith(y).abs()
            top = corr_target.nlargest(30).index.tolist()
            self.selected_base_features = self._enforce_categories(top)
            self.save_state()
            return self.selected_base_features

        # 5. Fallback Check 2
        if len(X) < 100:
            logger.warning(f"After alignment only {len(X)} rows, using fallback")
            corr_target = X.corrwith(y).abs()
            top = corr_target.nlargest(30).index.tolist()
            self.selected_base_features = self._enforce_categories(top)
            self.save_state()
            return self.selected_base_features

        # 6. Fit BorutaShap safely
        model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        feat_selector = BorutaShap(model=model, importance_measure='shap', classification=True)

    try:
        feat_selector.fit(X=X, y=y, n_trials=self.n_trials, random_state=42, sample=False, train_or_test='train')
    
        # BorutaShap version compatibility
        if hasattr(feat_selector, 'accepted_features'):
            confirmed = feat_selector.accepted_features
            tentative = feat_selector.tentative_features
        elif hasattr(feat_selector, 'get_accepted_features'):
            confirmed = feat_selector.get_accepted_features()
            tentative = feat_selector.get_tentative_features()
        else:
            # Old versions (pre-2024)
            confirmed = feat_selector.confirmed_
            tentative = feat_selector.tentative_
    
        selected = confirmed if len(confirmed) >= 5 else (confirmed + tentative)
    except Exception as e:
        logger.error(f"BorutaShap execution failed: {e}. Falling back to correlation.")
        selected = []

        if len(selected) == 0:
            logger.warning("Boruta selected 0 features, using top 30 by correlation")
            corr_target = X.corrwith(y).abs()
            selected = corr_target.nlargest(30).index.tolist()

        selected = self._enforce_categories(selected)
        self.selected_base_features = selected
        self.save_state()
        
        logger.info(f"Brain: Final selected {len(selected)} base features")
        logger.info(f"Brain: Top 10: {selected[:10]}")
        return self.selected_base_features

    def auto_check_update(self, df_base: pd.DataFrame) -> List[str]:
        if self.should_run_update():
            logger.info("📅 Brain: Weekly update triggered")
            return self.select_base_features(df_base)
        else:
            logger.info("Brain: Using cached base features (no update)")
            return self.selected_base_features

brain = FeatureBrain()
