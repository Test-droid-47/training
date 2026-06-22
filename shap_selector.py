import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

class SHAPFeatureSelector:
    
    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.top_k = self.cfg.get('shap_top_k', 150)
        self.feat_path = self.cfg.get('features_path', 'selected_features.json')
        self._selected: List[str] = []
        self._importance_scores: Dict[str, float] = {}

    def fit_select(self, df: pd.DataFrame) -> List[str]:
        numeric_df = df.select_dtypes(include=[np.number])
        exclude = {'timestamp'}
        all_cols = [c for c in numeric_df.columns if c not in exclude]
        
        if not SHAP_AVAILABLE or not LGB_AVAILABLE:
            missing = []
            if not SHAP_AVAILABLE:
                missing.append("SHAP")
            if not LGB_AVAILABLE:
                missing.append("LightGBM")
            print(f"⚠️ {', '.join(missing)} not available — keeping all {len(all_cols)} features.")
            self._selected = all_cols
            self._save()
            return self._selected
        
        if len(df) < 200:
            print(f"⚠️ Only {len(df)} rows, may not be enough. Keeping all features.")
            self._selected = all_cols
            self._save()
            return self._selected
        
        print(f"Running SHAP on {len(all_cols)} features with {len(df)} rows...")
        
        X = numeric_df[all_cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.ffill(inplace=True)
        X.fillna(0.0, inplace=True)
        
        for col in X.columns:
            if X[col].std() < 1e-10:
                X[col] = X[col] + np.random.normal(0, 1e-8, len(X))
        
        if 'close' not in df.columns:
            print("'close' column missing for target creation")
            self._selected = all_cols
            self._save()
            return self._selected
        
        y = (df['close'].shift(-1) > df['close']).astype(int)
        valid = ~y.isna()
        X_v = X[valid].iloc[:-1]
        y_v = y[valid].iloc[:-1]
        
        if len(X_v) < 100:
            print(f"Only {len(X_v)} valid samples, keeping all features")
            self._selected = all_cols
            self._save()
            return self._selected
        
        split = int(len(X_v) * 0.8)
        if split < 10 or len(X_v) - split < 10:
            print("Train/val split too small, keeping all features")
            self._selected = all_cols
            self._save()
            return self._selected
        
        try:
            proxy = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            
            proxy.fit(
                X_v.iloc[:split],
                y_v.iloc[:split],
                eval_set=[(X_v.iloc[split:], y_v.iloc[split:])],
                callbacks=[lgb.early_stopping(10, verbose=False)]
            )
            
            sample_size = min(500, len(X_v.iloc[:split]))
            explainer = shap.TreeExplainer(proxy)
            shap_values = explainer.shap_values(X_v.iloc[:sample_size])
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            mean_abs = np.abs(shap_values).mean(axis=0)
            ranked = np.argsort(mean_abs)[::-1]
            top_k = min(self.top_k, len(all_cols))
            selected = [all_cols[i] for i in ranked[:top_k]]
            
            self._importance_scores = {all_cols[i]: float(mean_abs[i]) for i in range(len(all_cols))}
            
        except Exception as e:
            print(f"SHAP selection failed: {e}")
            selected = all_cols[:min(self.top_k, len(all_cols))]
        
        must_have = ['open', 'high', 'low', 'close', 'volume', 'atr', 'rsi', 'macd', 'regime', 'hurst_exp', 'adx', 'bb_width', 'vol_ratio']
        for m in must_have:
            if m in all_cols and m not in selected:
                selected.append(m)
        
        if 'timestamp' in selected:
            selected.remove('timestamp')
        
        self._selected = selected
        self._save()
        
        print(f"Selected {len(selected)}/{len(all_cols)} features.")
        print(f"Top 10 features: {selected[:10]}")
        
        return self._selected

    def _save(self) -> None:
        data = {
            'selected_features': self._selected,
            'importance_scores': self._importance_scores,
            'top_k': self.top_k
        }
        try:
            with open(self.feat_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self._selected)} features to {self.feat_path}")
        except Exception as e:
            print(f"Failed to save: {e}")

    def load(self) -> List[str]:
        if not os.path.exists(self.feat_path):
            print(f"Feature file not found: {self.feat_path}")
            return []
        
        try:
            with open(self.feat_path, 'r') as f:
                data = json.load(f)
            
            if 'selected_features' in data:
                self._selected = data['selected_features']
            elif isinstance(data, list):
                self._selected = data
            else:
                self._selected = []
            
            self._importance_scores = data.get('importance_scores', {})
            
            print(f"Loaded {len(self._selected)} features from {self.feat_path}")
            return self._selected
        except Exception as e:
            print(f"Failed to load: {e}")
            return []

    def get_top_features(self, n: int = 20) -> List[tuple]:
        sorted_features = sorted(self._importance_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def get_feature_importance_df(self) -> pd.DataFrame:
        if not self._importance_scores:
            return pd.DataFrame()
        
        df_importance = pd.DataFrame([
            {'feature': k, 'importance': v} for k, v in self._importance_scores.items()
        ])
        df_importance = df_importance.sort_values('importance', ascending=False)
        return df_importance

    @property
    def selected(self) -> List[str]:
        if not self._selected:
            self.load()
        return self._selected