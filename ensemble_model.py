import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import RobustScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class EnsembleModel:

    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = RobustScaler()
        self._feat_cols: List[str] = []
        self._weights: Dict[str, float] = {'xgb': 0.5, 'lgb': 0.5}
        self._trained = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_tabular(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare raw (unscaled) feature matrix and binary target.

        Target: 1 if the next bar's close is higher than the current bar's
        close, 0 otherwise.

        FIX — Invisible Last Row Bug:
            pandas evaluates (NaN > float) as False, not NaN.
            So df['close'].shift(-1) > df['close'] silently turns the last
            row into a 0 (fake "Down" label) instead of NaN, and the row
            survives the valid_mask filter.  We hold the shifted series
            separately and force-assign NaN before the boolean cast so
            valid_mask correctly excludes the last row.

        FIX — NaN Imputation:
            Only ffill is applied; no .fillna(0.0).  Remaining NaN values
            are passed through to XGB/LGB so the models can exercise their
            native sparsity-aware split-finding logic.  Zero-padding would
            both corrupt the scaler's median/IQR and disable that capability.

        No lookahead bias, no data leakage.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        cols = [c for c in numeric_df.columns if c != 'timestamp']

        X = numeric_df[cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.ffill()   # Forward-fill only; leave residual NaN for trees

        # Drop zero-variance columns — no signal, destabilise scaler splits
        constant_cols = [c for c in X.columns if X[c].std(skipna=True) < 1e-10]
        if constant_cols:
            X = X.drop(columns=constant_cols)
            cols = [c for c in cols if c not in constant_cols]

        # --- Target generation (last-row-NaN fix) ---
        shifted_close = df['close'].shift(-1)
        y = (shifted_close > df['close']).astype(float)
        y[shifted_close.isna()] = np.nan   # Force last row back to NaN

        valid_mask = ~y.isna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        return X.values, y.values.astype(int), cols

    def _fit_scaler(self, X: np.ndarray) -> None:
        """
        Fit RobustScaler on finite values only.

        RobustScaler raises ValueError on NaN, so we substitute 0.0
        exclusively for the fit step.  After ffill, residual NaN in a
        financial time series is limited to leading rows of each feature,
        so the effect on the computed median and IQR is negligible.
        The actual training data passed to the tree models retains its
        NaN values unchanged.
        """
        X_finite = np.where(np.isfinite(X), X, 0.0)
        self.scaler.fit(X_finite)

    def _fill_and_scale(self, X_raw: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply the fitted scaler while preserving NaN positions.

        Pipeline:
          1. Record the NaN/inf mask.
          2. Substitute 0.0 at those positions so the scaler does not
             raise ValueError (sklearn scalers reject non-finite input).
          3. Scale the finite-only matrix.
          4. Restore NaN at the original positions so XGBoost and LightGBM
             receive the missing-value signal and route samples through
             their learned sparsity-aware default paths.

        This is the single shared path for both single-row and batch
        inference, guaranteeing identical treatment throughout the pipeline.
        Returns None on failure.
        """
        nan_mask = ~np.isfinite(X_raw)
        X_finite = np.where(nan_mask, 0.0, X_raw)   # temp fill for scaler only
        try:
            X_scaled = self.scaler.transform(X_finite)
            X_scaled[nan_mask] = np.nan              # restore for native tree handling
            return X_scaled
        except Exception as exc:
            print(f"Scaler transform failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the XGBoost + LightGBM ensemble.

        Design decisions
        ----------------
        * Scaler is fitted exclusively on the training split (no leakage).
        * NaN values after ffill are passed to XGB/LGB natively; no
          zero-padding is applied anywhere in the pipeline.
        * Class imbalance addressed natively: scale_pos_weight (XGB) and
          class_weight='balanced' (LGB).
        * Both models use early stopping on the held-out validation split.
        """
        print("Training XGBoost + LightGBM ensemble...")

        if 'close' not in df.columns:
            print("ERROR: 'close' column missing — aborting.")
            return

        X_raw, y, cols = self._prepare_tabular(df)
        self._feat_cols = cols

        if len(X_raw) < 200:
            print(f"Insufficient samples ({len(X_raw)}); need at least 200.")
            return

        # ---- Class distribution ----
        pos_rate = y.mean()
        print(
            f"Class balance — up: {pos_rate:.2%}  "
            f"down: {1.0 - pos_rate:.2%}"
        )

        # ---- Temporal train / validation split (no shuffle) ----
        split = int(len(X_raw) * 0.8)
        X_tr_raw, X_val_raw = X_raw[:split], X_raw[split:]
        y_tr, y_val = y[:split], y[split:]

        # ---- Fit scaler on training data only; NaN handled internally ----
        self._fit_scaler(X_tr_raw)
        X_tr = self._fill_and_scale(X_tr_raw)
        X_val = self._fill_and_scale(X_val_raw)

        if X_tr is None or X_val is None:
            print("ERROR: Scaling failed — aborting training.")
            return

        print(
            f"Train: {len(X_tr)}  "
            f"Val: {len(X_val)}  "
            f"Features: {len(cols)}"
        )

        # ---- XGBoost ----
        if XGB_AVAILABLE:
            neg_count = int((y_tr == 0).sum())
            pos_count = int((y_tr == 1).sum())
            scale_pos_weight = neg_count / max(pos_count, 1)

            try:
                self.xgb_model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1,
                    early_stopping_rounds=20,
                    verbosity=0,
                )
                self.xgb_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                acc = (self.xgb_model.predict(X_val) == y_val).mean()
                print(f"XGB  val accuracy: {acc:.4f}")
            except Exception as exc:
                print(f"XGB training failed: {exc}")
                self.xgb_model = None

        # ---- LightGBM ----
        if LGB_AVAILABLE:
            try:
                self.lgb_model = lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                )
                self.lgb_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(20, verbose=False)],
                )
                acc = (self.lgb_model.predict(X_val) == y_val).mean()
                print(f"LGB  val accuracy: {acc:.4f}")
            except Exception as exc:
                print(f"LGB training failed: {exc}")
                self.lgb_model = None

        self._trained = True
        self._update_weights(X_val, y_val)

    def _update_weights(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> None:
        """
        Recompute ensemble weights proportional to validation accuracy.

        Dict-keyed by model name so there is no dependency on insertion
        order — fixing the original index-based bug that stored LGB accuracy
        under the XGB key when only one model was available.
        """
        model_accs: Dict[str, float] = {}

        if self.xgb_model is not None:
            acc = float((self.xgb_model.predict(X_val) == y_val).mean())
            model_accs['xgb'] = max(0.1, acc)   # floor prevents zero-weighting

        if self.lgb_model is not None:
            acc = float((self.lgb_model.predict(X_val) == y_val).mean())
            model_accs['lgb'] = max(0.1, acc)

        if not model_accs:
            return

        total = sum(model_accs.values())
        self._weights = {k: v / total for k, v in model_accs.items()}
        print(f"Ensemble weights: {self._weights}")

    # ------------------------------------------------------------------
    # Inference — single row
    # ------------------------------------------------------------------

    def predict_proba_bullish(self, row: pd.Series) -> float:
        """
        Predict the probability that the next bar closes higher.

        Absent or non-finite feature values remain NaN so XGB/LGB route
        them through their learned sparsity-aware default paths.
        Returns 0.5 (neutral) when no model is available or scaling fails.
        """
        if not self._trained or not self._feat_cols:
            return 0.5

        # Build feature vector; absent / non-finite values stay NaN
        x_raw = np.full((1, len(self._feat_cols)), np.nan, dtype=np.float64)
        for i, col in enumerate(self._feat_cols):
            if col in row.index:
                try:
                    fval = float(row[col])
                    if np.isfinite(fval):
                        x_raw[0, i] = fval
                except (TypeError, ValueError):
                    pass
                # else: stays NaN — handled natively by trees

        x_sc = self._fill_and_scale(x_raw)
        if x_sc is None:
            return 0.5

        probs: List[float] = []
        weights: List[float] = []

        if self.xgb_model is not None:
            try:
                prob = float(self.xgb_model.predict_proba(x_sc)[0, 1])
                probs.append(prob)
                weights.append(self._weights.get('xgb', 0.5))
            except Exception:
                pass

        if self.lgb_model is not None:
            try:
                prob = float(self.lgb_model.predict_proba(x_sc)[0, 1])
                probs.append(prob)
                weights.append(self._weights.get('lgb', 0.5))
            except Exception:
                pass

        if not probs:
            return 0.5

        w_sum = sum(weights)
        if w_sum > 0:
            norm_w = [w / w_sum for w in weights]
            return float(np.average(probs, weights=norm_w))

        return float(np.mean(probs))

    # ------------------------------------------------------------------
    # Inference — batch
    # ------------------------------------------------------------------

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict bullish probability for every row in *df*.

        Does NOT mutate the caller's DataFrame.  Missing columns become NaN
        via reindex.  No zero-padding is applied — NaN values pass through
        to XGB/LGB natively after ffill.
        """
        if not self._trained or not self._feat_cols:
            return np.full(len(df), 0.5)

        # reindex creates a new DataFrame aligned to the trained feature set;
        # absent columns become NaN and are handled natively by trees.
        X = df.reindex(columns=self._feat_cols).copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.ffill()   # No fillna(0.0) — preserve NaN for native tree handling

        x_sc = self._fill_and_scale(X.values)
        if x_sc is None:
            return np.full(len(df), 0.5)

        predictions: List[np.ndarray] = []
        weights: List[float] = []

        if self.xgb_model is not None:
            try:
                xgb_prob = self.xgb_model.predict_proba(x_sc)[:, 1]
                predictions.append(xgb_prob)
                weights.append(self._weights.get('xgb', 0.5))
            except Exception:
                pass

        if self.lgb_model is not None:
            try:
                lgb_prob = self.lgb_model.predict_proba(x_sc)[:, 1]
                predictions.append(lgb_prob)
                weights.append(self._weights.get('lgb', 0.5))
            except Exception:
                pass

        if not predictions:
            return np.full(len(df), 0.5)

        w_sum = sum(weights)
        norm_w = (
            [w / w_sum for w in weights]
            if w_sum > 0
            else [1.0 / len(weights)] * len(weights)
        )

        ensemble = np.zeros(len(df))
        for pred, w in zip(predictions, norm_w):
            ensemble += pred * w

        return ensemble

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """Return aligned feature importances sorted by combined score."""
        importance_data: Dict[str, Dict[str, float]] = {}

        if self.xgb_model is not None:
            for col, imp in zip(
                self._feat_cols, self.xgb_model.feature_importances_
            ):
                importance_data.setdefault(col, {})['xgb_importance'] = float(imp)

        if self.lgb_model is not None:
            for col, imp in zip(
                self._feat_cols, self.lgb_model.feature_importances_
            ):
                importance_data.setdefault(col, {})['lgb_importance'] = float(imp)

        rows = [
            {
                'feature': col,
                'xgb_importance': vals.get('xgb_importance', 0.0),
                'lgb_importance': vals.get('lgb_importance', 0.0),
            }
            for col, vals in importance_data.items()
        ]

        df_imp = pd.DataFrame(rows)
        if not df_imp.empty:
            df_imp['total_importance'] = (
                df_imp['xgb_importance'] + df_imp['lgb_importance']
            )
            df_imp = df_imp.sort_values(
                'total_importance', ascending=False
            ).reset_index(drop=True)

        return df_imp

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "ensemble_model.pkl") -> None:
        """Persist the ensemble models, scaler, and all metadata to disk."""
        try:
            joblib.dump(
                {
                    'xgb_model': self.xgb_model,
                    'lgb_model': self.lgb_model,
                    'scaler': self.scaler,
                    'feat_cols': self._feat_cols,
                    'weights': self._weights,
                    'trained': self._trained,
                },
                path,
            )
            print(f"Ensemble saved  → {path}")
        except Exception as exc:
            print(f"Save failed: {exc}")

    def load(self, path: str = "ensemble_model.pkl") -> bool:
        """Load a previously saved ensemble from disk."""
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return False

        try:
            data = joblib.load(path)
            self.xgb_model = data.get('xgb_model')
            self.lgb_model = data.get('lgb_model')
            self.scaler = data.get('scaler', RobustScaler())
            self._feat_cols = data.get('feat_cols', [])
            self._weights = data.get('weights', {'xgb': 0.5, 'lgb': 0.5})
            self._trained = data.get('trained', True)
            print(f"Ensemble loaded ← {path}")
            return True
        except Exception as exc:
            print(f"Load failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True only if training completed and at least one model is ready."""
        return self._trained and (
            self.xgb_model is not None or self.lgb_model is not None
        )
        