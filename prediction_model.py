import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input,
    MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, BatchNormalization,
    Conv1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight  # Added for professional class balancing

class PredictionModel:
    OUTPUT_NAMES = ['price_pred', 'direction', 'entry_quality', 'exit_bar', 'position_size']
    DIR_LABELS = {0: 'STRONG_SELL', 1: 'HOLD', 2: 'STRONG_BUY'}

    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.model: Optional[Model] = None
        self.scaler = None
        self._feature_cols: List[str] = []
        self._close_idx: int = 0
        self._calib_price: np.ndarray = np.array([])
        self._calib_quality: np.ndarray = np.array([])
        self._calib_exit: np.ndarray = np.array([])

    def _transformer_block(self, x, num_heads, key_dim, ff_dim, dropout, name=''):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f'mha_{name}')(x, x)
        attn = Dropout(dropout)(attn)
        x1 = LayerNormalization(name=f'ln1_{name}')(x + attn)
        ff = Dense(ff_dim, activation='gelu', name=f'ff1_{name}')(x1)
        ff = Dense(x1.shape[-1], name=f'ff2_{name}')(ff)
        ff = Dropout(dropout)(ff)
        return LayerNormalization(name=f'ln2_{name}')(x1 + ff)

    def build(self, input_shape: Tuple[int, int]) -> Model:
        dr = self.cfg.get('dropout_rate', 0.2)
        inp = Input(shape=input_shape, name='seq_input')
        x = Conv1D(128, kernel_size=3, padding='causal', activation='relu', name='conv_local')(inp)
        x = Conv1D(64, kernel_size=5, padding='causal', activation='relu', name='conv_med')(x)
        x = BatchNormalization(name='bn_conv')(x)
        x = LSTM(self.cfg.get('lstm_units_1', 128), return_sequences=True, name='lstm_1')(x)
        x = Dropout(dr, name='drop1')(x)
        x = LSTM(self.cfg.get('lstm_units_2', 64), return_sequences=True, name='lstm_2')(x)
        x = Dropout(dr, name='drop2')(x)
        x = self._transformer_block(x, self.cfg.get('attention_heads', 8), self.cfg.get('attention_key_dim', 64), 256, dr, 't1')
        x = self._transformer_block(x, self.cfg.get('attention_heads', 8)//2, self.cfg.get('attention_key_dim', 64), 128, dr, 't2')
        trunk = GlobalAveragePooling1D(name='gap')(x)
        trunk = Dense(256, activation='gelu', name='trunk_1')(trunk)
        trunk = Dropout(dr * 0.5, name='drop_trunk')(trunk)
        trunk = Dense(128, activation='gelu', name='trunk_2')(trunk)

        def head(trunk, units, activation, name):
            h = Dense(64, activation='relu', name=f'h_{name}_1')(trunk)
            h = Dense(32, activation='relu', name=f'h_{name}_2')(h)
            return Dense(units, activation=activation, name=name)(h)

        out_price = head(trunk, 1, 'linear', 'price_pred')
        out_dir = head(trunk, 3, 'softmax', 'direction')
        out_eq = head(trunk, 1, 'sigmoid', 'entry_quality')
        out_exit = head(trunk, 1, 'sigmoid', 'exit_bar')
        out_pos = head(trunk, 1, 'sigmoid', 'position_size')

        model = Model(inputs=inp, outputs=[out_price, out_dir, out_eq, out_exit, out_pos], name='PredictionModel')
        model.compile(
            optimizer=Adam(learning_rate=self.cfg.get('learning_rate', 0.001), clipnorm=1.0),
            loss={
                'price_pred': 'huber',
                'direction': 'sparse_categorical_crossentropy',
                'entry_quality': 'huber',  # FIX: Changed from binary_crossentropy to huber for continuous targets
                'exit_bar': 'huber',
                'position_size': 'huber'
            },
            loss_weights={
                'price_pred': 0.5,       # FIX: Balanced down to prevent drowning out classification gradients
                'direction': 2.0,        # FIX: Increased weight to force model out of majority class guessing
                'entry_quality': 1.5,    # FIX: Optimized for regression-based quality calibration
                'exit_bar': 0.8,
                'position_size': 0.8
            },
            metrics={
                'price_pred': ['mae'],
                'direction': ['accuracy'],
                'entry_quality': ['mae'],
                'exit_bar': ['mae'],
                'position_size': ['mae']
            }
        )
        self.model = model
        return model

    @staticmethod
    def _engineer_targets(df: pd.DataFrame, close_scaled: np.ndarray, max_exit_bars: int = 10):
        n = len(df)
        closes = df['close'].values.astype(np.float64)
        atrs = df['atr'].values.astype(np.float64) if 'atr' in df.columns else closes * 0.01
        adxs = df['adx'].values.astype(np.float64) if 'adx' in df.columns else np.full(n, 25.0)
        regimes = df['regime'].values.astype(int) if 'regime' in df.columns else np.zeros(n, int)
        hursts = df['hurst_exp'].values.astype(np.float64) if 'hurst_exp' in df.columns else np.full(n, 0.5)

        # FIX: Changed from absolute scale target to stationary scaled delta to eliminate validation drift
        y_price = close_scaled[1:] - close_scaled[:-1]

        y_direction = np.ones(n-1, dtype=np.int32)
        for i in range(n-1):
            fwd_ret = (closes[i+1] - closes[i]) / (closes[i] + 1e-10)
            atr_pct = atrs[i] / (closes[i] + 1e-10)
            threshold = max(atr_pct * 0.5, 0.003)
            if fwd_ret > threshold:
                y_direction[i] = 2
            elif fwd_ret < -threshold:
                y_direction[i] = 0

        y_entry_quality = np.zeros(n-1, dtype=np.float32)
        for i in range(n-1):
            horizon = min(5, n-i-1)
            if horizon < 2:
                y_entry_quality[i] = 0.5
                continue
            fwd_rets = np.diff(closes[i+1:i+1+horizon]) / (closes[i+1:i+horizon] + 1e-10)
            if len(fwd_rets) < 2 or np.std(fwd_rets) < 1e-10:
                y_entry_quality[i] = 0.5
                continue
            sharpe = np.mean(fwd_rets) / (np.std(fwd_rets) + 1e-10)
            y_entry_quality[i] = float(1.0 / (1.0 + np.exp(-sharpe * 2)))

        y_exit = np.full(n-1, 0.5, dtype=np.float32)
        for i in range(n-1):
            horizon = min(max_exit_bars, n-i-1)
            if horizon < 1:
                continue
            fwd_returns = [(closes[i+j] - closes[i]) / (closes[i] + 1e-10) for j in range(1, horizon+1)]
            best_bar = int(np.argmax(fwd_returns)) + 1 if fwd_returns else 1
            y_exit[i] = float(best_bar / max_exit_bars)

        y_pos_size = np.full(n-1, 0.25, dtype=np.float32)
        for i in range(n-1):
            base = y_entry_quality[i]
            regime = regimes[i]
            hurst = hursts[i]
            adx = adxs[i]
            regime_mult = {1: 1.0, 2: 0.8, 0: 0.5, 3: 0.2}.get(regime, 0.5)
            hurst_mult = float(np.clip((hurst - 0.3) / 0.4, 0.2, 1.0))
            adx_mult = float(np.clip(adx / 50.0, 0.2, 1.0))
            y_pos_size[i] = float(np.clip(base * regime_mult * hurst_mult * adx_mult, 0.05, 1.0))

        return y_price, y_direction, y_entry_quality, y_exit, y_pos_size

    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str] = None):
        numeric_df = df.select_dtypes(include=[np.number])
        if feature_cols is None:
            feature_cols = [c for c in numeric_df.columns if c not in ['timestamp']]
        else:
            valid_cols = [c for c in feature_cols if c in numeric_df.columns]
            feature_cols = valid_cols

        data = numeric_df[feature_cols].copy()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.ffill().fillna(0.0)

        if 'close' not in feature_cols:
            raise ValueError("'close' missing from features.")
        close_idx = feature_cols.index('close')

        n_rows = len(data)
        split_idx = int(n_rows * self.cfg.get('train_split', 0.8))

        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]

        if self.scaler is None:
            self.scaler = RobustScaler()
            self.scaler.fit(train_data)
        train_scaled = self.scaler.transform(train_data).astype(np.float32)
        val_scaled = self.scaler.transform(val_data).astype(np.float32)

        scaled = np.vstack([train_scaled, val_scaled])
        close_scaled = scaled[:, close_idx]

        y_price, y_dir, y_eq, y_exit, y_pos = self._engineer_targets(df, close_scaled)

        window = self.cfg.get('window', 120)
        n = len(scaled)
        if n <= window:
            raise ValueError(f"Data length {n} <= window {window}")

        X = np.array([scaled[i-window:i] for i in range(window, n)], dtype=np.float32)
        y_price = y_price[window-1:n-1].astype(np.float32)
        y_dir = y_dir[window-1:n-1].astype(np.int32)
        y_eq = y_eq[window-1:n-1].astype(np.float32)
        y_exit = y_exit[window-1:n-1].astype(np.float32)
        y_pos = y_pos[window-1:n-1].astype(np.float32)

        min_len = min(len(X), len(y_price), len(y_dir), len(y_eq), len(y_exit), len(y_pos))
        X = X[:min_len]
        y_price = y_price[:min_len]
        y_dir = y_dir[:min_len]
        y_eq = y_eq[:min_len]
        y_exit = y_exit[:min_len]
        y_pos = y_pos[:min_len]

        train_seq = min_len - (min_len - (split_idx - window + 1))
        if train_seq <= 0 or train_seq >= min_len:
            train_seq = int(min_len * self.cfg.get('train_split', 0.8))

        self._feature_cols = feature_cols
        self._close_idx = close_idx

        X_train = X[:train_seq]
        X_val = X[train_seq:]
        y_train = {
            'price_pred': y_price[:train_seq],
            'direction': y_dir[:train_seq],
            'entry_quality': y_eq[:train_seq],
            'exit_bar': y_exit[:train_seq],
            'position_size': y_pos[:train_seq]
        }
        y_val = {
            'price_pred': y_price[train_seq:],
            'direction': y_dir[train_seq:],
            'entry_quality': y_eq[train_seq:],
            'exit_bar': y_exit[train_seq:],
            'position_size': y_pos[train_seq:]
        }
        return X_train, X_val, y_train, y_val, feature_cols, close_idx

    def train(self, X_train, X_val, y_train, y_val):
        if self.model is None:
            self.build((X_train.shape[1], X_train.shape[2]))

        # FIX: Compute dynamic class weights for the 'direction' head to fight the 80% majority class imbalance trap
        classes = np.unique(y_train['direction'])
        computed_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train['direction'])
        class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, computed_weights)}
        
        # Target specific multi-output dictionary structure for Keras
        multi_output_class_weights = {'direction': class_weight_dict}

        patience = self.cfg.get('early_stop_patience', 15)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        ]
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.cfg.get('epochs', 100),
            batch_size=self.cfg.get('batch_size', 32),
            callbacks=callbacks,
            class_weight=multi_output_class_weights,  # FIX: Attached balanced dictionary here
            shuffle=False,
            verbose=1
        )
        val_outs = self.model.predict(X_val, verbose=0)
        self._calib_price = np.abs(val_outs[0].flatten() - y_val['price_pred'])
        self._calib_quality = np.abs(val_outs[2].flatten() - y_val['entry_quality'])
        self._calib_exit = np.abs(val_outs[3].flatten() - y_val['exit_bar'])

    def predict_live_bar(self, state: np.ndarray, current_row: pd.Series) -> Dict[str, Any]:
        if self.model is None:
            return {'direction_class': 1, 'entry_quality': 0.5, 'position_size': 0.1, 'pred_price': current_row['close']}
        if state.ndim == 2:
            state = np.expand_dims(state, axis=0)
        pred_price, direction_probs, entry_quality, exit_bar, position_size = self.model.predict(state, verbose=0)
        pred_price_val = float(pred_price[0, 0])
        direction_class = int(np.argmax(direction_probs[0]))
        entry_quality_val = float(entry_quality[0, 0])
        position_size_val = float(position_size[0, 0])
        exit_bar_val = float(exit_bar[0, 0])
        return {
            'pred_price': pred_price_val,
            'direction_class': direction_class,
            'entry_quality': entry_quality_val,
            'exit_bar': exit_bar_val,
            'position_size': position_size_val,
            'action': self.DIR_LABELS.get(direction_class, 'HOLD')
        }

    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None or not self._feature_cols:
            df['pred_direction'] = 1
            df['pred_entry_quality'] = 0.5
            df['pred_position_size'] = 0.1
            df['pred_price'] = df['close']
            return df
        numeric_df = df.select_dtypes(include=[np.number])
        valid_cols = [c for c in self._feature_cols if c in numeric_df.columns]
        data = numeric_df[valid_cols].copy()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.ffill().fillna(0.0)
        scaled = self.scaler.transform(data).astype(np.float32)
        window = self.cfg.get('window', 120)
        if len(scaled) < window:
            df['pred_direction'] = 1
            df['pred_entry_quality'] = 0.5
            df['pred_position_size'] = 0.1
            df['pred_price'] = df['close']
            return df
        X = np.array([scaled[i-window:i] for i in range(window, len(scaled))], dtype=np.float32)
        _, direction_probs, entry_quality, _, position_size = self.model.predict(X, verbose=0)
        df = df.iloc[window:].copy()
        df['pred_direction'] = np.argmax(direction_probs, axis=1)
        df['pred_entry_quality'] = entry_quality.flatten()
        df['pred_position_size'] = position_size.flatten()
        return df

    def save(self, path=None):
        save_path = path or self.cfg.get('model_save_path', 'models/lstm_model.keras')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        if self.scaler:
            scaler_path = self.cfg.get('scaler_save_path', 'models/scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        calib = {
            'price': self._calib_price.tolist(),
            'quality': self._calib_quality.tolist(),
            'exit': self._calib_exit.tolist(),
            'feature_cols': self._feature_cols,
            'close_idx': self._close_idx
        }
        calib_path = save_path.replace('.keras', '_calib.json')
        with open(calib_path, 'w') as f:
            json.dump(calib, f)
        print(f"Model saved to {save_path}")

    def load(self, path=None):
        load_path = path or self.cfg.get('model_save_path', 'models/lstm_model.keras')
        self.model = tf.keras.models.load_model(load_path)
        scaler_path = self.cfg.get('scaler_save_path', 'models/scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        calib_path = load_path.replace('.keras', '_calib.json')
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                c = json.load(f)
            self._calib_price = np.array(c['price'])
            self._calib_quality = np.array(c['quality'])
            self._calib_exit = np.array(c['exit'])
            self._feature_cols = c.get('feature_cols', [])
            self._close_idx = c.get('close_idx', 0)
        print(f"Model loaded from {load_path}")
        