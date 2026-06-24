import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, BatchNormalization,
    Conv1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

class PredictionModel:
    OUTPUT_NAMES = ['price_pred', 'direction', 'entry_quality', 'exit_bar', 'position_size']
    DIR_LABELS = {0: 'STRONG_SELL', 1: 'HOLD', 2: 'STRONG_BUY'}

    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.model: Optional[Model] = None
        self.scaler = None
        self._feature_cols: List[str] = []
        self._close_idx: int = 0
        self._cont_indices: List[int] = []
        self._cat_indices: List[int] = []
        self._num_cont_features: int = 0
        self._num_cat_features: int = 0
        self._calib_price: np.ndarray = np.array([])
        self._calib_quality: np.ndarray = np.array([])
        self._calib_exit: np.ndarray = np.array([])

    def _transformer_block(self, x, num_heads, key_dim, ff_dim, dropout, name=''):
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f'mha_{name}')(
            query=x, value=x, key=x, attention_mask=causal_mask
        )
        attn = Dropout(dropout)(attn)
        x1 = LayerNormalization(name=f'ln1_{name}')(x + attn)
        ff = Dense(ff_dim, activation='gelu', name=f'ff1_{name}')(x1)
        ff = Dense(x1.shape[-1], name=f'ff2_{name}')(ff)
        ff = Dropout(dropout)(ff)
        return LayerNormalization(name=f'ln2_{name}')(x1 + ff)

    def _call_model(self, inputs_cont, inputs_cat, training=False):
        # Direct execution to bypass graph retracing overhead during mixed shapes inference
        return self.model([inputs_cont, inputs_cat], training=training)

    def build(self, input_shape: Tuple[int, int]) -> Model:
        window = input_shape[0]
        dr = self.cfg.get('dropout_rate', 0.2)
        if not self._cont_indices:
            self._num_cat_features = len([c for c in self._feature_cols if c.startswith('regime_') or 'trigger' in c.lower()])
            self._num_cont_features = input_shape[1] - self._num_cat_features

        in_cont = Input(shape=(window, self._num_cont_features), name='cont_input')
        in_cat = Input(shape=(window, self._num_cat_features), name='cat_input')

        x_cont = Conv1D(128, kernel_size=3, padding='causal', activation='relu', name='conv_local')(in_cont)
        x_cont = Conv1D(64, kernel_size=5, padding='causal', activation='relu', name='conv_med')(x_cont)
        # Replaced BatchNormalization with LayerNormalization for stable HFT/Live Inference
        x_cont = LayerNormalization(name='ln_conv')(x_cont)
        
        x_cont = LSTM(self.cfg.get('lstm_units_1', 128), return_sequences=True, name='lstm_1')(x_cont)
        x_cont = Dropout(dr, name='drop_lstm1')(x_cont)
        x_cont = LSTM(self.cfg.get('lstm_units_2', 64), return_sequences=True, name='lstm_2')(x_cont)

        x_cat = Dense(32, activation='gelu', name='cat_latent_projection')(in_cat)
        fused = Concatenate(axis=-1, name='quant_feature_fusion')([x_cont, x_cat])

        x = self._transformer_block(fused, self.cfg.get('attention_heads', 8), self.cfg.get('attention_key_dim', 64), 256, dr, 't1')
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

        model = Model(inputs=[in_cont, in_cat], outputs=[out_price, out_dir, out_eq, out_exit, out_pos], name='Professional_Quant_Model')

        model.compile(
            optimizer=Adam(learning_rate=self.cfg.get('learning_rate', 0.001), clipnorm=1.0),
            loss={
                'price_pred': 'huber',
                'direction': 'sparse_categorical_crossentropy',
                'entry_quality': 'huber',
                'exit_bar': 'huber',
                'position_size': 'huber'
            },
            loss_weights={
                'price_pred': 0.5,
                'direction': 2.0,
                'entry_quality': 1.5,
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
            entry_price = closes[i]
            atr_pct = atrs[i] / (entry_price + 1e-10)
            exit_bar_idx = 0
            max_ret = -np.inf
            for j in range(1, horizon + 1):
                ret = (closes[i+j] - entry_price) / (entry_price + 1e-10)
                if ret > max_ret:
                    max_ret = ret
                if ret < -0.5 * atr_pct and j < horizon:
                    exit_bar_idx = j
                    break
                if ret == max_ret and ret > 0:
                    exit_bar_idx = j
            if exit_bar_idx == 0 and max_ret > 0:
                for j in range(1, horizon + 1):
                    ret = (closes[i+j] - entry_price) / (entry_price + 1e-10)
                    if ret == max_ret:
                        exit_bar_idx = j
                        break
            if exit_bar_idx == 0:
                exit_bar_idx = horizon
            y_exit[i] = float(exit_bar_idx / max_exit_bars)

        y_pos_size = np.zeros(n-1, dtype=np.float32)
        for i in range(n-1):
            horizon = min(max_exit_bars, n-i-1)
            if horizon < 2:
                y_pos_size[i] = 0.25
                continue
            fwd_rets = np.diff(closes[i+1:i+1+horizon]) / (closes[i+1:i+horizon] + 1e-10)
            if len(fwd_rets) < 2 or np.std(fwd_rets) < 1e-10:
                y_pos_size[i] = 0.25
                continue
            sharpe = np.mean(fwd_rets) / (np.std(fwd_rets) + 1e-10)
            clipped_sharpe = np.clip(sharpe, -1.5, 1.5)
            y_pos_size[i] = float((clipped_sharpe + 1.5) / 3.0)

        return y_price, y_direction, y_entry_quality, y_exit, y_pos_size

    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str] = None):
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        closes = numeric_df['close'].values
        atrs = numeric_df['atr'].values if 'atr' in numeric_df.columns else closes * 0.01
        smc_price_cols = [col for col in numeric_df.columns if any(x in col.lower() for x in ['ob_', 'fvg_', 'liquidity_'])]
        for col in smc_price_cols:
            numeric_df[col] = (numeric_df[col] - closes) / (atrs + 1e-10)

        if 'hurst_exp' in numeric_df.columns:
            numeric_df['hurst_exp'] = numeric_df['hurst_exp'].ewm(span=8, adjust=False).mean()
        if 'efficiency_ratio' in numeric_df.columns:
            numeric_df['efficiency_ratio'] = numeric_df['efficiency_ratio'].ewm(span=10, adjust=False).mean()

        skew_cols = [col for col in numeric_df.columns if 'skew' in col.lower()]
        kurt_cols = [col for col in numeric_df.columns if 'kurt' in col.lower()]
        numeric_df.drop(columns=skew_cols + kurt_cols, inplace=True, errors='ignore')

        zscore_cols = [col for col in numeric_df.columns if 'zscore' in col.lower() or 'z_score' in col.lower()]
        for col in zscore_cols:
            numeric_df[col] = np.clip(numeric_df[col], -3.0, 3.0)

        if 'regime' in numeric_df.columns:
            regime_dummies = pd.get_dummies(numeric_df['regime'], prefix='regime').astype(np.float32)
            numeric_df = pd.concat([numeric_df.drop(columns=['regime']), regime_dummies], axis=1)

        if feature_cols is None:
            feature_cols = [c for c in numeric_df.columns if c not in ['timestamp']]
        else:
            new_regime_cols = [c for c in numeric_df.columns if c.startswith('regime_')]
            feature_cols = [c for c in feature_cols if c in numeric_df.columns] + new_regime_cols

        data = numeric_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        if 'close' not in feature_cols:
            raise ValueError("'close' missing from features.")
        close_idx = feature_cols.index('close')

        cat_cols = [c for c in feature_cols if c.startswith('regime_') or 'trigger' in c.lower()]
        cont_cols = [c for c in feature_cols if c not in cat_cols]

        self._cont_indices = [feature_cols.index(c) for c in cont_cols]
        self._cat_indices = [feature_cols.index(c) for c in cat_cols]
        self._num_cont_features = len(cont_cols)
        self._num_cat_features = len(cat_cols)

        n_rows = len(data)
        split_idx = int(n_rows * self.cfg.get('train_split', 0.8))

        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()

        if self.scaler is None:
            self.scaler = RobustScaler()
            self.scaler.fit(train_data[cont_cols])

        train_scaled = train_data.values.astype(np.float32)
        val_scaled = val_data.values.astype(np.float32)

        train_scaled[:, self._cont_indices] = self.scaler.transform(train_data[cont_cols])
        val_scaled[:, self._cont_indices] = self.scaler.transform(val_data[cont_cols])

        scaled = np.vstack([train_scaled, val_scaled])
        close_scaled = scaled[:, close_idx]

        y_price, y_dir, y_eq, y_exit, y_pos = self._engineer_targets(df, close_scaled)

        window = self.cfg.get('window', 120)
        n = len(scaled)
        if n <= window:
            raise ValueError(f"Data length {n} <= window {window}")

        # FIXED: Correct 2D sliding window strategy to preserve (Batch, Window, Features) alignment
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            X = sliding_window_view(scaled, window_shape=(window, scaled.shape[1])).squeeze(1).astype(np.float32)
            X = X[:-1] # Drop future forward window to align with bounded directional labels
        except:
            X = np.array([scaled[i-window:i] for i in range(window, n)], dtype=np.float32)

        y_price = y_price[window-1:n-1].astype(np.float32)
        y_dir = y_dir[window-1:n-1].astype(np.int32)
        y_eq = y_eq[window-1:n-1].astype(np.float32)
        y_exit = y_exit[window-1:n-1].astype(np.float32)
        y_pos = y_pos[window-1:n-1].astype(np.float32)

        min_len = min(len(X), len(y_price), len(y_dir), len(y_eq), len(y_exit), len(y_pos))
        X, y_price, y_dir, y_eq, y_exit, y_pos = X[:min_len], y_price[:min_len], y_dir[:min_len], y_eq[:min_len], y_exit[:min_len], y_pos[:min_len]

        train_seq = min_len - (min_len - (split_idx - window + 1))
        if train_seq <= 0 or train_seq >= min_len:
            train_seq = int(min_len * self.cfg.get('train_split', 0.8))

        self._feature_cols = feature_cols
        self._close_idx = close_idx

        X_train = X[:train_seq]
        X_val = X[train_seq:]
        y_train = {'price_pred': y_price[:train_seq], 'direction': y_dir[:train_seq], 'entry_quality': y_eq[:train_seq], 'exit_bar': y_exit[:train_seq], 'position_size': y_pos[:train_seq]}
        y_val = {'price_pred': y_price[train_seq:], 'direction': y_dir[train_seq:], 'entry_quality': y_eq[train_seq:], 'exit_bar': y_exit[train_seq:], 'position_size': y_pos[train_seq:]}

        return X_train, X_val, y_train, y_val, feature_cols, close_idx

    def _split_to_multi_input(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'cont_input': X[:, :, self._cont_indices],
            'cat_input': X[:, :, self._cat_indices]
        }

    def train(self, X_train, X_val, y_train, y_val):
        if self.model is None:
            self.build((X_train.shape[1], X_train.shape[2]))

        classes = np.unique(y_train['direction'])
        computed_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train['direction'])
        class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, computed_weights)}
        multi_output_class_weights = {'direction': class_weight_dict}

        X_train_multi = self._split_to_multi_input(X_train)
        X_val_multi = self._split_to_multi_input(X_val)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.cfg.get('early_stop_patience', 15), restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        ]
        self.model.fit(
            X_train_multi, y_train,
            validation_data=(X_val_multi, y_val),
            epochs=self.cfg.get('epochs', 100),
            batch_size=self.cfg.get('batch_size', 32),
            callbacks=callbacks,
            class_weight=multi_output_class_weights,
            shuffle=False,
            verbose=1
        )

        val_outs = self._call_model(X_val_multi['cont_input'], X_val_multi['cat_input'], training=False)
        self._calib_price = np.abs(val_outs[0].numpy().flatten() - y_val['price_pred'])
        self._calib_quality = np.abs(val_outs[2].numpy().flatten() - y_val['entry_quality'])
        self._calib_exit = np.abs(val_outs[3].numpy().flatten() - y_val['exit_bar'])

    def predict_live_bar(self, state: np.ndarray, current_row: pd.Series) -> Dict[str, Any]:
        if self.model is None:
            return {'direction_class': 1, 'entry_quality': 0.5, 'position_size': 0.1, 'pred_price': current_row['close']}
        if state.ndim == 2:
            state = np.expand_dims(state, axis=0)

        state_multi = self._split_to_multi_input(state)
        outputs = self._call_model(state_multi['cont_input'], state_multi['cat_input'], training=False)

        pred_price = float(outputs[0].numpy()[0, 0])
        direction_class = int(np.argmax(outputs[1].numpy()[0]))
        entry_quality = float(outputs[2].numpy()[0, 0])
        exit_bar = float(outputs[3].numpy()[0, 0])
        position_size = float(outputs[4].numpy()[0, 0])

        return {
            'pred_price': pred_price,
            'direction_class': direction_class,
            'entry_quality': entry_quality,
            'exit_bar': exit_bar,
            'position_size': position_size,
            'action': self.DIR_LABELS.get(direction_class, 'HOLD')
        }

    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None or not self._feature_cols:
            df['pred_direction'] = 1
            df['pred_entry_quality'] = 0.5
            df['pred_position_size'] = 0.1
            df['pred_price'] = df['close']
            return df

        numeric_df = df.select_dtypes(include=[np.number]).copy()
        valid_cols = [c for c in self._feature_cols if c in numeric_df.columns]
        data = numeric_df[valid_cols].ffill().fillna(0.0)

        scaled = data.values.astype(np.float32)
        scaled[:, self._cont_indices] = self.scaler.transform(data.iloc[:, self._cont_indices])

        window = self.cfg.get('window', 120)
        if len(scaled) < window:
            df['pred_direction'] = 1
            df['pred_entry_quality'] = 0.5
            df['pred_position_size'] = 0.1
            df['pred_price'] = df['close']
            return df

        # FIXED: Structured sliding window matching exactly with df.iloc[window:] index boundaries
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            X = sliding_window_view(scaled, window_shape=(window, scaled.shape[1])).squeeze(1).astype(np.float32)
            X = X[:-1] 
        except:
            X = np.array([scaled[i-window:i] for i in range(window, len(scaled))], dtype=np.float32)

        X_multi = self._split_to_multi_input(X)
        outputs = self._call_model(X_multi['cont_input'], X_multi['cat_input'], training=False)

        df = df.iloc[window:].copy()
        df['pred_direction'] = np.argmax(outputs[1].numpy(), axis=1)
        df['pred_entry_quality'] = outputs[2].numpy().flatten()
        df['pred_position_size'] = outputs[4].numpy().flatten()
        return df

    def save(self, path=None):
        save_path = path or self.cfg.get('model_save_path', 'models/lstm_model.keras')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        if self.scaler:
            joblib.dump(self.scaler, self.cfg.get('scaler_save_path', 'models/scaler.pkl'))
        calib = {
            'price': self._calib_price.tolist(), 'quality': self._calib_quality.tolist(), 'exit': self._calib_exit.tolist(),
            'feature_cols': self._feature_cols, 'close_idx': self._close_idx,
            'cont_indices': self._cont_indices, 'cat_indices': self._cat_indices,
            'num_cont': self._num_cont_features, 'num_cat': self._num_cat_features
        }
        with open(save_path.replace('.keras', '_calib.json'), 'w') as f:
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
            self._cont_indices = c.get('cont_indices', [])
            self._cat_indices = c.get('cat_indices', [])
            self._num_cont_features = c.get('num_cont', 0)
            self._num_cat_features = c.get('num_cat', 0)
        print(f"Model loaded from {load_path}")