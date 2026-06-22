import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')

SIGNAL_BUY = 1
SIGNAL_SELL = 0
SIGNAL_HOLD = 2

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    cyan = "\x1b[36;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    FORMATS = {
        logging.DEBUG:    grey + "[%(asctime)s] [DEBUG] %(message)s" + reset,
        logging.INFO:     cyan + "[%(asctime)s] [INFO] %(message)s" + reset,
        logging.WARNING:  yellow + "[%(asctime)s] [WARNING] %(message)s" + reset,
        logging.ERROR:    red + "[%(asctime)s] [ERROR] %(message)s" + reset,
        logging.CRITICAL: bold_red + "[%(asctime)s] [CRITICAL] %(message)s" + reset,
    }
    def format(self, record):
        fmt = logging.Formatter(self.FORMATS.get(record.levelno), datefmt='%Y-%m-%d %H:%M:%S')
        return fmt.format(record)

logger = logging.getLogger('BacktestWFO')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    fh = logging.FileHandler('backtest_wfo.log')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

class FileDiscovery:
    @staticmethod
    def find_models(base_path: str = '.') -> Dict[str, str]:
        found = {}
        if not base_path: base_path = '.'
        search = [base_path, os.path.join(base_path, 'models')]
        logger.info("Searching for model files...")
        for sp in search:
            try:
                if not os.path.exists(sp) or not os.path.isdir(sp): continue
                for f in os.listdir(sp):
                    fp = os.path.join(sp, f)
                    fl = f.lower()
                    if fl == 'lstm_model.keras':
                        found['lstm'] = fp
                        logger.info(f"Matched LSTM Model: {fp}")
                    elif fl == 'ppo_agent_actor.keras':
                        found['ppo_actor'] = fp
                        logger.info(f"Matched PPO Actor: {fp}")
                    elif fl == 'ppo_agent_critic.keras':
                        found['ppo_critic'] = fp
                        logger.info(f"Matched PPO Critic: {fp}")
                    elif fl == 'ensemble_model.pkl':
                        found['ensemble'] = fp
                        logger.info(f"Matched Ensemble Model: {fp}")
                    elif fl == 'scaler.pkl':
                        found['scaler'] = fp
                        logger.info(f"Matched Scaler: {fp}")
                    elif f.endswith('.json') and 'feature' in fl:
                        found['features'] = fp
                        logger.info(f"Matched Features Config: {fp}")
            except Exception as e:
                logger.error(f"Error scanning directory {sp}: {e}")
        return found

    @staticmethod
    def find_data(data_path: str = None) -> Dict[str, str]:
        found = {}
        search = []
        if data_path:
            if os.path.isfile(data_path) and data_path.endswith('.csv'):
                found['ohlcv'] = data_path
                search.append(os.path.dirname(data_path))
            else:
                search.append(data_path)
        search.extend(['.', './data'])
        for sp in search:
            try:
                if not sp or not os.path.exists(sp): continue
                if os.path.isfile(sp) and sp.endswith('.csv'):
                    if 'ohlcv' in sp.lower() or 'price' in sp.lower():
                        if 'ohlcv' not in found: found['ohlcv'] = sp
                    elif 'fear' in sp.lower() or 'greed' in sp.lower():
                        if 'fear_greed' not in found: found['fear_greed'] = sp
                elif os.path.isdir(sp):
                    for f in os.listdir(sp):
                        fp = os.path.join(sp, f)
                        if f.endswith('.csv'):
                            if 'ohlcv' in f.lower() or 'price' in f.lower():
                                if 'ohlcv' not in found: found['ohlcv'] = fp
                            elif 'fear' in f.lower() or 'greed' in f.lower():
                                if 'fear_greed' not in found: found['fear_greed'] = fp
            except Exception as e:
                logger.error(f"Error scanning directory {sp}: {e}")
        return found

class SignalEngine:
    def __init__(self, models: Dict, config: Dict):
        self.models = models
        self.config = config
        self.confidence_threshold = config.get('ensemble_confidence_threshold', 0.55)
        self.helper_score_threshold = config.get('helper_score_threshold', 0.60)
        self.ppo_enabled = config.get('enable_ppo_helpers', False)

    def _run_lstm(self, X: np.ndarray, window: int, n_bars: int) -> Tuple[np.ndarray, np.ndarray]:
        direction = np.full(n_bars, SIGNAL_HOLD, dtype=int)
        quality = np.full(n_bars, 0.5)
        if 'lstm' not in self.models or X is None or len(X) == 0:
            return direction, quality
        try:
            out = self.models['lstm'].predict(X, verbose=0)
            if isinstance(out, list) and len(out) >= 2:
                probs = out[1]
                pred_len = min(len(probs), n_bars - window)
                if pred_len > 0 and probs.shape[1] >= 3:
                    direction[window:window+pred_len] = np.argmax(probs[:pred_len, :3], axis=1)
                    if len(out) >= 3:
                        quality[window:window+pred_len] = np.clip(out[2][:pred_len].flatten(), 0, 1)
            elif isinstance(out, np.ndarray):
                pred_len = min(len(out), n_bars - window)
                if pred_len > 0:
                    if out.ndim == 2 and out.shape[1] >= 3:
                        direction[window:window+pred_len] = np.argmax(out[:pred_len, :3], axis=1)
                        quality[window:window+pred_len] = np.max(out[:pred_len, :3], axis=1)
                    elif out.ndim == 2 and out.shape[1] == 2:
                        direction[window:window+pred_len] = np.argmax(out[:pred_len], axis=1)
                        quality[window:window+pred_len] = np.max(out[:pred_len], axis=1)
                    else:
                        vals = out.flatten()[:pred_len]
                        direction[window:window+pred_len] = np.where(vals >= 0.5, SIGNAL_BUY, SIGNAL_SELL)
                        quality[window:window+pred_len] = np.where(vals >= 0.5, vals, 1 - vals)
        except Exception as e:
            logger.error(f"LSTM Inference Failed: {e}")

        nan_mask = np.isnan(quality)
        nan_count = np.sum(nan_mask)
        if nan_count > 0:
            logger.warning(f"Detected {nan_count} NaN values in raw LSTM quality matrix! Sanitizing to neutral baseline (0.5).")
            quality = np.nan_to_num(quality, nan=0.5)
        direction = np.nan_to_num(direction, nan=SIGNAL_HOLD).astype(int)
        return direction, quality

    def _run_ppo_actor(self, X: np.ndarray) -> np.ndarray:
        if not self.ppo_enabled or 'ppo_actor' not in self.models or X is None or len(X) == 0:
            return np.full(max(1, len(X) if X is not None else 1), 0.5)
        try:
            out = self.models['ppo_actor'].predict(X, verbose=0)
            res = np.clip(out[:, 1] if (isinstance(out, np.ndarray) and out.ndim == 2 and out.shape[1] >= 2) else out.flatten(), 0, 1)
            return np.nan_to_num(res, nan=0.5)
        except Exception:
            return np.full(len(X), 0.5)

    def _run_ppo_critic(self, X: np.ndarray) -> np.ndarray:
        if not self.ppo_enabled or 'ppo_critic' not in self.models or X is None or len(X) == 0:
            return np.full(max(1, len(X) if X is not None else 1), 0.5)
        try:
            out = self.models['ppo_critic'].predict(X, verbose=0)
            res = 1 / (1 + np.exp(-out.flatten()))
            return np.nan_to_num(res, nan=0.5)
        except Exception:
            return np.full(len(X), 0.5)

    def _run_ensemble(self, df: pd.DataFrame, features: List[str], scaler=None) -> np.ndarray:
        if 'ensemble' not in self.models or df is None or len(df) == 0:
            return np.full(max(1, len(df) if df is not None else 1), 0.5)
        if scaler is not None:
            available = [f for f in features if f in df.columns]
            if len(available) != len(features):
                logger.warning(f"Ensemble missing some scaler features. Using available: {available[:5]}...")
            X_ens = scaler.transform(df[available].fillna(0).values)
        else:
            available = [f for f in features if f in df.columns] or df.select_dtypes(include=[np.number]).columns.tolist()
            X_ens = df[available].fillna(0).values
        mdl = self.models['ensemble']
        try:
            if hasattr(mdl, 'predict_proba'):
                raw = mdl.predict_proba(X_ens)
                res = raw[:, 1] if raw.ndim == 2 and raw.shape[1] >= 2 else raw.flatten()
            else:
                res = np.clip(mdl.predict(X_ens).flatten(), 0, 1)
            return np.nan_to_num(res, nan=0.5)
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return np.full(len(df), 0.5)

    def generate(self, df: pd.DataFrame, X: np.ndarray, window: int, ensemble_features: List[str], scaler=None) -> pd.DataFrame:
        n = len(df)
        lstm_dir, lstm_qual = self._run_lstm(X, window, n)
        ens_probs = self._run_ensemble(df, ensemble_features, scaler)

        lstm_dir = np.nan_to_num(lstm_dir, nan=SIGNAL_HOLD).astype(int)
        lstm_qual = np.nan_to_num(lstm_qual, nan=0.5)
        ens_probs = np.nan_to_num(ens_probs, nan=0.5)

        ens_full = np.full(n, 0.5)
        ens_full[window:] = ens_probs[window:] if len(ens_probs) >= n else ens_probs[:n-window]

        actor_full = np.full(n, 0.5)
        critic_full = np.full(n, 0.5)
        if self.ppo_enabled:
            p_actor = self._run_ppo_actor(X)
            p_critic = self._run_ppo_critic(X)
            p_actor = np.nan_to_num(p_actor, nan=0.5)
            p_critic = np.nan_to_num(p_critic, nan=0.5)
            actor_full[window:] = p_actor[:n-window] if len(p_actor) >= (n-window) else p_actor
            critic_full[window:] = p_critic[:n-window] if len(p_critic) >= (n-window) else p_critic

        final_signal = np.full(n, SIGNAL_HOLD, dtype=int)
        final_quality = np.full(n, 0.0)
        helper_scores = np.zeros(n)

        for i in range(n):
            lstm_sig = int(lstm_dir[i])
            ens_prob = float(ens_full[i])

            if self.ppo_enabled:
                raw_hs = 0.50 * ens_prob + 0.25 * actor_full[i] + 0.25 * critic_full[i]
                hs = raw_hs if lstm_sig != SIGNAL_SELL else (1.0 - raw_hs)
            else:
                hs = 1.0 if lstm_sig == SIGNAL_HOLD else (ens_prob if lstm_sig == SIGNAL_BUY else (1.0 - ens_prob))
            helper_scores[i] = np.clip(hs, 0, 1)

            if lstm_sig == SIGNAL_BUY and ens_prob >= self.confidence_threshold and hs >= self.helper_score_threshold:
                final_signal[i] = SIGNAL_BUY
                final_quality[i] = lstm_qual[i] * ens_prob * hs
            elif lstm_sig == SIGNAL_SELL and ens_prob <= (1.0 - self.confidence_threshold) and hs >= self.helper_score_threshold:
                final_signal[i] = SIGNAL_SELL
                final_quality[i] = lstm_qual[i] * (1.0 - ens_prob) * hs
            else:
                final_signal[i] = SIGNAL_HOLD
                final_quality[i] = lstm_qual[i]

        df = df.copy()
        df['lstm_signal'], df['lstm_quality'] = lstm_dir, lstm_qual
        df['ensemble_prob'], df['helper_score'] = ens_full, helper_scores
        df['final_signal'] = final_signal
        df['final_quality'] = np.nan_to_num(final_quality, nan=0.5)

        if self.ppo_enabled:
            df['ppo_actor_prob'], df['ppo_critic_conf'] = actor_full, critic_full
        return df

class BacktestRunner:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.models, self.data = {}, {}
        self.warmup_bars = self.config.get('warmup_bars', 200)
        self.skip_feature_engineering = False

    def _load_config(self, path: str = None) -> dict:
        defaults = {
            'symbol': 'BTC/USDT', 'timeframe': '1h',
            'fee_rate': 0.001, 'slippage': 0.0005,
            'initial_capital': 10000, 'window': 60,
            'max_position_pct': 1.0, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.10,
            'enable_ppo_helpers': True, 'ensemble_confidence_threshold': 0.55,
            'helper_score_threshold': 0.60,
            'warmup_bars': 200
        }
        if path and os.path.exists(path):
            try: return {**defaults, **json.load(open(path))}
            except Exception: pass
        return defaults

    def _parse_timestamp(self, series: pd.Series) -> pd.Series:
        for kwargs in [{'utc': True, 'infer_datetime_format': True}, {'unit': 'ms', 'utc': True}, {'unit': 's', 'utc': True}]:
            try: return pd.to_datetime(series, **kwargs)
            except Exception: pass
        parsed = pd.to_datetime(series, errors='coerce')
        return parsed.dt.tz_localize('UTC') if parsed.dt.tz is None else parsed.dt.tz_convert('UTC')

    def _discover_and_load_models(self, models_dir: str = None) -> bool:
        found = FileDiscovery.find_models(models_dir or '.')
        if 'lstm' not in found or 'scaler' not in found: return False
        try:
            from tensorflow.keras.models import load_model
            import joblib
            self.models['lstm'] = load_model(found['lstm'])
            self.models['scaler'] = joblib.load(found['scaler'])
            self.models['scaler_features'] = list(self.models['scaler'].feature_names_in_)

            if 'ppo_actor' in found: self.models['ppo_actor'] = load_model(found['ppo_actor'])
            if 'ppo_critic' in found: self.models['ppo_critic'] = load_model(found['ppo_critic'])
            if 'ensemble' in found: self.models['ensemble'] = joblib.load(found['ensemble'])
            self.models['ensemble_features'] = self.models['scaler_features']
            return True
        except Exception as e:
            logger.error(f"Model Load Error: {e}")
            return False

    def _discover_and_load_data(self, data_path: str = None) -> bool:
        found = FileDiscovery.find_data(data_path)
        if 'ohlcv' not in found: return False
        try:
            df = pd.read_csv(found['ohlcv'])
            df.columns = [c.strip().lower() for c in df.columns]
            if 'timestamp' in df.columns: df['timestamp'] = self._parse_timestamp(df['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=['close'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            if 'fear_greed' in found:
                fg = pd.read_csv(found['fear_greed'])
                fg.columns = [c.strip().lower() for c in fg.columns]
                ts_c = next((c for c in fg.columns if 'time' in c or 'date' in c), None)
                v_c = next((c for c in fg.columns if any(k in c for k in ('fear','greed','value'))), None)
                if ts_c and v_c:
                    fg['timestamp'] = self._parse_timestamp(fg[ts_c])
                    fg['_date'] = fg['timestamp'].dt.normalize()
                    df['_date'] = df['timestamp'].dt.normalize()
                    fg['fear_greed'] = pd.to_numeric(fg[v_c], errors='coerce')
                    df = df.merge(fg[['_date','fear_greed']], on='_date', how='left')
                    df.drop(columns=['_date'], inplace=True)
            if 'fear_greed' not in df.columns: df['fear_greed'] = 50
            df['fear_greed'] = df['fear_greed'].ffill().bfill().fillna(50)
            self.data['raw_df'] = df
            return True
        except Exception as e:
            logger.error(f"Data Load Error: {e}")
            return False

    def _prepare_features(self, raw_df: pd.DataFrame = None) -> bool:
        if raw_df is None:
            raw_df = self.data['raw_df']

        if not self.skip_feature_engineering:
            try:
                from feature_engine import FeatureEngine
                from alpha_factors import AlphaFactorEngine
                from smart_money import SmartMoneyEngine
                df = raw_df.copy()
                df = FeatureEngine.build_all(df)
                df = AlphaFactorEngine(self.config).build_all(df)
                df = SmartMoneyEngine(self.config).build_all(df)
            except ImportError as e:
                logger.error(f"Feature Script Import Missing: {e}")
                return False
        else:
            df = raw_df.copy()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.ffill().fillna(0.0)

        if len(df) <= self.warmup_bars + self.config.get('window', 60):
            logger.error(f"Data length ({len(df)}) too short for warmup+window.")
            return False

        scaler_features = self.models['scaler_features']
        missing = [f for f in scaler_features if f not in df.columns]
        if missing:
            raise RuntimeError(f"Scaler Missing Features: {missing[:5]}")

        scaled_data = self.models['scaler'].transform(df[scaler_features].fillna(0)).astype(np.float32)
        self.data['df'] = df
        self.data['scaled_data'] = scaled_data
        self.data['ensemble_features'] = scaler_features
        return True

    def _generate_signals(self) -> bool:
        df = self.data['df']
        scaled_data = self.data['scaled_data']
        window = self.config.get('window', 60)

        try:
            from numpy.lib.stride_tricks import sliding_window_view
            X = np.ascontiguousarray(sliding_window_view(scaled_data, window_shape=window, axis=0))
        except (ImportError, ValueError):
            X = np.array([scaled_data[i-window:i] for i in range(window, len(scaled_data))])

        engine = SignalEngine(self.models, self.config)
        self.data['df'] = engine.generate(df=df, X=X, window=window,
                                         ensemble_features=self.data.get('ensemble_features', []),
                                         scaler=self.models.get('scaler'))
        return True

    def _run_backtest_loop(self, start_idx: int = 0) -> Dict:
        df = self.data['df']
        initial_capital = float(self.config.get('initial_capital', 10000))
        if initial_capital <= 0:
            logger.warning(f"Invalid initial_capital ({initial_capital}) detected. Defaulting to 10000.0.")
            initial_capital = 10000.0

        fee, slippage = self.config.get('fee_rate', 0.001), self.config.get('slippage', 0.0005)
        sl_pct, tp_pct = self.config.get('stop_loss_pct', 0.05), self.config.get('take_profit_pct', 0.10)

        capital, position, entry_price, stop_loss, take_profit = initial_capital, 0.0, 0.0, 0.0, 0.0
        trades, portfolio = [], [capital]

        for i in tqdm(range(start_idx, len(df)), desc="  Running Institutional Backtest", leave=False):
            try:
                open_p = float(df['open'].iloc[i])
                high_p = float(df['high'].iloc[i])
                low_p = float(df['low'].iloc[i])
                close_p = float(df['close'].iloc[i])

                if (open_p <= 0 or high_p <= 0 or low_p <= 0 or close_p <= 0 or
                    np.isnan(open_p) or np.isnan(high_p) or np.isnan(low_p) or np.isnan(close_p)):
                    portfolio.append(portfolio[-1])
                    continue

                if position > 0:
                    if low_p <= stop_loss:
                        sell_p = max(1e-8, stop_loss * (1 - slippage))
                        cash_received = position * sell_p * (1 - fee)
                        cash_pnl = cash_received - trades[-1]['cash_spent']
                        capital += cash_received
                        trades.append({'type': 'sell', 'price': sell_p, 'pnl': (sell_p - entry_price) / (entry_price if entry_price > 0 else 1e-8), 'cash_pnl': cash_pnl, 'bar': i, 'reason': 'stop_loss'})
                        position = 0.0
                        portfolio.append(capital)
                        continue
                    elif high_p >= take_profit:
                        sell_p = max(1e-8, take_profit * (1 - slippage))
                        cash_received = position * sell_p * (1 - fee)
                        cash_pnl = cash_received - trades[-1]['cash_spent']
                        capital += cash_received
                        trades.append({'type': 'sell', 'price': sell_p, 'pnl': (sell_p - entry_price) / (entry_price if entry_price > 0 else 1e-8), 'cash_pnl': cash_pnl, 'bar': i, 'reason': 'take_profit'})
                        position = 0.0
                        portfolio.append(capital)
                        continue

                signal = int(df['final_signal'].iloc[i])
                quality = float(df['final_quality'].iloc[i])
                if np.isnan(quality): quality = 0.0

                if signal == SIGNAL_BUY and position == 0 and quality >= 0.5:
                    buy_p = close_p * (1 + slippage)
                    if buy_p > 0:
                        pos_frac = min(quality, self.config.get('max_position_pct', 1.0))
                        cash_spent = capital * pos_frac
                        position = (cash_spent / buy_p) * (1 - fee)
                        capital -= cash_spent
                        entry_price = buy_p
                        stop_loss = entry_price * (1 - sl_pct)
                        take_profit = entry_price * (1 + tp_pct)
                        trades.append({'type': 'buy', 'price': buy_p, 'bar': i, 'quality': quality, 'cash_spent': cash_spent})
                elif signal == SIGNAL_SELL and position > 0:
                    sell_p = max(1e-8, close_p * (1 - slippage))
                    cash_received = position * sell_p * (1 - fee)
                    cash_pnl = cash_received - trades[-1]['cash_spent']
                    capital += cash_received
                    trades.append({'type': 'sell', 'price': sell_p, 'pnl': (sell_p - entry_price) / (entry_price if entry_price > 0 else 1e-8), 'cash_pnl': cash_pnl, 'bar': i, 'reason': 'signal'})
                    position = 0.0

                portfolio.append(capital + position * close_p)
            except Exception as e:
                logger.error(f"Error inside backtest row execution loop at index {i}: {e}")
                portfolio.append(portfolio[-1])

        if position > 0:
            sell_p = max(1e-8, float(df['close'].iloc[-1]) * (1 - slippage))
            cash_pnl = (position * sell_p * (1 - fee)) - trades[-1]['cash_spent']
            capital += (position * sell_p * (1 - fee))
            trades.append({'type': 'sell', 'price': sell_p, 'pnl': (sell_p - entry_price) / (entry_price if entry_price > 0 else 1e-8), 'cash_pnl': cash_pnl, 'bar': len(df)-1, 'reason': 'forced_close'})
            portfolio[-1] = capital

        tf = self.config.get('timeframe', '1h').lower()
        tf_map = {'1m': 365*24*60, '5m': 365*24*12, '15m': 365*24*4, '30m': 365*24*2, '1h': 365*24, '4h': 365*6, '1d': 365}
        ann_factor = tf_map.get(tf, 365*24)

        rets = np.diff(portfolio) / (np.array(portfolio[:-1]) + 1e-10)
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(ann_factor)) if len(rets) else 0
        neg = rets[rets < 0]
        sortino = float(np.mean(rets) / (np.std(neg) + 1e-10) * np.sqrt(ann_factor)) if len(neg) else sharpe
        max_dd = float(((np.array(portfolio) - np.maximum.accumulate(portfolio)) / (np.maximum.accumulate(portfolio) + 1e-10)).min())

        sell_trades = [t for t in trades if t['type'] == 'sell']
        cash_wins = [t['cash_pnl'] for t in sell_trades if t['cash_pnl'] > 0]
        cash_loss = [abs(t['cash_pnl']) for t in sell_trades if t['cash_pnl'] <= 0]

        denom_cap = initial_capital if initial_capital > 0 else 1e-8
        return {
            'total_return': (portfolio[-1] - initial_capital) / denom_cap * 100,
            'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_dd,
            'win_rate': len(cash_wins) / len(sell_trades) if sell_trades else 0,
            'profit_factor': sum(cash_wins) / sum(cash_loss) if sum(cash_loss) > 0 else (float('inf') if sum(cash_wins) > 0 else 0),
            'avg_pnl': float(np.mean([t['pnl'] for t in sell_trades])) if sell_trades else 0,
            'total_trades': len(sell_trades), 'final_capital': portfolio[-1], 'portfolio': portfolio, 'trades': trades
        }

    def run(self, models_dir: str = None, data_path: str = None) -> Dict[str, Any]:
        t_start = time.time()
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " PRODUCTION BACKTEST ENGINE v3.0 (DEFINITIVE)".center(78) + "║")
        print("╚" + "═" * 78 + "╝")

        if not self._discover_and_load_models(models_dir):
            return {'error': 'Model loading failed'}
        if not self._discover_and_load_data(data_path):
            return {'error': 'Data loading failed'}

        warmup = self.warmup_bars
        raw_df = self.data['raw_df']
        if len(raw_df) <= warmup:
            return {'error': 'Data too short for warmup'}
        self.data['raw_df'] = raw_df.iloc[warmup:].reset_index(drop=True)

        if not self._prepare_features() or not self._generate_signals():
            return {'error': 'Feature or signal generation failed'}

        window = self.config.get('window', 60)
        results = self._run_backtest_loop(start_idx=window)

        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " PRODUCTION METRICS REPORT".center(78) + "║")
        print("╠" + "═" * 78 + "╣")
        for k, v, unit in [("Total Return", results['total_return'], "%"), ("Sharpe Ratio", results['sharpe'], ""), ("Sortino Ratio", results['sortino'], ""), ("Max Drawdown", results['max_drawdown']*100, "%"), ("Win Rate", results['win_rate']*100, "%"), ("Profit Factor", results['profit_factor'], "")]:
            print(f"║  🚀 {k:<25}: {v:>18.4f}{unit:<4} ║")
        print("╚" + "═" * 78 + "╝")
        return results
class WalkForwardValidator:
    def __init__(self, config_path: str = None):
        self.backtest = BacktestRunner(config_path)

    def run_validation(self, models_dir: str = None, data_path: str = None) -> Tuple[bool, Dict]:
        logger.info("Initiating Anti-Leakage Walk-Forward Slicing...")
        if not self.backtest._discover_and_load_models(models_dir):
            return False, {}

        found = FileDiscovery.find_data(data_path)
        if 'ohlcv' not in found:
            return False, {'error': 'No OHLCV data found'}

        try:
            df_raw = pd.read_csv(found['ohlcv'])
            df_raw.columns = [c.strip().lower() for c in df_raw.columns]
            if 'timestamp' in df_raw.columns:
                df_raw['timestamp'] = self.backtest._parse_timestamp(df_raw['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            df_raw.dropna(subset=['close'], inplace=True)
            df_raw.reset_index(drop=True, inplace=True)

            if 'fear_greed' in found:
                fg = pd.read_csv(found['fear_greed'])
                fg.columns = [c.strip().lower() for c in fg.columns]
                ts_c = next((c for c in fg.columns if 'time' in c or 'date' in c), None)
                v_c = next((c for c in fg.columns if any(k in c for k in ('fear','greed','value'))), None)
                if ts_c and v_c:
                    fg['timestamp'] = self.backtest._parse_timestamp(fg[ts_c])
                    fg['_date'] = fg['timestamp'].dt.normalize()
                    df_raw['_date'] = df_raw['timestamp'].dt.normalize()
                    fg['fear_greed'] = pd.to_numeric(fg[v_c], errors='coerce')
                    df_raw = df_raw.merge(fg[['_date','fear_greed']], on='_date', how='left')
                    df_raw.drop(columns=['_date'], inplace=True)
            if 'fear_greed' not in df_raw.columns:
                df_raw['fear_greed'] = 50
        except Exception as e:
            return False, {'error': f"Data load failed: {e}"}

        warmup = self.backtest.warmup_bars
        if len(df_raw) <= warmup:
            return False, {'error': 'Data too short for warmup'}
        test_df = df_raw.iloc[warmup:].reset_index(drop=True)
        total_bars = len(test_df)
        num_folds = 4
        fold_size = total_bars // num_folds
        window = self.backtest.config.get('window', 60)
        lookback = 200 + window + warmup

        folds_metrics = []

        for i in range(num_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < num_folds - 1 else total_bars

            raw_start = start + warmup
            raw_end = end + warmup

            slice_start = max(0, raw_start - lookback)
            slice_end = raw_end
            raw_slice = df_raw.iloc[slice_start:slice_end].copy().reset_index(drop=True)

            self.backtest.data['raw_df'] = raw_slice
            self.backtest.skip_feature_engineering = False

            if not self.backtest._prepare_features():
                logger.warning(f"Fold {i+1} feature preparation failed, skipping.")
                continue

            df_fold = self.backtest.data['df']
            prepend_count = raw_start - slice_start
            if prepend_count > 0:
                df_fold = df_fold.iloc[prepend_count:].reset_index(drop=True)
                scaler_features = self.backtest.models['scaler_features']
                if len(df_fold) >= window:
                    scaled = self.backtest.models['scaler'].transform(df_fold[scaler_features].fillna(0)).astype(np.float32)
                    self.backtest.data['df'] = df_fold
                    self.backtest.data['scaled_data'] = scaled
                else:
                    logger.warning(f"Fold {i+1} too short after trimming, skipping.")
                    continue

            if not self.backtest._generate_signals():
                logger.warning(f"Fold {i+1} signal generation failed, skipping.")
                continue

            if len(self.backtest.data['df']) <= window:
                logger.warning(f"Fold {i+1} too short for backtest, skipping.")
                continue

            res = self.backtest._run_backtest_loop(start_idx=window)
            folds_metrics.append(res['sharpe'])
            logger.info(f"  📊 Fold {i+1} Institutional Sharpe: {res['sharpe']:.4f}")

        if not folds_metrics:
            logger.error("All verification folds failed to calculate valid metrics matrix.")
            return False, {'mean_sharpe': 0.0, 'folds': []}

        mean_sharpe = np.mean(folds_metrics)
        passed = mean_sharpe > self.backtest.config.get('min_sharpe_required', 0.5) and min(folds_metrics) > 0
        logger.info(f"Walk-Forward Verdict: {'PASSED' if passed else 'FAILED'} (Mean Sharpe: {mean_sharpe:.4f})")
        return passed, {'mean_sharpe': mean_sharpe, 'folds': folds_metrics}

def main():
    parser = argparse.ArgumentParser(description='Production Quant Strategy Sandbox')
    parser.add_argument('--mode', required=True, choices=['backtest', 'validate'])
    parser.add_argument('--models', default=None)
    parser.add_argument('--data', default=None)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    if args.mode == 'backtest':
        runner = BacktestRunner(config_path=args.config)
        runner.run(models_dir=args.models, data_path=args.data)
    else:
        validator = WalkForwardValidator(config_path=args.config)
        validator.run_validation(models_dir=args.models, data_path=args.data)

if __name__ == '__main__':
    main()