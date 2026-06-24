#!/usr/bin/env python3
"""
Part 2 – Model Training Pipeline (Standalone - Streamlined HFT Edition with Optuna Active)
- Automatically finds OHLCV CSV data (output of Part 1)
- Splits data temporally: 80% train (pure), 20% validation (with history buffer)
- Direct Feature Injection: Bypasses feature selection completely to use all elite metrics
- Strict separation: final_features (only inputs) vs df_cols_to_keep (includes target/timestamp)
- Leakage‑proof scaling for LSTM & Active Optuna Tuner integration
- Automatically updates configuration with Optuna's best discovered parameters
- Trains LSTM+Transformer, Ensemble (XGB+LightGBM), and PPO (training split only)
- Saves all artifacts needed for Part 3 (signal generation)
- Rich logging and progress bars
"""

import os
import sys
import time
import json
import argparse
import logging
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engine import FeatureEngine
from regime_detector import MarketRegimeDetector
from optuna_tuner import OptunaTuner
from prediction_model import PredictionModel
from ensemble_model import EnsembleModel
from ppo_agent import PPOAgent
from trading_env import TradingEnvironment

# --------------------------------------------------------------------------
# Logging setup (rich, with colours and file output)
# --------------------------------------------------------------------------
class ColouredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: cyan + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: red + format_str + reset
    }
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

logger = logging.getLogger('TrainingPipeline')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(ColouredFormatter())
    logger.addHandler(ch)
    fh = logging.FileHandler('training_pipeline.log')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)


class TrainingPipeline:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.start_time = None
        self.stats = {
            'success': False,
            'duration_seconds': 0,
            'ohlcv_bars': 0,
            'feature_engine_columns_count': 0,
            'lstm_tuned_with_optuna': False,
            'lstm_trained': False,
            'ensemble_trained': False,
            'ppo_trained': False,
            'regime_fitted': False
        }

    def _load_config(self, config_path: str = None) -> dict:
        defaults = {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'window': 120,
            'train_split': 0.8,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'lstm_units_1': 128,
            'lstm_units_2': 64,
            'attention_heads': 8,
            'attention_key_dim': 64,
            'dropout_rate': 0.2,
            'optuna_enabled': True,
            'optuna_trials': 15,
            'optuna_epochs': 10,
            'enable_ppo': True,
            'rl_n_episodes': 100,
            'rl_ppo_epochs': 5,
            'rl_gamma': 0.99,
            'rl_clip_epsilon': 0.2,
            'rl_entropy_coeff': 0.01,
            'initial_capital': 10000,
            'fee_rate': 0.001,
            'slippage': 0.0005,
            'max_risk_per_trade': 0.02,
            'max_position_pct': 0.5,
            'drawdown_penalty': 2.0,
            'trading_mode': 'spot',
            'leverage': 10,
            'target_col': 'target'
        }
        paths_to_try = [
            config_path,
            os.path.join(os.path.dirname(__file__), 'config.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json'),
            'config.json'
        ]
        cfg = defaults.copy()
        for path in paths_to_try:
            if path and os.path.exists(path):
                with open(path, 'r') as f:
                    user_cfg = json.load(f)
                cfg.update(user_cfg)
                logger.info(f"✅ Config loaded from {path}")
                break
        else:
            logger.warning("⚠️ No config.json found. Using defaults.")
        return cfg

    def _find_data_csv(self) -> str:
        search_paths = [
            '.',
            './data',
            '../data',
            os.path.dirname(os.path.dirname(__file__)),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        ]
        patterns = ['ohlcv_data.csv', '*ohlcv*.csv', '*_data.csv']
        for path in search_paths:
            if not os.path.exists(path):
                continue
            for pattern in patterns:
                matches = glob.glob(os.path.join(path, pattern))
                if matches:
                    found = matches[0]
                    logger.info(f"✅ Found data CSV: {found}")
                    return found
        raise FileNotFoundError("No OHLCV CSV file found. Please run Part 1 first.")

    def load_data(self) -> pd.DataFrame:
        csv_path = self._find_data_csv()
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        logger.info(f"Loaded {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        self.stats['ohlcv_bars'] = len(df)
        return df

    def train_lstm(self, df_train: pd.DataFrame, df_val: pd.DataFrame, feature_cols: List[str]) -> PredictionModel:
        logger.info("=" * 60)
        logger.info("STEP: Scaling Data & Initializing Neural Pipeline")
        logger.info("=" * 60)
        
        # Base setup to prepare sequences safely
        base_model = PredictionModel(self.config)
        X_train_seq, _, y_train_dict, _, _, _ = base_model.prepare_data(df_train, feature_cols=feature_cols)
        
        # Leakage-proof isolation for validation sequences
        original_fit_transform = base_model.scaler.fit_transform
        base_model.scaler.fit_transform = base_model.scaler.transform
        try:
            X_val_seq, _, y_val_dict, _, _, _ = base_model.prepare_data(df_val, feature_cols=feature_cols)
        finally:
            base_model.scaler.fit_transform = original_fit_transform

        # ----------------------------------------------------------------
        # ACTIVE OPTUNA TUNER EXECUTION
        # ----------------------------------------------------------------
        if self.config.get('optuna_enabled', True):
            logger.info("🎯 Optuna Triggered: Optimizing LSTM + Transformer Hyperparameters...")
            tuner = OptunaTuner(self.config)
            
            # Pass leakage-proof matrices directly to the study
            tuned_config = tuner.tune(X_train_seq, y_train_dict, X_val_seq, y_val_dict)
            
            # Override main runtime config with optimized values
            self.config.update(tuned_config)
            logger.info("✅ Optuna Study Finished. Best parameters dynamically injected into config.")
            self.stats['lstm_tuned_with_optuna'] = True
            
            # Re-initialize core model with the optimized hyperparameters
            model = PredictionModel(self.config)
            X_train_seq, _, y_train_dict, _, _, _ = model.prepare_data(df_train, feature_cols=feature_cols)
            
            model.scaler.fit_transform = model.scaler.transform
            try:
                X_val_seq, _, y_val_dict, _, _, _ = model.prepare_data(df_val, feature_cols=feature_cols)
            finally:
                model.scaler.fit_transform = original_fit_transform
        else:
            logger.info("⚠️ Optuna disabled via config. Training with default architectural values.")
            model = base_model

        logger.info("🧠 Commencing Final Training Phase for LSTM+Transformer Net...")
        model.build((X_train_seq.shape[1], X_train_seq.shape[2]))
        model.train(X_train_seq, X_val_seq, y_train_dict, y_val_dict)
        self.stats['lstm_trained'] = True
        return model

    def train_ensemble(self, df_train: pd.DataFrame) -> EnsembleModel:
        logger.info("=" * 60)
        logger.info("STEP: Training Ensemble (XGBoost + LightGBM) on training split")
        logger.info("=" * 60)
        ensemble = EnsembleModel(self.config)
        ensemble.train(df_train)
        self.stats['ensemble_trained'] = True
        return ensemble

    def train_ppo(self, df_train: pd.DataFrame, pred_model: PredictionModel, feature_cols: List[str]) -> Optional[PPOAgent]:
        logger.info("=" * 60)
        logger.info("STEP: Training PPO Agent on training split")
        logger.info("=" * 60)
        if not self.config.get('enable_ppo', True):
            logger.info("PPO disabled (set enable_ppo: false in config to enable)")
            self.stats['ppo_trained'] = False
            return None

        data = df_train[feature_cols].copy()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = data.ffill().fillna(0.0)

        if pred_model.scaler is None:
            logger.error("PredictionModel scaler is missing; cannot scale for PPO")
            return None
        scaled = pred_model.scaler.transform(data).astype(np.float32)

        if 'close' not in feature_cols:
            logger.error("'close' feature missing from feature list")
            return None
        close_idx = feature_cols.index('close')

        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df_train.columns for c in ohlcv_cols):
            logger.error("Original OHLCV columns missing from training DataFrame. PPO cannot run.")
            return None

        env = TradingEnvironment(df_train, scaled, self.config, close_idx)
        state_shape = (self.config['window'], scaled.shape[1])
        ppo = PPOAgent(self.config, state_shape=state_shape)
        ppo.train(env)
        self.stats['ppo_trained'] = True
        return ppo

    def save_artifacts(self, pred_model: PredictionModel, ensemble: EnsembleModel,
                       ppo: Optional[PPOAgent], final_features: List[str], regime_features: List[str]):
        logger.info("=" * 60)
        logger.info("STEP: Saving Models and Artifacts")
        logger.info("=" * 60)
        os.makedirs('models', exist_ok=True)

        pred_model.save('models/lstm_model.keras')
        ensemble.save('models/ensemble_model.pkl')
        if ppo:
            ppo.save('models/ppo_agent')

        feature_file = 'models/final_features.json'
        with open(feature_file, 'w') as f:
            json.dump({
                'final_features': final_features,
                'regime_features': regime_features,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)
        logger.info(f"Final features saved to {feature_file}")
        logger.info(f"  - final_features ({len(final_features)} features): used by LSTM, Ensemble, PPO")
        logger.info(f"  - regime_features ({len(regime_features)} features): used by MarketRegimeDetector")

        stats_file = 'models/training_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        logger.info(f"Training statistics saved to {stats_file}")

        config_file = 'models/training_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        logger.info(f"Training config saved to {config_file}")

    def run(self) -> dict:
        self.start_time = time.time()
        logger.info("=" * 70)
        logger.info("PART 2 – STREAMLINED DIRECT DATA ROUTING PIPELINE")
        logger.info("=" * 70)

        try:
            df_raw = self.load_data()

            split_pct = self.config.get('train_split', 0.8)
            split_idx = int(len(df_raw) * split_pct)
            window_size = self.config.get('window', 120)

            logger.info(f"Splitting Data: {split_pct*100:.0f}% Train up to index {split_idx}, {(1-split_pct)*100:.0f}% Val after.")

            df_train_raw = df_raw.iloc[:split_idx].copy().reset_index(drop=True)
            df_val_raw = df_raw.iloc[split_idx - window_size:].copy().reset_index(drop=True)

            if 'timestamp' in df_train_raw.columns:
                df_train_raw = df_train_raw.set_index(pd.to_datetime(df_train_raw['timestamp'], utc=True))
            if 'timestamp' in df_val_raw.columns:
                df_val_raw = df_val_raw.set_index(pd.to_datetime(df_val_raw['timestamp'], utc=True))

            logger.info("Building features on isolated Training Split via Unified FeatureEngine...")
            fe=FeatureEngine(cfg=self.config)
            df_train_feats = fe.build_all(df_train_raw.copy())
            df_train_feats = df_train_feats.reset_index(drop=True)
            df_train_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_train_feats = df_train_feats.ffill().bfill()

            logger.info("Building features on isolated Validation Split (with history buffer)...")
            df_val_feats = fe.build_all(df_val_raw.copy())
            df_val_feats = df_val_feats.reset_index(drop=True)
            df_val_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_val_feats = df_val_feats.ffill().bfill()

            logger.info("➤ Fitting MarketRegimeDetector strictly on 80% Train Data...")
            regime_detector = MarketRegimeDetector(self.config)
            regime_detector.fit(df_train_feats)
            df_train_feats = regime_detector.annotate(df_train_feats)
            df_val_feats = regime_detector.annotate(df_val_feats)
            regime_detector.save_map()
            self.stats['regime_fitted'] = True

            regime_features = self.config.get('regime_features',
                ['log_ret_20', 'log_ret_5', 'natr', 'adx', 'hurst_exp', 'vol_ratio', 'rsi'])
            logger.info(f"Regime features (from config): {regime_features}")

            # ----------------------------------------------------------------
            # DIRECT ROUTING: EXTRACT ALL COLUMNS FROM FEATURE ENGINE
            # ----------------------------------------------------------------
            protected_infrastructure = ['timestamp', self.config.get('target_col', 'target')]
            final_features = [c for c in df_train_feats.columns if c not in protected_infrastructure]

            self.stats['feature_engine_columns_count'] = len(final_features)
            logger.info(f"🎯 Pure Direct HFT Approach: Injecting all {len(final_features)} features straight to Optuna & Models.")

            # ----------------------------------------------------------------
            # DATAFRAME SLICE LIST (Features + Infrastructure)
            # ----------------------------------------------------------------
            df_cols_to_keep = final_features.copy()
            target_col = self.config.get('target_col', 'target')
            if target_col in df_train_feats.columns and target_col not in df_cols_to_keep:
                df_cols_to_keep.append(target_col)
            if 'timestamp' in df_train_feats.columns and 'timestamp' not in df_cols_to_keep:
                df_cols_to_keep.append('timestamp')

            df_train_final = df_train_feats[df_cols_to_keep].copy().dropna()
            df_val_final = df_val_feats[df_cols_to_keep].iloc[window_size:].copy().reset_index(drop=True).dropna()

            logger.info(f"Final Synchronized Shapes: Train={df_train_final.shape}, Val={df_val_final.shape}")

            lstm_model = self.train_lstm(df_train_final, df_val_final, final_features)
            ensemble_model = self.train_ensemble(df_train_final)
            ppo_model = self.train_ppo(df_train_final, lstm_model, final_features)

            self.save_artifacts(lstm_model, ensemble_model, ppo_model, final_features, regime_features)

            self.stats['duration_seconds'] = round(time.time() - self.start_time, 2)
            self.stats['success'] = True

            logger.info("=" * 70)
            logger.info("✅ PART 2 – TRAINING COMPLETED SUCCESSFULLY (WITH ACTIVE OPTUNA)")
            logger.info("=" * 70)
            logger.info(f"Duration: {self.stats['duration_seconds']} sec")
            logger.info(f"OHLCV bars: {self.stats['ohlcv_bars']}")
            logger.info(f"Total FeatureEngine Features Embedded: {self.stats['feature_engine_columns_count']}")
            logger.info(f"Optuna Optimization Applied: {self.stats['lstm_tuned_with_optuna']}")
            logger.info(f"LSTM trained: {self.stats['lstm_trained']}")
            logger.info(f"Ensemble trained: {self.stats['ensemble_trained']}")
            logger.info(f"PPO trained: {self.stats['ppo_trained']}")
            logger.info("=" * 70)
            return self.stats

        except Exception as e:
            self.stats['success'] = False
            self.stats['error'] = str(e)
            logger.error(f"❌ Pipeline routing failed: {e}", exc_info=True)
            return self.stats


def main():
    parser = argparse.ArgumentParser(description='Part 2: Model Training Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    args = parser.parse_args()

    pipeline = TrainingPipeline(config_path=args.config)
    result = pipeline.run()
    return 0 if result['success'] else 1


if __name__ == '__main__':
    exit(main())
    
