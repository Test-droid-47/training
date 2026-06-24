#!/usr/bin/env python3
"""
Part 2 – Model Training Pipeline (Standalone - Streamlined HFT Edition with Optuna Active)
- Beautiful & Highly Visual Terminal Interface Edition
- Visual Phase Banners and Dynamic Progress Micro-tracking
- Chronological 8-Step Verification Grid
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
# Advanced Professional Neon-Themed Terminal Formatter
# --------------------------------------------------------------------------
class AestheticFormatter(logging.Formatter):
    HEADER    = "\x1b[95m"
    BLUE      = "\x1b[94m"
    CYAN      = "\x1b[96m"
    GREEN     = "\x1b[92m"
    YELLOW    = "\x1b[93m"
    RED       = "\x1b[91m"
    BOLD      = "\x1b[1m"
    UNDERLINE = "\x1b[4m"
    RESET     = "\x1b[0m"
    
    format_str = "%(asctime)s | %(message)s"

    def format(self, record):
        log_fmt = self.format_str
        if record.levelno == logging.INFO:
            msg = record.msg
            if any(x in str(msg) for x in ["┌─", "├─", "└─", "│"]):
                log_fmt = f"{self.BLUE}%(message)s{self.RESET}"
            elif "★" in str(msg) or "🎯" in str(msg) or "✅" in str(msg):
                log_fmt = f"{self.GREEN}{self.BOLD}%(asctime)s | %(message)s{self.RESET}"
            elif "⚠️" in str(msg):
                log_fmt = f"{self.YELLOW}%(asctime)s | %(message)s{self.RESET}"
            else:
                log_fmt = f"{self.CYAN}%(asctime)s | %(message)s{self.RESET}"
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            log_fmt = f"{self.RED}{self.BOLD}%(asctime)s [CRITICAL] | %(message)s{self.RESET}"
        elif record.levelno == logging.WARNING:
            log_fmt = f"{self.YELLOW}%(asctime)s [WARNING]  | %(message)s{self.RESET}"
            
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

logger = logging.getLogger('TrainingPipeline')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(AestheticFormatter())
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

    def _show_banner(self, step_title: str, step_num: int):
        """Displays a clean premium console box for major execution milestones."""
        border_len = 70
        padding = (border_len - len(step_title) - 12) // 2
        pad_str = " " * padding
        logger.info(" ")
        logger.info(f"┌{'─' * border_len}┐")
        logger.info(f"│{pad_str}🚀 [STEP {step_num}/8] : {step_title.upper()}{pad_str}│")
        logger.info(f"└{'─' * border_len}┘")

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
                logger.info(f"   ├─ Config Found: Loaded from {path}")
                break
        else:
            logger.warning("   ├─ Warning: No config.json discovered. Using core software fallbacks.")
        return cfg

    def _find_data_csv(self) -> str:
        search_paths = [
            '.', './data', '../data',
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
                    return matches[0]
        raise FileNotFoundError("No OHLCV CSV database matrix detected. Run Part 1 first.")

    def load_data(self) -> pd.DataFrame:
        csv_path = self._find_data_csv()
        logger.info(f"   ├─ Database Scanner: Located target source -> {csv_path}")
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        logger.info(f"   ├─ Extracted Shape : {df.shape[0]} candles/bars matrix loaded.")
        logger.info(f"   └─ Historical Range: From {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        self.stats['ohlcv_bars'] = len(df)
        return df

    def train_lstm(self, df_train: pd.DataFrame, df_val: pd.DataFrame, feature_cols: List[str]) -> PredictionModel:
        self._show_banner("Neural Pipeline Data Preparation", 5)
        logger.info("   ├─ Vectorizing time-series indices into windowed tensor structures...")
        
        base_model = PredictionModel(self.config)
        X_train_seq, _, y_train_dict, _, _, _ = base_model.prepare_data(df_train, feature_cols=feature_cols)
        
        original_fit_transform = base_model.scaler.fit_transform
        base_model.scaler.fit_transform = base_model.scaler.transform
        try:
            X_val_seq, _, y_val_dict, _, _, _ = base_model.prepare_data(df_val, feature_cols=feature_cols)
        finally:
            base_model.scaler.fit_transform = original_fit_transform

        logger.info(f"   ├─ Tensor Matrix Ready: X_train Tensor Shape = {X_train_seq.shape}")
        logger.info(f"   └─ Tensor Matrix Ready: X_val Tensor Shape   = {X_val_seq.shape}")

        # ----------------------------------------------------------------
        # ACTIVE OPTUNA TUNER EXECUTION
        # ----------------------------------------------------------------
        if self.config.get('optuna_enabled', True):
            self._show_banner("Active Optuna Hyperparameter Optimization", 6)
            logger.info("   ⚡ Starting Bayesian optimization loop. Running parallel trials...")
            tuner = OptunaTuner(self.config)
            
            # Direct non-leaking matrices passing
            tuned_config = tuner.tune(X_train_seq, y_train_dict, X_val_seq, y_val_dict)
            self.config.update(tuned_config)
            
            logger.info("   🎯 Optuna Study Finished. Best architectural weights cloned into config.")
            self.stats['lstm_tuned_with_optuna'] = True
            
            # Re-initialize core model with optimized hyperparameters
            model = PredictionModel(self.config)
            X_train_seq, _, y_train_dict, _, _, _ = model.prepare_data(df_train, feature_cols=feature_cols)
            model.scaler.fit_transform = model.scaler.transform
            try:
                X_val_seq, _, y_val_dict, _, _, _ = model.prepare_data(df_val, feature_cols=feature_cols)
            finally:
                model.scaler.fit_transform = original_fit_transform
        else:
            logger.warning("   ⚠️ Optuna Optimization Bypassed via Configuration Flag. Training defaults.")
            model = base_model

        logger.info(" ")
        logger.info("   🧠 [TRAINING] Re-compiling optimized LSTM+Transformer Model Architecture...")
        model.build((X_train_seq.shape[1], X_train_seq.shape[2]))
        
        # Live visual training bar wrap
        logger.info("   🧠 [TRAINING] Fitting neural nodes. Processing epochs safely...")
        model.train(X_train_seq, X_val_seq, y_train_dict, y_val_dict)
        
        logger.info("   ✅ Neural Network architecture successfully consolidated and weights locked.")
        self.stats['lstm_trained'] = True
        return model

    def train_ensemble(self, df_train: pd.DataFrame) -> EnsembleModel:
        self._show_banner("Gradient Boosted Trees (Ensemble Model)", 7)
        logger.info("   ⚡ Initializing XGBoost + LightGBM Joint Multi-Regressors...")
        ensemble = EnsembleModel(self.config)
        
        with tqdm(total=1, desc="   Training Boosted Trees", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            ensemble.train(df_train)
            pbar.update(1)
            
        logger.info("   🎯 Tree weights calculated. Structural split correlations mapped.")
        self.stats['ensemble_trained'] = True
        return ensemble

    def train_ppo(self, df_train: pd.DataFrame, pred_model: PredictionModel, feature_cols: List[str]) -> Optional[PPOAgent]:
        self._show_banner("Deep Reinforcement Learning (PPO Agent)", 8)
        if not self.config.get('enable_ppo', True):
            logger.warning("   ⚠️ PPO Deep Actor-Critic Agent deactivated via settings configuration.")
            self.stats['ppo_trained'] = False
            return None

        logger.info("   ⚡ Re-aligning mathematical indicators for Markov Decision Process...")
        data = df_train[feature_cols].copy().replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        if pred_model.scaler is None:
            logger.error("PredictionModel scaler missing. Aborting PPO compilation.")
            return None
        scaled = pred_model.scaler.transform(data).astype(np.float32)

        close_idx = feature_cols.index('close')
        env = TradingEnvironment(df_train, scaled, self.config, close_idx)
        state_shape = (self.config['window'], scaled.shape[1])
        
        logger.info(f"   ⚡ Launching PPO Policy Gradient optimization across {self.config.get('rl_n_episodes', 100)} simulation runs...")
        ppo = PPOAgent(self.config, state_shape=state_shape)
        ppo.train(env)
        
        logger.info("   🎯 Reinforcement policy weights updated. Neural actions space normalized.")
        self.stats['ppo_trained'] = True
        return ppo

    def save_artifacts(self, pred_model: PredictionModel, ensemble: EnsembleModel,
                       ppo: Optional[PPOAgent], final_features: List[str], regime_features: List[str]):
        logger.info(" ")
        logger.info("┌──────────────────────────────────────────────────────────────────────┐")
        logger.info("│ 💾 ARCHIVING AND EXPORTING CORE COMPILED PIPELINE ARTIFACTS          │")
        logger.info("└──────────────────────────────────────────────────────────────────────┘")
        os.makedirs('models', exist_ok=True)

        pred_model.save('models/lstm_model.keras')
        ensemble.save('models/ensemble_model.pkl')
        if ppo:
            ppo.save('models/ppo_agent')

        with open('models/final_features.json', 'w') as f:
            json.dump({'final_features': final_features, 'regime_features': regime_features, 'timestamp': datetime.now(timezone.utc).isoformat()}, f, indent=2)
        with open('models/training_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        with open('models/training_config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
            
        logger.info("   ⚙️ [SAVED] 'models/lstm_model.keras' saved.")
        logger.info("   ⚙️ [SAVED] 'models/ensemble_model.pkl' saved.")
        if ppo:
            logger.info("   ⚙️ [SAVED] 'models/ppo_agent' policy path stored.")
        logger.info("   ⚙️ [SAVED] 'models/final_features.json' configuration matrices exported.")

    def run(self) -> dict:
        self.start_time = time.time()
        logger.info("========================================================================")
        logger.info("🌌 QUANTUM HFT SYSTEM TRADING ENGINE : PART 2 TRAINING PIPELINE MODULE")
        logger.info("========================================================================")

        try:
            # STEP 1
            self._show_banner("Data Engine Initialization & Parsing", 1)
            df_raw = self.load_data()

            split_pct = self.config.get('train_split', 0.8)
            split_idx = int(len(df_raw) * split_pct)
            window_size = self.config.get('window', 120)

            logger.info(f"   ├─ Temporal Matrix Cut: {split_pct*100:.0f}% Training Slice vs {(1-split_pct)*100:.0f}% Forward Validation.")
            
            df_train_raw = df_raw.iloc[:split_idx].copy().reset_index(drop=True)
            df_val_raw = df_raw.iloc[split_idx - window_size:].copy().reset_index(drop=True)

            if 'timestamp' in df_train_raw.columns:
                df_train_raw = df_train_raw.set_index(pd.to_datetime(df_train_raw['timestamp'], utc=True))
            if 'timestamp' in df_val_raw.columns:
                df_val_raw = df_val_raw.set_index(pd.to_datetime(df_val_raw['timestamp'], utc=True))

            # --------------------------------------------------------------
            # ✅ FIX 1: FeatureEngine INSTANCE banaya (build() error khatam)
            # --------------------------------------------------------------
            fe = FeatureEngine(cfg=self.config)

            # STEP 2
            self._show_banner("Feature Engineering Layer (Training Pipeline Matrix)", 2)
            logger.info("   ⚡ Running multi-threaded FeatureEngine calculation calculations...")
            with tqdm(total=1, desc="   Processing Train Features", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
                df_train_feats = fe.build_all(df_train_raw.copy())
                pbar.update(1)

            # [RECOVERY STEP] - MRD (Regime Detector) ke liye OHLCV wapas add karo
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df_train_raw.columns:
                    df_train_feats[col] = df_train_raw[col]
            df_train_feats['timestamp'] = df_train_feats.index
            df_train_feats = df_train_feats.reset_index(drop=True).replace([np.inf, -np.inf], np.nan).ffill().bfill()

            # STEP 3
            self._show_banner("Feature Engineering Layer (Validation Pipeline Matrix)", 3)
            logger.info("   ⚡ Injecting validation index frame buffer to isolate history leakage...")
            with tqdm(total=1, desc="   Processing Val Features  ", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
                df_val_feats = fe.build_all(df_val_raw.copy())
                pbar.update(1)

            # [RECOVERY STEP] - Validation set ke liye bhi OHLCV wapas add karo
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df_val_raw.columns:
                    df_val_feats[col] = df_val_raw[col]
            df_val_feats['timestamp'] = df_val_feats.index
            df_val_feats = df_val_feats.reset_index(drop=True).replace([np.inf, -np.inf], np.nan).ffill().bfill()

            # STEP 4
            self._show_banner("Unsupervised Market Regime Mapping", 4)
            logger.info("   ⚡ Fitting Gaussian Mixture Clustering Model strictly onto Training Splits...")
            regime_detector = MarketRegimeDetector(self.config)
            regime_detector.fit(df_train_feats)  # Ab isme OHLCV hain, is liye error nahi aayega
            
            df_train_feats = regime_detector.annotate(df_train_feats)
            df_val_feats = regime_detector.annotate(df_val_feats)
            regime_detector.save_map()
            self.stats['regime_fitted'] = True
            logger.info("   🎯 Contextual market states generated and aligned to active rows.")

            # --------------------------------------------------------------
            # ✅ FIX 2: Annotate ke BAAD OHLCV HATAYA (close leakage block)
            # --------------------------------------------------------------
            df_train_feats = df_train_feats.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore')
            df_val_feats = df_val_feats.drop(columns=['open', 'high', 'low', 'close', 'volume'], errors='ignore')

            regime_features = self.config.get('regime_features', ['log_ret_20', 'log_ret_5', 'natr', 'adx', 'hurst_exp', 'vol_ratio', 'rsi'])

            # Direct Routing Feature Extraction
            protected_infrastructure = ['timestamp', self.config.get('target_col', 'target')]
            final_features = [c for c in df_train_feats.columns if c not in protected_infrastructure]
            self.stats['feature_engine_columns_count'] = len(final_features)
            
            logger.info(f"   🎯 Direct Routing Mode Active: Synchronized all {len(final_features)} features to downstream nets.")
            logger.info(f"   🔥 (Close column successfully blocked from reaching LSTM/Ensemble/PPO)")

            df_cols_to_keep = final_features.copy()
            target_col = self.config.get('target_col', 'target')
            if target_col in df_train_feats.columns and target_col not in df_cols_to_keep:
                df_cols_to_keep.append(target_col)
            if 'timestamp' in df_train_feats.columns and 'timestamp' not in df_cols_to_keep:
                df_cols_to_keep.append('timestamp')

            df_train_final = df_train_feats[df_cols_to_keep].copy().dropna()
            df_val_final = df_val_feats[df_cols_to_keep].iloc[window_size:].copy().reset_index(drop=True).dropna()

            # Execute Model Routines
            lstm_model = self.train_lstm(df_train_final, df_val_final, final_features)
            ensemble_model = self.train_ensemble(df_train_final)
            ppo_model = self.train_ppo(df_train_final, lstm_model, final_features)

            self.save_artifacts(lstm_model, ensemble_model, ppo_model, final_features, regime_features)

            self.stats['duration_seconds'] = round(time.time() - self.start_time, 2)
            self.stats['success'] = True

            # Final Summary Screen Block
            logger.info(" ")
            logger.info("========================================================================")
            logger.info("★ ✅ PART 2 – PIPELINE CORES SUCCESSFULLY DISPATCHED & LOGGED ★")
            logger.info("========================================================================")
            logger.info(f"   ⚡ Total Runtime Profile : {self.stats['duration_seconds']} seconds")
            logger.info(f"   ⚡ Bars Engine Evaluated : {self.stats['ohlcv_bars']} rows")
            logger.info(f"   ⚡ Global Active Signals : {self.stats['feature_engine_columns_count']} items")
            logger.info(f"   ⚡ Optuna Optimization  : {self.stats['lstm_tuned_with_optuna']}")
            logger.info(f"   ⚡ Deep LSTM Subnet     : {self.stats['lstm_trained']}")
            logger.info(f"   ⚡ Boosted Ensemble Net : {self.stats['ensemble_trained']}")
            logger.info(f"   ⚡ Actor-Critic RL PPO  : {self.stats['ppo_trained']}")
            logger.info("========================================================================")
            return self.stats

        except Exception as e:
            self.stats['success'] = False
            self.stats['error'] = str(e)
            logger.error(f"❌ Core Pipeline routing crashed: {e}", exc_info=True)
            return self.stats


def main():
    parser = argparse.ArgumentParser(description='Part 2: Model Training Pipeline Architecture')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    args = parser.parse_args()

    pipeline = TrainingPipeline(config_path=args.config)
    result = pipeline.run()
    return 0 if result['success'] else 1


if __name__ == '__main__':
    exit(main())