import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger('OptunaTuner')

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    from optuna.integration import TFKerasPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError as e:
    print(f"Optuna import error: {e}")
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed.")

class OptunaTuner:
    
    def __init__(self, cfg: Dict = None):
        self.cfg = cfg or {}
        self.best_params: Dict[str, Any] = {}
        self.study = None
        
        self.enabled = self.cfg.get('optuna_enabled', True)
        self.n_trials = self.cfg.get('optuna_trials', 15)
        self.n_epochs = self.cfg.get('optuna_epochs', 10)

    def tune(self, X_train: np.ndarray, y_train_dict: Dict,
             X_val: np.ndarray, y_val_dict: Dict) -> Dict:
        
        if not self.enabled or not OPTUNA_AVAILABLE:
            logger.info("Optuna disabled or not available — using default parameters.")
            return self._defaults()
        
        if X_train.shape[0] < 50 or X_val.shape[0] < 50:
            logger.warning(f"Insufficient data: train={X_train.shape[0]}, val={X_val.shape[0]}. Using defaults.")
            return self._defaults()
        
        logger.info(f"Starting Optuna hyperparameter tuning: {self.n_trials} trials, {self.n_epochs} epochs each.")
        
        def objective(trial):
            lu1 = trial.suggest_categorical('lstm_units_1', [128, 192, 256])
            lu2 = trial.suggest_categorical('lstm_units_2', [64, 96, 128])
            dr = trial.suggest_float('dropout_rate', 0.1, 0.4, step=0.05)
            lr = trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True)
            ah = trial.suggest_categorical('attention_heads', [4, 8])
            akd = trial.suggest_categorical('attention_key_dim', [32, 64])
            batch_size = trial.suggest_categorical('batch_size', [16, 32])
            
            params = {
                **self.cfg,
                'lstm_units_1': lu1,
                'lstm_units_2': lu2,
                'dropout_rate': dr,
                'learning_rate': lr,
                'attention_heads': ah,
                'attention_key_dim': akd,
                'batch_size': batch_size,
                'epochs': self.n_epochs
            }
            
            try:
                from prediction_model import PredictionModel
                from tensorflow.keras.callbacks import EarlyStopping
                import tensorflow as tf
                
                model = PredictionModel(params)
                model.build((X_train.shape[1], X_train.shape[2]))
                
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0),
                    TFKerasPruningCallback(trial, 'val_loss')
                ]
                
                history = model.model.fit(
                    X_train, y_train_dict,
                    validation_data=(X_val, y_val_dict),
                    epochs=self.n_epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    shuffle=False,
                    verbose=0
                )
                
                val_loss = min(history.history.get('val_loss', [9999]))
                if np.isnan(val_loss) or np.isinf(val_loss):
                    return 9999.0
                return float(val_loss)
                
            except Exception as e:
                if "TrialPruned" in str(type(e)):
                    raise e
                logger.warning(f"Trial failed: {e}")
                return 9999.0
            finally:
                import tensorflow as tf
                tf.keras.backend.clear_session()
        
        try:
            pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)
            sampler = TPESampler(seed=42)
            
            self.study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
                pruner=pruner,
                study_name='lstm_tuning'
            )
            
            self.study.optimize(
                objective,
                n_trials=self.n_trials,
                show_progress_bar=True,
                n_jobs=1
            )
            
            self.best_params = self.study.best_params
            logger.info(f"Best params: {self.best_params}")
            logger.info(f"Best validation loss: {self.study.best_value:.4f}")
            
            tuned_cfg = {**self.cfg, **self.best_params}
            
            if self.study.best_value > 10:
                logger.warning(f"Best loss {self.study.best_value:.4f} too high, using defaults")
                return self._defaults()
            
            return tuned_cfg
            
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            return self._defaults()

    def _defaults(self) -> Dict:
        return {
            'lstm_units_1': self.cfg.get('lstm_units_1', 128),
            'lstm_units_2': self.cfg.get('lstm_units_2', 64),
            'dropout_rate': self.cfg.get('dropout_rate', 0.2),
            'learning_rate': self.cfg.get('learning_rate', 0.001),
            'attention_heads': self.cfg.get('attention_heads', 4),
            'attention_key_dim': self.cfg.get('attention_key_dim', 32),
            'batch_size': self.cfg.get('batch_size', 32),
        }

    def get_best_params(self) -> Dict:
        return self.best_params if self.best_params else self._defaults()

    def get_trials_dataframe(self) -> pd.DataFrame:
        if self.study is not None and OPTUNA_AVAILABLE:
            try:
                return self.study.trials_dataframe()
            except Exception as e:
                logger.warning(f"Could not get trials dataframe: {e}")
        return pd.DataFrame()
