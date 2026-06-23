import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger('OptunaTuner')

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    # FIX: Legacy TFKerasPruningCallback removed as it crashes on Keras 3 multi-output
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Hyperparameter tuning disabled.")

# FIX: Professional Keras 3 Native Pruning Callback to handle multi-output gracefully
import tensorflow as tf
class Keras3OptunaPruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial, monitor: str = 'val_loss'):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get(self.monitor)
        if val_loss is not None:
            self.trial.report(float(val_loss), step=epoch)
            if self.trial.should_prune():
                message = f"Trial pruned at epoch {epoch}."
                raise optuna.TrialPruned(message)

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
                from sklearn.utils.class_weight import compute_class_weight
                
                model = PredictionModel(params)
                model.build((X_train.shape[1], X_train.shape[2]))
                
                # FIX: Injected balanced class weights so Optuna optimizes for real features, not majority class guessing
                classes = np.unique(y_train_dict['direction'])
                computed_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_dict['direction'])
                class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, computed_weights)}
                multi_output_class_weights = {'direction': class_weight_dict}
                
                # FIX: Swapped to our robust Keras 3 native pruning callback
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0),
                    Keras3OptunaPruningCallback(trial, 'val_loss')
                ]
                
                history = model.model.fit(
                    X_train, y_train_dict,
                    validation_data=(X_val, y_val_dict),
                    epochs=self.n_epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    class_weight=multi_output_class_weights,  # FIX: Passed balanced weights to fit
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
            
            # FIX: Adjusted threshold filter safety buffer to accommodate multi-task combined loss ranges
            if self.study.best_value > 50.0:
                logger.warning(f"Best loss {self.study.best_value:.4f} too high, using defaults")
                return self._defaults()
            
            return tuned_cfg
            
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            return self._defaults()

    def _defaults(self) -> Dict:
        # FIX: Ensure fallback dictionary preserves the entire context of self.cfg instead of wiping metadata fields
        defaults_dict = {
            'lstm_units_1': self.cfg.get('lstm_units_1', 128),
            'lstm_units_2': self.cfg.get('lstm_units_2', 64),
            'dropout_rate': self.cfg.get('dropout_rate', 0.2),
            'learning_rate': self.cfg.get('learning_rate', 0.001),
            'attention_heads': self.cfg.get('attention_heads', 4),
            'attention_key_dim': self.cfg.get('attention_key_dim', 32),
            'batch_size': self.cfg.get('batch_size', 32),
        }
        return {**self.cfg, **defaults_dict}

    def get_best_params(self) -> Dict:
        return self.best_params if self.best_params else self._defaults()

    def get_trials_dataframe(self) -> pd.DataFrame:
        if self.study is not None and OPTUNA_AVAILABLE:
            try:
                return self.study.trials_dataframe()
            except Exception as e:
                logger.warning(f"Could not get trials dataframe: {e}")
        return pd.DataFrame()
                
