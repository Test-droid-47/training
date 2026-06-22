import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any

class TradingEnvironment:
    """ RL environment with trade-aware state for manual exit (action 3). """
    
    # Number of extra trade features appended to each bar
    N_TRADE_FEATURES = 8
    
    def __init__(self, df: pd.DataFrame, scaled_features: np.ndarray, cfg: Dict, close_idx: int):
        self.df = df.reset_index(drop=True)
        self.features = scaled_features          # shape (n_bars, base_features)
        self.cfg = cfg
        self.close_idx = close_idx
        self.window = cfg.get('window', 120)
        self.trading_mode = cfg.get('trading_mode', 'spot')
        self.leverage = cfg.get('leverage', 10) if self.trading_mode == 'future' else 1
        
        if self.trading_mode == 'future':
            self.fee_rate = cfg.get('fee_rate', 0.0004)
        else:
            self.fee_rate = cfg.get('fee_rate', 0.001)
        
        self.slippage = cfg.get('slippage', 0.0005)
        self.initial_capital = cfg.get('initial_capital', 10000.0)
        self.max_risk_per_trade = cfg.get('max_risk_per_trade', 0.02)
        self.max_position_pct = cfg.get('max_position_pct', 0.5)
        self.drawdown_penalty = cfg.get('drawdown_penalty', 2.0)
        self.invalid_action_penalty = cfg.get('invalid_action_penalty', -0.05)
        
        self.n_bars = len(df)
        self.base_feature_count = self.features.shape[1]
        self.state_shape = (self.window, self.base_feature_count + self.N_TRADE_FEATURES)
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_idx = self.window - 1
        self.capital = self.initial_capital
        self.margin_locked = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_idx = 0
        self.peak_capital = self.initial_capital
        self.done = False
        self.trades = []
        self.returns_history = []
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.trailing_activated = False
        self.trailing_peak = 0.0
        self.dynamic_sl = 0.0
        self.dynamic_tp = 0.0
        self.position_side = 'long'
        # trade tracking
        self.mfe = 0.0          # max favorable excursion (percent)
        self.mae = 0.0          # max adverse excursion (percent)
        return self._get_state()

    # ---------- Private market helpers (unchanged) ----------
    def _get_state(self) -> np.ndarray:
        end = self.current_idx + 1
        start = end - self.window
        if start < 0 or end > len(self.features):
            logging.warning(f"State out of bounds fallback triggered. start: {start}, end: {end}")
            return np.zeros(self.state_shape, dtype=np.float32)
        
        base_state = self.features[start:end].astype(np.float32)          # (window, base)
        trade_vec = self._get_trade_feature_vector().astype(np.float32)    # (N_TRADE_FEATURES,)
        trade_state = np.tile(trade_vec, (self.window, 1))                # (window, N_TRADE_FEATURES)
        return np.concatenate([base_state, trade_state], axis=1)

    def _get_trade_feature_vector(self) -> np.ndarray:
        """ Return current trade context as a flat vector of shape (N_TRADE_FEATURES,). """
        if self.position == 0:
            return np.zeros(self.N_TRADE_FEATURES, dtype=np.float32)
        
        price = self._current_price()
        # Duration
        duration = min(1.0, (self.current_idx - self.entry_idx) / self.window)
        # Unrealized PnL %
        if self.position_side == 'long':
            pnl_pct = (price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price
        
        # Distance to SL/TP (percentage, positive = away from stop in profitable direction)
        if self.position_side == 'long':
            dist_sl = (price - self.dynamic_sl) / price
            dist_tp = (self.dynamic_tp - price) / price
        else:
            dist_sl = (self.dynamic_sl - price) / price
            dist_tp = (price - self.dynamic_tp) / price
        
        trailing_flag = 1.0 if self.trailing_activated else 0.0
        side_flag = 1.0 if self.position_side == 'long' else -1.0
        
        return np.array([
            duration,
            pnl_pct,
            self.mfe,
            self.mae,
            dist_sl,
            dist_tp,
            trailing_flag,
            side_flag
        ], dtype=np.float32)

    def _update_excursions(self, price: float):
        """ Track MFE/MAE during a trade. """
        if self.position == 0:
            return
        if self.position_side == 'long':
            current_mfe = (price - self.entry_price) / self.entry_price
            current_mae = (self.entry_price - price) / self.entry_price
        else:
            current_mfe = (self.entry_price - price) / self.entry_price
            current_mae = (price - self.entry_price) / self.entry_price
        
        self.mfe = max(self.mfe, current_mfe)
        self.mae = max(self.mae, current_mae)

    # ---------- Market indicators (unchanged) ----------
    def _current_price(self) -> float:
        return float(self.df['close'].iloc[self.current_idx])

    def _current_high(self) -> float:
        return float(self.df['high'].iloc[self.current_idx])

    def _current_low(self) -> float:
        return float(self.df['low'].iloc[self.current_idx])

    def _get_atr(self) -> float:
        if 'atr' in self.df.columns and self.current_idx < len(self.df):
            return float(self.df['atr'].iloc[self.current_idx])
        return self._current_price() * 0.01

    def _get_atr_mean(self) -> float:
        if 'atr' in self.df.columns and self.current_idx >= 50:
            return float(self.df['atr'].iloc[max(0, self.current_idx-50):self.current_idx].mean())
        return self._get_atr()

    def _get_adx(self) -> float:
        if 'adx' in self.df.columns and self.current_idx < len(self.df):
            return float(self.df['adx'].iloc[self.current_idx])
        return 25.0

    def _get_hurst(self) -> float:
        if 'hurst_exp' in self.df.columns and self.current_idx < len(self.df):
            return float(self.df['hurst_exp'].iloc[self.current_idx])
        return 0.5

    def _get_regime(self) -> int:
        if 'regime' in self.df.columns and self.current_idx < len(self.df):
            return int(self.df['regime'].iloc[self.current_idx])
        return 0

    def _get_dynamic_sl_tp_pct(self) -> Tuple[float, float]:
        atr = self._get_atr()
        atr_mean = self._get_atr_mean()
        adx = self._get_adx()
        hurst = self._get_hurst()
        regime = self._get_regime()
        vol_ratio = atr / (atr_mean + 1e-10)
        
        if vol_ratio > 1.5:
            sl_pct = 0.035
            tp_pct = 0.035
        elif vol_ratio < 0.7:
            sl_pct = 0.015
            tp_pct = 0.045
        else:
            sl_pct = 0.02
            tp_pct = 0.04
        
        if adx > 35:
            tp_pct *= 1.3
        elif adx > 25:
            tp_pct *= 1.15
        elif adx < 20:
            sl_pct *= 0.85
            tp_pct *= 0.85
        
        if hurst > 0.6:
            tp_pct *= 1.2
        elif hurst < 0.4:
            sl_pct *= 1.15
            tp_pct *= 0.9
        
        if regime in (1, 2):
            tp_pct *= 1.15
        elif regime == 3:
            sl_pct *= 0.9
            tp_pct *= 0.85
        
        sl_pct = np.clip(sl_pct, 0.01, 0.05)
        tp_pct = np.clip(tp_pct, 0.02, 0.08)
        return sl_pct, tp_pct

    def _get_liquidation_price(self, entry_price: float, side: str) -> float:
        if self.trading_mode != 'future':
            return 0.0
        if side == 'long':
            return entry_price * (1 - 1/self.leverage)
        else:
            return entry_price * (1 + 1/self.leverage)

    def _get_position_size(self, capital: float, sl_pct: float) -> float:
        current_risk_multiplier = 0.5 if self.consecutive_losses >= 2 else 1.0
        active_risk_pct = self.max_risk_per_trade * current_risk_multiplier
        
        risk_amount = capital * active_risk_pct
        target_notional = risk_amount / (sl_pct + 1e-10)
        
        if self.trading_mode == 'future':
            max_allowed_notional = capital * self.max_position_pct * self.leverage
        else:
            max_allowed_notional = capital * self.max_position_pct
            
        notional = min(target_notional, max_allowed_notional)
        return max(notional, capital * 0.01)

    def _update_trailing(self, current_price: float, current_high: float, current_low: float,
                         entry_price: float, side: str) -> Tuple[float, bool]:
        if not self.trailing_activated:
            pnl_pct = (current_price - entry_price) / entry_price if side == 'long' else (entry_price - current_price) / entry_price
            if pnl_pct >= 0.02:
                self.trailing_activated = True
                self.trailing_peak = current_high if side == 'long' else current_low
            else:
                return self.dynamic_sl, False
        
        if side == 'long':
            if current_high > self.trailing_peak:
                self.trailing_peak = current_high
            trail_pct = 0.01
            new_sl = self.trailing_peak * (1 - trail_pct)
        else:
            if current_low < self.trailing_peak:
                self.trailing_peak = current_low
            trail_pct = 0.01
            new_sl = self.trailing_peak * (1 + trail_pct)
        
        if (side == 'long' and new_sl > self.dynamic_sl) or (side == 'short' and new_sl < self.dynamic_sl):
            return new_sl, self.trailing_activated
        return self.dynamic_sl, self.trailing_activated

    def get_portfolio_value(self) -> float:
        price = self._current_price()
        if self.position == 0:
            return self.capital
        if self.trading_mode == 'spot':
            return self.capital + self.position * price
        else:
            unrealized_pnl = self.position * (price - self.entry_price) if self.position_side == 'long' else self.position * (self.entry_price - price)
            return self.capital + self.margin_locked + unrealized_pnl

    # ---------- Main step (fully upgraded) ----------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action processed: {action}. Must be inside discrete interval [0, 1, 2, 3].")

        self.current_idx += 1
        if self.current_idx >= self.n_bars - 1:
            self.done = True

        price = self._current_price()
        high = self._current_high()
        low = self._current_low()
        reward = 0.0

        # Track excursions if in position (before any exit logic)
        if self.position > 0:
            self._update_excursions(price)

        # ---------- Flat state: entries ----------
        if self.position == 0:
            long_action = (action == 1)
            short_action = (action == 2 and self.trading_mode == 'future')
            if long_action or short_action:
                side = 'long' if long_action else 'short'
                sl_pct, tp_pct = self._get_dynamic_sl_tp_pct()
                notional_exposure = self._get_position_size(self.capital, sl_pct)
                entry_price = price * (1 + self.slippage) if side == 'long' else price * (1 - self.slippage)
                size = notional_exposure / entry_price

                if self.trading_mode == 'future':
                    margin = notional_exposure / self.leverage
                    entry_fee = notional_exposure * self.fee_rate
                    if margin + entry_fee <= self.capital:
                        self.capital -= (margin + entry_fee)
                        self.margin_locked = margin
                        self.position = size
                        self.entry_price = entry_price
                        self.entry_idx = self.current_idx
                        self.position_side = side
                        self.trailing_activated = False
                        self.mfe = 0.0
                        self.mae = 0.0
                        if side == 'long':
                            self.dynamic_sl = entry_price * (1 - sl_pct)
                            self.dynamic_tp = entry_price * (1 + tp_pct)
                        else:
                            self.dynamic_sl = entry_price * (1 + sl_pct)
                            self.dynamic_tp = entry_price * (1 - tp_pct)
                    else:
                        reward = self.invalid_action_penalty
                else:  # spot
                    cost = notional_exposure * (1 + self.fee_rate)
                    if cost <= self.capital:
                        self.capital -= cost
                        self.position = size
                        self.entry_price = entry_price
                        self.entry_idx = self.current_idx
                        self.position_side = 'long'
                        self.trailing_activated = False
                        self.mfe = 0.0
                        self.mae = 0.0
                        self.dynamic_sl = entry_price * (1 - sl_pct)
                        self.dynamic_tp = entry_price * (1 + tp_pct)
                    else:
                        reward = self.invalid_action_penalty

        # ---------- Position active: exits, stops, liquidation ----------
        elif self.position > 0:
            # Liquidation check
            if self.trading_mode == 'future':
                liq_price = self._get_liquidation_price(self.entry_price, self.position_side)
                is_liquidated = (self.position_side == 'long' and low <= liq_price) or (self.position_side == 'short' and high >= liq_price)
                if is_liquidated:
                    reward = -0.5
                    wiped_margin = self.margin_locked
                    self.margin_locked = 0.0
                    self.trades.append({
                        'entry_idx': self.entry_idx,
                        'exit_idx': self.current_idx,
                        'entry_price': self.entry_price,
                        'exit_price': liq_price,
                        'pnl_pct': -1.0,
                        'pnl_cash': -wiped_margin,
                        'mae': self.mae,
                        'mfe': self.mfe
                        'bars_held': self.current_idx - self.entry_idx,
                        'exit_reason': 'liquidation',
                        'side': self.position_side
                    })
                    self.returns_history.append(-1.0)
                    self.position = 0.0
                    self.entry_price = 0.0
                    self.trailing_activated = False
                    self.mfe = 0.0
                    self.mae = 0.0
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0

                    # Drawdown penalty on liquidation bar
                    port_val = self.get_portfolio_value()
                    if port_val > self.peak_capital:
                        self.peak_capital = port_val
                    drawdown = (self.peak_capital - port_val) / (self.peak_capital + 1e-10)
                    if drawdown > 0.2:
                        reward -= self.drawdown_penalty * drawdown
                    return self._get_state(), reward, self.done

            # Trailing update
            new_sl, trail_activated = self._update_trailing(price, high, low, self.entry_price, self.position_side)
            if new_sl != self.dynamic_sl:
                self.dynamic_sl = new_sl
                self.trailing_activated = trail_activated

            # Exit logic (SL, TP, manual)
            should_exit = False
            exit_price = 0.0
            exit_reason = ""

            if self.position_side == 'long':
                if low <= self.dynamic_sl:
                    exit_price = self.dynamic_sl * (1 - self.slippage)
                    exit_reason = "stop_loss"
                    should_exit = True
                elif high >= self.dynamic_tp:
                    exit_price = self.dynamic_tp * (1 - self.slippage)
                    exit_reason = "take_profit"
                    should_exit = True
                elif action == 3:
                    exit_price = price * (1 - self.slippage)
                    exit_reason = "manual_sell"
                    should_exit = True
            else:  # short
                if high >= self.dynamic_sl:
                    exit_price = self.dynamic_sl * (1 + self.slippage)
                    exit_reason = "stop_loss"
                    should_exit = True
                elif low <= self.dynamic_tp:
                    exit_price = self.dynamic_tp * (1 + self.slippage)
                    exit_reason = "take_profit"
                    should_exit = True
                elif action == 3:
                    exit_price = price * (1 + self.slippage)
                    exit_reason = "manual_cover"
                    should_exit = True

            if should_exit:
                if self.trading_mode == 'future':
                    realized_pnl = self.position * (self.entry_price - exit_price) if self.position_side == 'short' else self.position * (exit_price - self.entry_price)
                    exit_fee = (self.position * exit_price) * self.fee_rate
                    entry_fee = (self.position * self.entry_price) * self.fee_rate
                    margin = self.margin_locked
                    pnl_pct = realized_pnl / (margin + 1e-10)
                    total_friction_pct = (entry_fee + exit_fee) / (margin + 1e-10)
                    reward = pnl_pct - total_friction_pct
                    pnl_cash = realized_pnl - (entry_fee + exit_fee)
                    self.capital += (margin + realized_pnl - exit_fee)
                    self.margin_locked = 0.0
                else:
                    gross = self.position * exit_price
                    exit_fee = gross * self.fee_rate
                    entry_cost = self.entry_price * self.position
                    net = gross - exit_fee
                    self.capital += net
                    pnl_pct = (exit_price - self.entry_price) / self.entry_price
                    pnl_cash = net - entry_cost
                    reward = pnl_pct - (exit_fee / gross)

                self.trades.append({
                    'entry_idx': self.entry_idx,
                    'exit_idx': self.current_idx,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_cash': pnl_cash,
                    'mae': self.mae,
                    'mfe': self.mfe,
                    'bars_held': self.current_idx - self.entry_idx,
                    'exit_reason': exit_reason,
                    'side': self.position_side
                })
                self.returns_history.append(pnl_pct)
                if pnl_pct > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                self.position = 0.0
                self.entry_price = 0.0
                self.trailing_activated = False
                self.mfe = 0.0
                self.mae = 0.0

        # Global drawdown penalty after any exit
        port_val = self.get_portfolio_value()
        if port_val > self.peak_capital:
            self.peak_capital = port_val
        drawdown = (self.peak_capital - port_val) / (self.peak_capital + 1e-10)
        if drawdown > 0.2:
            reward -= self.drawdown_penalty * drawdown

        # Forced exit at episode end
        if self.done and self.position > 0:
            terminal_price = self._current_price()
            exit_price = terminal_price * (1 - self.slippage) if self.position_side == 'long' else terminal_price * (1 + self.slippage)
            if self.trading_mode == 'future':
                realized_pnl = self.position * (self.entry_price - exit_price) if self.position_side == 'short' else self.position * (exit_price - self.entry_price)
                exit_fee = (self.position * exit_price) * self.fee_rate
                entry_fee = (self.position * self.entry_price) * self.fee_rate
                margin = self.margin_locked
                actual_pnl_pct = realized_pnl / (margin + 1e-10) if margin != 0 else 0.0
                total_friction_pct = (entry_fee + exit_fee) / (margin + 1e-10) if margin != 0 else 0.0
                reward = actual_pnl_pct - total_friction_pct
                pnl_cash = realized_pnl - (entry_fee + exit_fee)
                self.capital += (margin + realized_pnl - exit_fee)
                self.margin_locked = 0.0
            else:
                gross = self.position * exit_price
                exit_fee = gross * self.fee_rate
                entry_cost = self.entry_price * self.position
                net = gross - exit_fee
                self.capital += net
                actual_pnl_pct = (exit_price - self.entry_price) / self.entry_price
                pnl_cash = net - entry_cost
                reward = actual_pnl_pct - (exit_fee / gross)

            self.trades.append({
                'entry_idx': self.entry_idx,
                'exit_idx': self.current_idx,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'pnl_pct': actual_pnl_pct,
                'pnl_cash': pnl_cash,
                'mae': self.mae,
                'mfe': self.mfe,
                'bars_held': self.current_idx - self.entry_idx,
                'exit_reason': 'forced_exit',
                'side': self.position_side
            })
            if actual_pnl_pct != 0.0:
                self.returns_history.append(actual_pnl_pct)
            self.position = 0.0
            self.entry_price = 0.0
            self.trailing_activated = False
            self.mfe = 0.0
            self.mae = 0.0

            # Re‑apply drawdown penalty after forced exit
            port_val = self.get_portfolio_value()
            if port_val > self.peak_capital:
                self.peak_capital = port_val
            drawdown = (self.peak_capital - port_val) / (self.peak_capital + 1e-10)
            if drawdown > 0.2:
                reward -= self.drawdown_penalty * drawdown

        return self._get_state(), reward, self.done

    def get_trade_statistics(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_return': (self.get_portfolio_value() - self.initial_capital) / self.initial_capital
            }
        wins_cash = [t['pnl_cash'] for t in self.trades if t['pnl_cash'] > 0]
        losses_cash = [t['pnl_cash'] for t in self.trades if t['pnl_cash'] <= 0]
        total_trades = len(self.trades)
        win_rate = len(wins_cash) / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(wins_cash) if wins_cash else 0.0
        avg_loss = abs(np.mean(losses_cash)) if losses_cash else 0.0
        gross_profit = sum(wins_cash) if wins_cash else 0.0
        gross_loss = abs(sum(losses_cash)) if losses_cash else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        final_value = self.get_portfolio_value()
        total_return = (final_value - self.initial_capital) / self.initial_capital

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'final_capital': final_value
        }