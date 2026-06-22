import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

class PPOAgent:
    
    def __init__(self, cfg: Dict = None, state_shape: Tuple = env.state_shape, n_actions: int = 4):
        self.cfg = cfg or {}
        self.gamma = self.cfg.get('rl_gamma', 0.99)
        self.clip_eps = self.cfg.get('rl_clip_epsilon', 0.2)
        self.ppo_epochs = self.cfg.get('rl_ppo_epochs', 10)
        self.ent_coeff = self.cfg.get('rl_entropy_coeff', 0.01)
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        self.actor = None
        self.critic = None
        self.actor_opt = None
        self.critic_opt = None
        
        self.recent_performance = deque(maxlen=168)
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.last_7_days_win_rate = 0.5
        self.last_7_days_sharpe = 0.0
        self.weekly_confidence_boost = 0.0
        self.weekly_position_multiplier = 1.0
        
        if state_shape is not None:
            self._build_networks()

    def _build_networks(self):
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.actor_opt = Adam(learning_rate=1e-4, clipnorm=1.0)
        self.critic_opt = Adam(learning_rate=1e-3, clipnorm=1.0)

    def _build_actor(self) -> Model:
        inp = Input(shape=self.state_shape)
        x = LSTM(128, return_sequences=True)(inp)
        x = LSTM(64, return_sequences=False)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(self.n_actions, activation='softmax', name='action_probs')(x)
        return Model(inp, out)

    def _build_critic(self) -> Model:
        inp = Input(shape=self.state_shape)
        x = LSTM(128, return_sequences=True)(inp)
        x = LSTM(64, return_sequences=False)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        out = Dense(1, activation='linear', name='state_value')(x)
        return Model(inp, out)

    def _compute_returns_advantages(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        lam = 0.95
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        values = values + [0.0]
        
        for t in reversed(range(n)):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - int(dones[t])) - values[t]
            gae = delta + self.gamma * lam * (1 - int(dones[t])) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        return returns, advantages

    def collect_trajectory(self, env, max_steps=500) -> Dict[str, np.ndarray]:
        state = env.reset()
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            s_t = tf.constant(state[np.newaxis], tf.float32)
            probs = self.actor(s_t, training=False).numpy()[0]
            action = int(np.random.choice(self.n_actions, p=probs))
            log_prob = float(np.log(probs[action] + 1e-10))
            value = float(self.critic(s_t, training=False).numpy()[0, 0])
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            state = next_state
            steps += 1
        
        returns, advantages = self._compute_returns_advantages(rewards, values, dones)
        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.int32),
            'log_probs': np.array(log_probs, dtype=np.float32),
            'returns': returns.astype(np.float32),
            'advantages': advantages.astype(np.float32),
            'total_reward': float(np.sum(rewards))
        }

    @tf.function
    def _train_step(self, s_t, a_t, lp_t, r_t, adv_t):
        with tf.GradientTape() as at:
            probs = self.actor(s_t, training=True)
            batch_sz = tf.shape(probs)[0]
            indices_act = tf.stack([tf.range(batch_sz), a_t], axis=1)
            taken_probs = tf.gather_nd(probs, indices_act)
            lp = tf.math.log(taken_probs + 1e-10)
            ratio = tf.exp(lp - lp_t)
            surr1 = ratio * adv_t
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
            a_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - self.ent_coeff * entropy
        
        a_grads = at.gradient(a_loss, self.actor.trainable_variables)
        a_grads = [tf.clip_by_norm(g, 0.5) for g in a_grads if g is not None]
        a_vars = [v for v in self.actor.trainable_variables if v is not None]
        if a_grads and a_vars:
            self.actor_opt.apply_gradients(zip(a_grads, a_vars))
        
        with tf.GradientTape() as ct:
            v_pred = tf.squeeze(self.critic(s_t, training=True), axis=-1)
            c_loss = tf.reduce_mean(tf.square(r_t - v_pred))
        
        c_grads = ct.gradient(c_loss, self.critic.trainable_variables)
        c_grads = [tf.clip_by_norm(g, 0.5) for g in c_grads if g is not None]
        c_vars = [v for v in self.critic.trainable_variables if v is not None]
        if c_grads and c_vars:
            self.critic_opt.apply_gradients(zip(c_grads, c_vars))
        
        return a_loss, c_loss

    def update(self, states: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray,
               returns: np.ndarray, advantages: np.ndarray) -> Tuple[float, float]:
        
        total_a_loss = 0.0
        total_c_loss = 0.0
        batch_size = 256
        n_samples = len(states)
        
        for _ in range(self.ppo_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                
                s_t = tf.constant(states[batch_idx], tf.float32)
                a_t = tf.constant(actions[batch_idx], tf.int32)
                lp_t = tf.constant(old_log_probs[batch_idx], tf.float32)
                r_t = tf.constant(returns[batch_idx], tf.float32)
                adv_t = tf.constant(advantages[batch_idx], tf.float32)
                
                a_loss, c_loss = self._train_step(s_t, a_t, lp_t, r_t, adv_t)
                total_a_loss += a_loss.numpy()
                total_c_loss += c_loss.numpy()
        
        n_epochs = max(1, self.ppo_epochs)
        batches_per_epoch = (n_samples + batch_size - 1) // batch_size
        total_updates = n_epochs * batches_per_epoch
        return total_a_loss / total_updates, total_c_loss / total_updates

    def train(self, env, n_episodes: int = None) -> List[float]:
        n_ep = n_episodes or self.cfg.get('rl_n_episodes', 200)
        print(f"Training {n_ep} episodes...")
        rewards_history = []
        best_reward = -np.inf
        no_improve = 0
        
        for ep in range(1, n_ep + 1):
            traj = self.collect_trajectory(env)
            a_loss, c_loss = self.update(
                traj['states'], traj['actions'], traj['log_probs'],
                traj['returns'], traj['advantages']
            )
            rewards_history.append(traj['total_reward'])
            
            if traj['total_reward'] > best_reward:
                best_reward = traj['total_reward']
                no_improve = 0
            else:
                no_improve += 1
            
            if ep % 25 == 0:
                avg_reward = np.mean(rewards_history[-25:])
                print(f"  EP {ep:4d}/{n_ep} | R={avg_reward:+.4f} | A={a_loss:.4f} C={c_loss:.4f}")
            
            if ep >= 100 and no_improve >= 50:
                print(f"Converged at ep {ep}.")
                break
        
        tf.keras.backend.clear_session()
        print(f"Best reward: {best_reward:+.4f}")
        return rewards_history

    def record_trade_result(self, pnl_pct: float):
        self.recent_performance.append(pnl_pct)
        if pnl_pct > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def update_weekly_performance(self):
        if len(self.recent_performance) < 50:
            return
        recent_array = np.array(list(self.recent_performance)[-168:])
        wins = recent_array[recent_array > 0]
        losses = recent_array[recent_array <= 0]
        self.last_7_days_win_rate = len(wins) / len(recent_array) if len(recent_array) > 0 else 0.5
        mean_ret = np.mean(recent_array)
        std_ret = np.std(recent_array) + 1e-8
        self.last_7_days_sharpe = mean_ret / std_ret * np.sqrt(168)
        
        if self.last_7_days_win_rate > 0.55:
            self.weekly_confidence_boost = min(0.08, (self.last_7_days_win_rate - 0.50) * 1.5)
        elif self.last_7_days_win_rate < 0.45:
            self.weekly_confidence_boost = -min(0.08, (0.50 - self.last_7_days_win_rate) * 1.5)
        else:
            self.weekly_confidence_boost = 0.0
        
        if self.consecutive_wins >= 3:
            self.weekly_position_multiplier = min(1.3, 1.0 + (self.consecutive_wins - 2) * 0.1)
        elif self.consecutive_losses >= 2:
            self.weekly_position_multiplier = max(0.5, 1.0 - (self.consecutive_losses - 1) * 0.15)
        else:
            self.weekly_position_multiplier = 1.0
        
        print(f"Weekly Update: WR={self.last_7_days_win_rate:.3f}, Sharpe={self.last_7_days_sharpe:.3f}, Boost={self.weekly_confidence_boost:+.3f}, PosMult={self.weekly_position_multiplier:.2f}")

    def get_confidence_boost(self, lstm_confidence: float, market_volatility: float = 1.0) -> float:
        boost = self.weekly_confidence_boost
        if market_volatility > 1.5 and self.last_7_days_sharpe < 0:
            boost -= 0.03
        if market_volatility < 0.7 and self.last_7_days_win_rate > 0.6:
            boost += 0.02
        final_confidence = lstm_confidence + boost
        return float(np.clip(final_confidence, 0.3, 0.95))

    def get_position_multiplier(self, lstm_position_size: float, market_volatility: float = 1.0) -> float:
        multiplier = self.weekly_position_multiplier
        if market_volatility > 1.5:
            multiplier *= 0.7
        elif market_volatility < 0.7:
            multiplier *= 1.15
        if self.last_7_days_sharpe < -1.0:
            multiplier *= 0.5
        final_size = lstm_position_size * multiplier
        return float(np.clip(final_size, 0.05, 0.25))

    def act(self, state: np.ndarray, greedy: bool = True) -> Tuple[int, np.ndarray]:
        s_t = tf.constant(state[np.newaxis], tf.float32)
        probs = self.actor(s_t, training=False).numpy()[0]
        if greedy:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.n_actions, p=probs))
        return action, probs

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        s_t = tf.constant(state[np.newaxis], tf.float32)
        return self.actor(s_t, training=False).numpy()[0]

    def save(self, path: str = "ppo_agent"):
        try:
            self.actor.save(f"{path}_actor.keras")
            self.critic.save(f"{path}_critic.keras")
            print(f"PPO saved to {path}_actor.keras and {path}_critic.keras")
        except Exception as e:
            print(f"Save failed: {e}")

    def load(self, path: str = "ppo_agent"):
        try:
            self.actor = tf.keras.models.load_model(f"{path}_actor.keras")
            self.critic = tf.keras.models.load_model(f"{path}_critic.keras")
            self.actor_opt = Adam(learning_rate=1e-4, clipnorm=1.0)
            self.critic_opt = Adam(learning_rate=1e-3, clipnorm=1.0)
            print(f"PPO loaded from {path}_actor.keras and {path}_critic.keras")
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False