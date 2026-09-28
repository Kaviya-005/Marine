import os
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

class StreamlitSACCallback(BaseCallback):
    """
    Custom Stable-Baselines3 Callback to update Streamlit UI during SAC training.
    """
    def __init__(self, total_timesteps, progress_callback=None, verbose=0):
        super(StreamlitSACCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_callback = progress_callback
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_outcomes = []
        self.episodes = []
        self.current_ep_reward = 0.0
        self.current_ep_length = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Accumulate reward and length
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if rewards is not None:
            self.current_ep_reward += float(rewards[0])
            self.current_ep_length += 1

        if dones is not None and dones[0]:
            self.episode_count += 1
            status = 'UNKNOWN'
            if infos is not None and len(infos) > 0:
                status = infos[0].get('status', 'IN_PROGRESS')

            self.episodes.append(self.episode_count)
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            self.episode_outcomes.append(status)

            # Extract actor/critic loss if available in logger
            critic_loss = 0.0

            stats = {
                'episodes': self.episodes,
                'rewards': self.episode_rewards,
                'steps': self.episode_lengths,
                'outcomes': self.episode_outcomes,
                'losses': [0.0] * len(self.episodes)
            }

            if self.progress_callback is not None:
                self.progress_callback(
                    self.num_timesteps,
                    self.total_timesteps,
                    self.current_ep_reward,
                    critic_loss,
                    status,
                    stats
                )

            # Reset episode tracking
            self.current_ep_reward = 0.0
            self.current_ep_length = 0

        return True


def create_sac_model(env, learning_rate=3e-4, buffer_size=50000, batch_size=128, gamma=0.99, tau=0.005):
    """
    Creates a Soft Actor-Critic (SAC) model using Stable-Baselines3.
    """
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=200,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        ent_coef="auto",
        verbose=0
    )
    return model


def train_sac_agent(env, model=None, total_timesteps=10000, progress_callback=None):
    """
    Trains the SAC model in the environment with live callback reporting.
    """
    if model is None:
        model = create_sac_model(env)

    cb = StreamlitSACCallback(total_timesteps=total_timesteps, progress_callback=progress_callback)
    model.learn(total_timesteps=total_timesteps, callback=cb)

    training_stats = {
        'episodes': cb.episodes,
        'rewards': cb.episode_rewards,
        'steps': cb.episode_lengths,
        'outcomes': cb.episode_outcomes,
        'losses': [0.0] * len(cb.episodes)
    }

    return model, training_stats


def save_sac_model(model, filepath):
    """
    Saves the trained SAC model using Stable-Baselines3.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    model.save(filepath)


def load_sac_model(filepath, env=None):
    """
    Loads a trained SAC model using Stable-Baselines3.
    """
    # Normalize zip extension if necessary
    clean_path = filepath
    if filepath.endswith('.zip'):
        clean_path = filepath[:-4]
    
    if not os.path.exists(filepath) and not os.path.exists(filepath + '.zip'):
        raise FileNotFoundError(f"SAC Model file not found at {filepath}")

    model = SAC.load(clean_path, env=env)
    return model
