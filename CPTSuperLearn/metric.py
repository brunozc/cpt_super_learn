import os
from collections import defaultdict
import matplotlib.pylab as plt
import numpy as np

from CPTSuperLearn.agent import DQLAgent
from CPTSuperLearn.environment import CPTEnvironment
from CPTSuperLearn.utils import moving_average

class MetricsTracker:
    def __init__(self, output_folder):
        """
        Advanced metrics tracking and visualization

        Parameters:
        -----------
        :param output_folder: Output folder for storing metrics
        """
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # Metric storage
        self.metrics = {
            'episode': [],
            'episode_rewards': [],
            'epsilon_values': [],
            'learning_rates': [],
            'action_distribution': defaultdict(int),
            'rmse_progression': [],
            'cpt_count_progression': [],
            'gradient_norms': []
        }

    def update(self,
               episode: int,
               reward: float,
               agent: DQLAgent,
               environment: CPTEnvironment,
               action: int):
        """
        Metrics update

        Parameters:
        -----------
        :param episode: Current training episode
        :param reward: Episode reward
        :param agent: Current DQL agent
        :param environment: Current environment
        :param action: Selected action
        """

        self.metrics['episode'].append(episode)
        self.metrics['episode_rewards'].append(reward)
        self.metrics['epsilon_values'].append(agent.epsilon)
        # Action distribution
        self.metrics['action_distribution'][action] += 1

        # Interpolation metrics
        if environment.predicted_data is not None:
            rmse = np.sqrt(np.mean((environment.true_data - environment.predicted_data) ** 2))
            self.metrics['rmse_progression'].append(rmse)
            self.metrics['cpt_count_progression'].append(len(environment.sampled_positions))

        # Gradient tracking
        total_grad_norm = 0
        for param in agent.qnetwork_local.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item()
        self.metrics['gradient_norms'].append(total_grad_norm)

    def visualize_metrics(self):
        """
        Generate figures of the metrics
        """
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        # Reward progression
        ax[0, 0].plot(self.metrics['episode'], self.metrics['episode_rewards'], label='Episode Rewards')
        ax[0, 0].plot(self.metrics['episode'], moving_average(np.array(self.metrics['episode_rewards']), 50),
                      label='Moving Average')
        ax[0, 0].set_title('Reward progression')
        ax[0, 0].set_xlabel('Episodes')
        ax[0, 0].set_ylabel('Rewards')
        ax[0, 0].set_xlim(0, len(self.metrics['episode_rewards']))
        ax[0, 0].grid()
        ax[0, 0].legend()

        # RMSE Progression
        ax[0, 1].plot(self.metrics['episode'], self.metrics['rmse_progression'], label='Episode RMSE')
        ax[0, 1].plot(self.metrics['episode'], moving_average(np.array(self.metrics['rmse_progression']), 50),
                      label='Moving Average')
        ax[0, 1].set_title('RMSE progression')
        ax[0, 1].set_xlabel('Episodes')
        ax[0, 1].set_ylabel('RMSE')
        ax[0, 1].set_xlim(0, len(self.metrics['rmse_progression']))
        ax[0, 1].set_ylim(bottom=0)
        ax[0, 1].grid()
        ax[0, 1].legend()

        # Epsilon decay
        ax[1, 0].plot(self.metrics['episode'], self.metrics['epsilon_values'])
        ax[1, 0].set_title('Exploration rate (Epsilon)')
        ax[1, 0].set_xlabel('Episodes')
        ax[1, 0].set_ylabel('Epsilon')
        ax[1, 0].set_xlim(0, len(self.metrics['epsilon_values']))
        ax[1, 0].grid()

        # Control Points Count
        ax[1, 1].plot(self.metrics['episode'], self.metrics['cpt_count_progression'])
        ax[1, 1].set_title('In-situ tests')
        ax[1, 1].set_xlabel('Episodes')
        ax[1, 1].set_ylabel('Number in-situ tests')
        ax[1, 1].set_xlim(0, len(self.metrics['cpt_count_progression']))
        ax[1, 1].set_ylim(bottom=0)
        ax[1, 1].grid()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'training_metrics.png'))
        plt.savefig(os.path.join(self.output_folder, 'training_metrics.pdf'))
        plt.close()