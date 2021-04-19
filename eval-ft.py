import click
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import random

import torch
import torch.nn as nn

from configs.default import default_config
from launch_experiment import deep_update_dict
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:")
print(device)


class PEARLFineTuningHelper:

    def __init__(
            self,
            env,
            agent,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            total_steps=int(1e6),
            max_path_length=200,
            num_exp_traj_eval=1,
            start_fine_tuning=10,
            fine_tuning_steps=1,
            should_freeze_z=True,
            replay_buffer_size=int(1e6),
            batch_size=256,
            discount=0.99,
            policy_lr=1e-4,
            qf_lr=1e-4,
            temp_lr=1e-4,
            target_entropy=None,
            optimizer_class=torch.optim.Adam,
            soft_target_tau=1e-2
    ):
        self.env = env
        self.agent = agent

        # Ctitic networks
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda')
        self.log_alpha.to(device)
        self.target_entropy = target_entropy

        # Experimental setting
        self.total_steps = total_steps
        self.max_path_length = max_path_length
        self.num_exp_traj_eval = num_exp_traj_eval
        self.start_fine_tuning = start_fine_tuning
        self.fine_tuning_steps = fine_tuning_steps
        self.should_freeze_z = should_freeze_z

        # Hyperparams
        self.batch_size = batch_size
        self.discount = discount
        self.soft_target_tau = soft_target_tau

        self.replay_buffer = SimpleReplayBuffer(
            max_replay_buffer_size=replay_buffer_size,
            observation_dim=int(np.prod(env.observation_space.shape)),
            action_dim=int(np.prod(env.action_space.shape)),
        )

        self.q_losses = []
        self.temp_losses = []
        self.policy_losses = []
        self.temp_vals = []

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.temp_optimizer = optimizer_class(
            [self.log_alpha],
            lr=temp_lr,
        )

        self.print_experiment_description()

    def get_mean(self, losses):
        if not losses:
            return None
        tot = 0
        for tensor in losses:
            tot += np.mean(tensor.to('cpu').detach().numpy())
        return tot / len(losses)

    def collect_samples(self, should_accum_context):
        path = self.rollout(should_accum_context)
        self.replay_buffer.add_path(path)
        steps = path['rewards'].shape[0]
        ret = sum(path['rewards'])[0]
        return ret, steps

    def rollout(self, should_accum_context):
        should_fine_tune = not should_accum_context
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        o = self.env.reset()
        next_o = None
        path_length = 0
        done = False

        while (not done):
            a, agent_info = self.agent.get_action(o)
            next_o, r, d, env_info = self.env.step(a)
            real_done = False if path_length == self.max_path_length else d
            observations.append(o)
            rewards.append(r)
            terminals.append(real_done)
            actions.append(a)
            agent_infos.append(agent_info)
            path_length += 1
            o = next_o
            env_infos.append(env_info)
            if should_accum_context:
                self.agent.update_context([o, a, r, next_o, d, env_info])
            if should_fine_tune:
                for j in range(self.fine_tuning_steps):
                    self.fine_tuning_step()
            if d or path_length >= self.max_path_length:
                done = True

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )

        if should_accum_context:
            self.agent.sample_z()

        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
        )

    def get_samples(self):
        batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(self.batch_size))
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return o, a, r, no, t

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_networks(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def fine_tuning_step(self):
        obs, actions, rewards, next_obs, terms = self.get_samples()

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs_flat = obs.view(t * b, -1)
        actions_flat = actions.view(t * b, -1)
        next_obs_flat = next_obs.view(t * b, -1)
        rewards_flat = rewards.view(self.batch_size, -1)
        terms_flat = terms.view(self.batch_size, -1)

        """
        QF Loss
        """
        with torch.no_grad():
            next_policy_outputs, task_z = self.agent(next_obs, self.agent.context)
            next_new_actions, _, _, next_log_prob = next_policy_outputs[:4]
            t_q1_pred = self.target_qf1(next_obs_flat, next_new_actions, task_z.detach())  # TODO: Remove .detach() if redundant
            t_q2_pred = self.target_qf2(next_obs_flat, next_new_actions, task_z.detach())
            t_q_min = torch.min(t_q1_pred, t_q2_pred)
            q_target = rewards_flat + (1. - terms_flat) * self.discount * (t_q_min - self.alpha * next_log_prob)
        q1_pred = self.qf1(obs_flat, actions_flat, task_z.detach())                    # TODO: Remove .detach() if redundant
        q2_pred = self.qf2(obs_flat, actions_flat, task_z.detach())
        qf_loss = torch.mean((q1_pred - q_target.detach()) ** 2) + torch.mean((q2_pred - q_target.detach()) ** 2)

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        """
        Policy and Temp Loss
        """
        for p in self.qf1.parameters():
            p.requires_grad = False
        for p in self.qf2.parameters():
            p.requires_grad = False

        policy_outputs, task_z = self.agent(obs, self.agent.context)
        new_actions, policy_mean, policy_log_std, log_prob = policy_outputs[:4]
        min_q_new_actions = self._min_q(obs_flat, new_actions, task_z)

        policy_loss = (self.alpha * log_prob - min_q_new_actions).mean()
        temp_loss = -self.alpha * (log_prob.detach() + self.target_entropy).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.temp_optimizer.zero_grad()
        temp_loss.backward()
        self.temp_optimizer.step()

        for p in self.qf1.parameters():
            p.requires_grad = True
        for p in self.qf2.parameters():
            p.requires_grad = True

        """
        Update Target Networks
        """
        self._update_target_networks()

        self.q_losses.append(qf_loss.detach())
        self.temp_losses.append(temp_loss.detach())
        self.policy_losses.append(policy_loss.detach())
        self.temp_vals.append(self.alpha.detach())

    def evaluate_agent(self, n_starts=10):
        reward_sum = 0
        for _ in range(n_starts):
            path = rollout(self.env, self.agent, max_path_length=self.max_path_length, accum_context=False)
            reward_sum += sum(path['rewards'])[0]
        return reward_sum / n_starts

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def fine_tune(self, variant, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        cumulative_timestep = 0
        i_episode = 0

        df = pd.DataFrame(
            columns=[
                'step',
                'real_step',
                'train_reward',
                'eval_reward',
                'loss/q-f1',
                'loss/alpha',
                'loss/policy',
                'val/alpha'
            ]
        )

        # For this experiment, we are evaluating in just one sampled task from the meta-test set
        tasks = self.env.get_all_task_idx()
        eval_tasks = list(tasks[-variant['n_eval_tasks']:])
        idx = eval_tasks[0]
        self.env.reset_task(idx)

        self.agent.clear_z()
        while cumulative_timestep < self.total_steps:
            i_episode += 1
            should_infer_posterior = self.num_exp_traj_eval <= i_episode < self.start_fine_tuning
            should_fine_tune = self.start_fine_tuning <= i_episode
            should_accum_context = not should_fine_tune
            if should_fine_tune and self.should_freeze_z and (not self.agent.freeze_z):
                self.agent.freeze_z = True
            train_reward, episode_steps = self.collect_samples(should_accum_context)
            cumulative_timestep += episode_steps
            if should_infer_posterior:
                self.agent.infer_posterior(self.agent.context)
            eval_reward = self.evaluate_agent()
            message = 'Episode {} \t\t Samples {} \t\t Real samples {} \t\t Train reward: {} \t\t Eval reward: {}'
            print(message.format(i_episode, i_episode * self.max_path_length, cumulative_timestep, train_reward,
                                 eval_reward))
            new_df_row = {
                'step': int(i_episode * self.max_path_length),
                'real_step': int(cumulative_timestep),
                'train_reward': train_reward,
                'eval_reward': eval_reward,
                'loss/q-f1': self.get_mean(self.q_losses),
                'loss/alpha': self.get_mean(self.temp_losses),
                'loss/policy': self.get_mean(self.policy_losses),
                'val/alpha': self.get_mean(self.temp_vals)
            }
            self.q_losses = []
            self.temp_losses = []
            self.policy_losses = []
            self.temp_vals = []
            df = df.append(new_df_row, ignore_index=True)
            results_path = "results_ft/{}/ft{}".format(variant['env_name'], "_{}".format(seed - 1))
            if not os.path.isdir(results_path):
                os.makedirs(results_path)
            df.to_csv("{}/progress.csv".format(results_path))

    def print_experiment_description(self):
        print("\n\n", " -" * 15, "\n")
        print("Total steps:  \t\t\t", self.total_steps)
        print("Max path length:  \t\t\t", self.max_path_length)
        print("Trajectory length with prior:  \t\t\t", self.num_exp_traj_eval)
        print("Start fine tuning after:  \t\t\t", self.start_fine_tuning)
        print("Number of fine-tuning steps:  \t\t\t", self.fine_tuning_steps)
        print("Should freeze Z during fine-tuning?  \t\t\t", self.should_freeze_z)
        print("Batch size:  \t\t\t", self.batch_size)
        print("Gamma:  \t\t\t", self.discount)
        print("Tau:  \t\t\t", self.soft_target_tau)


@click.command()
@click.argument('env_name', default=None)
@click.option('--seed', default=1)
@click.option('--deterministic', is_flag=True, default=False)
@click.option('--traj_prior', default=1)
@click.option('--start_ft_after', default=500)
@click.option('--ft_steps', default=1)
@click.option('--avoid_freezing_z', is_flag=True, default=False)
@click.option('--lr', default=1e-4)
@click.option('--batch_size', default=256)
@click.option('--avoid_loading_critics', is_flag=True, default=False)
def main(
        env_name,
        seed,
        deterministic,
        traj_prior,
        start_ft_after,
        ft_steps,
        avoid_freezing_z,
        lr,
        batch_size,
        avoid_loading_critics
):
    config = "configs/{}.json".format(env_name)
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    exp_name = variant['env_name']
    print("Experiment: {}".format(exp_name))

    env = NormalizedBoxEnv(ENVS[exp_name](**variant['env_params']))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    print("Observation space:")
    print(env.observation_space)
    print(obs_dim)
    print("Action space:")
    print(env.action_space)
    print(action_dim)
    print("-" * 10)

    # instantiate networks
    latent_dim = variant['latent_size']
    reward_dim = 1
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    target_qf1 = qf1.copy()
    target_qf2 = qf2.copy()
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )

    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    path_to_exp = "output/{}/pearl_{}".format(env_name, seed-1)
    print("Based on experiment: {}".format(path_to_exp))
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))
    if not avoid_loading_critics:
        qf1.load_state_dict(torch.load(os.path.join(path_to_exp, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path_to_exp, 'qf2.pth')))
        target_qf1.load_state_dict(torch.load(os.path.join(path_to_exp, 'target_qf1.pth')))
        target_qf2.load_state_dict(torch.load(os.path.join(path_to_exp, 'target_qf2.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        agent.to(device)
        policy.to(device)
        context_encoder.to(device)
        qf1.to(device)
        qf2.to(device)
        target_qf1.to(device)
        target_qf2.to(device)

    helper = PEARLFineTuningHelper(
        env=env,
        agent=agent,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,

        num_exp_traj_eval=traj_prior,
        start_fine_tuning=start_ft_after,
        fine_tuning_steps=ft_steps,
        should_freeze_z=(not avoid_freezing_z),

        replay_buffer_size=int(1e6),
        batch_size=batch_size,
        discount=0.99,
        policy_lr=lr,
        qf_lr=lr,
        temp_lr=lr,
        target_entropy=-action_dim,
    )

    helper.fine_tune(variant=variant, seed=seed)


if __name__ == '__main__':
    main()
