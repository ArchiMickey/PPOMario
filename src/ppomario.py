from collections import deque
from email import policy
import random
from typing import Any, Dict, List, Tuple
import numpy as np

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor, device
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules import ExperienceSourceDataset
from torch.distributions import Categorical
import torch.nn.functional as F
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
import wandb

from .models import PPO, PPG
from .env import make_mario
from .multienv import MultiActor
from .log import log_video
from .loss import clipped_value_loss, normalize, cal_actor_loss, cal_critic_loss, calc_advantage
from tqdm import tqdm
from loguru import logger

from .dataloader import AuxMemory, create_shuffled_dataloader

if _GYM_AVAILABLE:
    import gym
else:  # pragma: no cover
    warn_missing_pkg("gym")

class PPOMario(LightningModule):
    """PyTorch Lightning implementation of `Proximal Policy Optimization.
    <https://arxiv.org/abs/1707.06347>`_
    Paper authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
    """

    def __init__(
        self,
        # env: str,
        world: int = 1,
        stage: int = 1,
        gamma: float = 0.9,
        lam: float = 0.99,
        beta: float = 0.01,
        lr: float = 1e-4,
        lr_decay_ratio: float = 0.1,
        lr_decay_epoch: int = 350,
        lr_decay_step: int = 100000,
        max_episode_len: float = 200,
        hidden_size: int = 128,
        batch_epoch: int = 10,
        batch_size: int = 32,
        num_workers: int = 1,
        num_envs: int = 1,
        steps_per_epoch: int = 32,
        val_episodes: int = 5,
        nb_optim_iters: int = 1,
        clip_ratio: float = 0.2,
        value_clip: float = 0.4,
        render: bool = True,
        use_ppg: bool = False,
        aux_batch_size: int = 16,
        aux_batch_epoch: int = 9,
        aux_interval: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lam: advantage discount factor (lambda in the paper)
            lr_actor: learning rate of actor network
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
            nb_optim_iters: how many steps of gradient descent to perform on each batch
            clip_ratio: hyperparameter for clipping in the policy objective
        """
        super().__init__()

        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("This Module requires gym environment which is not installed yet.")

        # Hyperparameters
        self.world = world
        self.stage = stage
        self.lr = lr
        self.lr_decay_ratio = lr_decay_ratio
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_step = lr_decay_step
        self.steps_per_epoch = steps_per_epoch
        self.val_episodes = val_episodes
        self.nb_optim_iters = nb_optim_iters
        self.hidden_size = hidden_size
        self.batch_epoch = batch_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_envs = num_envs
        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.render = render
        if use_ppg:
            self.value_clip = value_clip
            self.aux_batch_size = aux_batch_size
            self.aux_batch_epoch = aux_batch_epoch
            self.aux_interval = aux_interval
        self.save_hyperparameters()
        self.alpha = 1.0
        self.use_ppg = use_ppg

        # self.env = gym.make(env)
        self.demo_env = make_mario(world, stage)
        self.env_actor = MultiActor(world, stage, num_workers, num_envs)
        
        if self.use_ppg:
            self.should_aux = False
            self.aux_memories = deque([])
            self.aux_dl = None
            self.automatic_optimization = False
            self.model = PPG(self.demo_env.observation_space.shape, self.hidden_size, self.demo_env.action_space.n)
        
        else:
            self.model = PPO(self.demo_env.observation_space.shape, self.hidden_size, self.demo_env.action_space.n)
        
        self.total_episodes: int = 0        
        self.ep_scores = deque([0 for _ in range(100)], maxlen=100)
        self.ep_steps = deque([0 for _ in range(100)], maxlen=100)
            
        self.curr_scores = np.zeros((self.num_workers, self.num_envs))
        self.curr_steps = np.zeros((self.num_workers, self.num_envs))

        self.state = torch.from_numpy(self.env_actor.reset_all())

    def eval_1_episode(self, is_test: bool = False):
        pbar = tqdm(desc="Evaluating", leave=False)
        state = torch.FloatTensor(self.demo_env.reset()).unsqueeze(0)
        done = False
        episode_reward: float = 0
        frames = []
        durations = []
        
        while not done:
            with torch.no_grad():
                if self.use_ppg:
                    action_prob, _ = self.model.actor(state.cuda())
                    dist = Categorical(action_prob)
                    action = dist.sample()
                    action = action.squeeze(0).cpu().numpy().item()
                else:
                    logits, _ = self.model(state.cuda())
                    pi = Categorical(logits=logits)
                    action = pi.sample()
                    action = action.squeeze(0).cpu().numpy().item()
                pbar.update(1)
            if self.render:
                frame = self.demo_env.render(mode="rgb_array")
                frame = np.array(frame)
                frames.append(frame)
                durations.append(1/24)
            next_state, reward, done, _ = self.demo_env.step(action)
            episode_reward += reward
            state = torch.FloatTensor(next_state).unsqueeze(0)
            
        if self.render:
            clip = log_video(
                    env_name='{}-{}'.format(self.world, self.stage),
                    frames=frames,
                    durations=durations,
                    curr_episodes=self.current_epoch + 1,
                    episode_reward=episode_reward,
                    fps=24,
                    is_test=is_test,
                    )
        else:
            clip = None
            
        pbar.close()
            
        return clip, episode_reward
    
    def generate_trajectory_samples(self) -> Tuple[List[Tensor], List [Tensor], List[Tensor]]:
        """Contains the logic for generating trajectory data to train policy and value network.
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """
        states = []
        actions = []
        logp = []
        dones = []
        rewards = []
        values = []
        batch_qval = []
        batch_adv = []
        
        self.ep_best_reward = 0
        
        for _ in tqdm(range(self.steps_per_epoch), leave=False, desc="Sampling Batch",
                      bar_format="{desc}: {percentage:3.0f}%|{bar:10}{r_bar}"):
            self.state = self.state.to(device=self.device)

            with torch.no_grad():
                if self.use_ppg:
                    action_probs, _ = self.model.actor(self.state.view(-1, *self.state.shape[2:]))
                    value = self.model.critic(self.state.view(-1, *self.state.shape[2:]))
                    
                    action_probs = action_probs.reshape(self.num_workers, self.num_envs, -1)
                    value = value.reshape(self.num_workers, self.num_envs)
                    
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    action = action.squeeze(0)
                    log_prob = dist.log_prob(action)
                else:
                    logits, value = self.model(self.state.view(-1, *self.state.shape[2:]))
                    logits = logits.reshape(self.num_workers, self.num_envs, -1)
                    value = value.reshape(self.num_workers, self.num_envs)
                    # ic(logits.shape, value.shape)
                    pi = Categorical(logits=logits)
                    action = pi.sample()
                    # ic(action.shape)
                    log_prob = pi.log_prob(action)
                    # ic(log_prob.shape)
                                
            try:
                next_state, reward, done, _ = self.env_actor.step(action.cpu().numpy())
            except:
                action = action.unsqueeze(0)
                next_state, reward, done, _ = self.env_actor.step(action.cpu().numpy())
            next_state = torch.FloatTensor(next_state)
            
            # ic(next_state.shape, reward, done)
            self.curr_scores += reward
            self.curr_steps += 1
            done_idx = np.where(done)
            for score in self.curr_scores[done_idx]:
                self.ep_scores.append(score)
            for end_step in self.curr_steps[done_idx]:
                self.ep_steps.append(end_step)
            self.curr_scores[done_idx] = 0
            self.curr_steps[done_idx] = 0
            self.total_episodes += done.sum()
            
            # ic(self.state.shape, action.shape, log_prob.shape, done.shape, reward.shape, value.shape)
            states.append(self.state)
            actions.append(action)
            logp.append(log_prob)
            dones.append(done)
            rewards.append(reward)
            values.append(value)

            self.state = next_state

        self.state = self.state.to(device=self.device)
        with torch.no_grad():
            if self.use_ppg:
                last_value = self.model.critic(self.state.view(-1 ,*self.state.shape[2:]))
            else:
                _, last_value = self.model(self.state.view(-1, *self.state.shape[2:]))
            last_value = last_value.reshape(self.num_workers, self.num_envs)
        
        batch_states = torch.stack(states)
        batch_actions = torch.stack(actions)
        batch_logp = torch.stack(logp)
        ep_values = torch.stack(values)
        ep_rewards = torch.FloatTensor(rewards)
        ep_dones = torch.FloatTensor(dones)
        
        # ic(batch_states.shape, batch_actions.shape, batch_logp.shape, ep_values.shape, ep_rewards.shape, ep_dones.shape, last_value.shape)
        # advantage
        qval, adv = calc_advantage(self.gamma, self.lam, ep_rewards, ep_dones, ep_values, last_value, self.device, self.use_ppg)
        batch_qval += qval
        batch_adv += adv
        batch_qval = torch.stack(batch_qval)
        batch_adv = torch.stack(batch_adv)
        
                
        if self.use_ppg:
            for state, reward, value in list(zip(batch_states, batch_qval, ep_values)):
                aux_memory = AuxMemory(state, reward, value)
                self.aux_memories.append(aux_memory)
        
        self.state = torch.from_numpy(self.env_actor.reset_all())
        self.curr_scores -= self.curr_scores
        self.curr_steps -= self.curr_steps
        
        train_data = list(zip(
            batch_states.view(-1, *batch_states.shape[-3:]), batch_actions.view(-1), batch_logp.view(-1), batch_qval.view(-1), batch_adv.view(-1)
        ))
        
        for _ in range(self.batch_epoch):
            random.shuffle(train_data)
            for data in train_data:
                state, action, old_logp, qval, adv = data
                yield state, action, old_logp, qval, adv
    
    def _update_network(self, loss: Tensor, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        """Carries out a single update to actor and critic network from a batch of replay buffer.
        Args:
            batch: batch of replay buffer/trajectory data
            batch_idx: not used
            optimizer_idx: idx that controls optimizing actor or critic network
        Returns:
            loss
        """
        # ic(states.shape, actions.shape, old_logps.shape, qvals.shape, advs.shape)
        
        if self.use_ppg:
            actor_opt, critic_opt = self.optimizers()
            if self.should_aux:
                for epoch in range(self.aux_batch_epoch):
                    for states, old_action_probs, rewards, old_values in tqdm(self.aux_dl, desc=f"auxiliary epoch {epoch}", leave=False):
                        action_probs, policy_values = self.model.actor(states)
                        action_logprobs = action_probs.log()
                        aux_loss = clipped_value_loss(policy_values.squeeze(-1), rewards, old_values, self.value_clip)
                        kl_loss = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                        policy_loss = aux_loss + kl_loss
                        self._update_network(policy_loss, actor_opt)
                        
                        values = self.model.critic(states)
                        values = values.squeeze(-1)
                        value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)
                        self._update_network(value_loss, critic_opt)
                        self.log_dict(
                            {
                                "loss/aux_loss": aux_loss,
                                "loss/kl_loss": kl_loss,
                                "loss/policy_loss": policy_loss,
                                "loss/value_loss": value_loss,
                            },
                            on_step=True,
                        )
                self.aux_dl = None
                self.should_aux = False
            
            states, actions, old_logps, qvals, advs = batch
            action_probs, _ = self.model.actor(states)
            values = self.model.critic(states)
            dist = Categorical(action_probs)
            action_log_probs = dist.log_prob(actions)
            entropy_loss = dist.entropy().mean()
            
            ratios = (action_log_probs - old_logps).exp()
            advs = normalize(advs)
            surr1 = ratios * advs
            # surr2 = ratios.clamp(1 - self.clip_ratio * self.alpha, 1 + self.clip_ratio * self.alpha) * advs
            surr2 = ratios.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advs
            actor_loss = -torch.min(surr1, surr2).mean()
            policy_loss = actor_loss - self.beta * entropy_loss
            
            self._update_network(policy_loss, actor_opt)
            new_values = self.model.critic(states)
            value_loss = clipped_value_loss(new_values, qvals, values, self.value_clip)
            
            self._update_network(value_loss, critic_opt)
            total_loss = policy_loss + value_loss
                        
            self.log_dict(
                {
                    "loss/entropy_loss": entropy_loss,
                    "loss/actor_loss": actor_loss,
                    "loss/policy_loss": policy_loss,
                    "loss/value_loss": value_loss,
                    "loss/total_loss": total_loss,
                },
                on_step=True,
            )
            self.log_dict(
                {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                },
                logger=False,
                prog_bar=True,
            )
            
        else:
            states, actions, old_logps, qvals, advs = batch
            logits, value = self.model(states.view(-1, *states.shape[-3:]))
            actor_loss = cal_actor_loss(logits, actions, old_logps, advs, self.clip_ratio, self.alpha)
            critic_loss = cal_critic_loss(qvals, value)
            entropy_loss = torch.mean(Categorical(logits=logits).entropy())
            total_loss = actor_loss + critic_loss - self.beta * entropy_loss
            
            # ic(entropy_loss, actor_loss, critic_loss, total_loss)
            
            self.log("loss/actor_loss", actor_loss, on_step=True, logger=True)
            self.log("loss/critic_loss", critic_loss, on_step=True, logger=True)
            self.log("loss/entropy_loss", entropy_loss, on_step=True, logger=True)
            self.log("loss/total_loss", total_loss, on_step=True, logger=True)
        
        self.log("train/num_games", self.total_episodes, prog_bar=True)
        self.log("performance/avg_ep_len", np.mean(self.ep_steps))
        self.log("performance/avg_ep_score", np.mean(self.ep_scores))
        
        return total_loss
    
    def on_train_epoch_end(self) -> None:
        # self.alpha = max(1 - (self.global_step / self.lr_decay_step), 0)
        # self.log("train/alpha", self.alpha)
        if (self.current_epoch + 1) % self.aux_interval == 0:
            self.aux_dl = self.aux_dataloader(self.aux_memories)
            self.aux_memories.clear()
            self.should_aux = True
    
    def validation_step(self, *args, **kwargs):
        val_scores = []
        if self.render:
            clips = []
             
        for _ in tqdm(range(self.val_episodes), desc="Validating in episode", leave=False,
                      bar_format="{desc}: {percentage:3.0f}%|{bar:10}{r_bar}"):
            clip, val_score = self.eval_1_episode()
            if clip is not None:
                clips.append(clip)
            val_scores.append(val_score)
            
        avg_score = sum(val_scores) / len(val_scores)
        self.log("avg_score", avg_score, logger=True)
        
        print('')
        log_str = f"Episode {self.current_epoch + 1}: Average score: {avg_score:.2f} | num_games: {self.total_episodes}"
        print('|' + '=' * len(log_str) + '|')
        print('|' + log_str + '|')
        print('|' + '=' * len(log_str) + '|')
        
        if self.render:
            wandb.log({"video/gameplay": clips[val_scores.index(max(val_scores))]})
            
    def test_step(self, *args, **kwargs):
        return self.eval_1_episode(is_test=True)
    
    # def configure_gradient_clipping(
    #     self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    # ):
    #     grad_norm = torch.nn.utils.clip_grad_norm_(
    #         sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
    #     )
    #     self.log("grad_norm", grad_norm)
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        if not self.use_ppg:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, capturable=True)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda x: self.alpha])
            
            return [optimizer], [lr_scheduler]

        else:
            opt_actor = torch.optim.Adam(self.model.actor.parameters(), lr=self.lr, capturable=True)
            opt_critic = torch.optim.Adam(self.model.critic.parameters(), lr=self.lr, capturable=True)
            
            actor_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt_actor, lr_lambda=[lambda x: self.alpha])
            critic_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt_critic, lr_lambda=[lambda x: self.alpha])
            
            return [opt_actor, opt_critic], [actor_lr_scheduler, critic_lr_scheduler]
    
    def aux_dataloader(self, aux_memories):
        # actor_opt, critic_opt = self.optimizers()
        
        states = []
        rewards = []
        old_values = [] 
        
        for state, reward, old_value in aux_memories:
            states.append(state)
            rewards.append(reward)
            old_values.append(old_value)
            
        states = torch.stack(states)
        rewards = torch.stack(rewards)
        old_values = torch.stack(old_values)
        
        old_action_probs, _ = self.model.actor(states.view(-1, *states.shape[-3:]))
        old_action_probs.detach_()
        dl = create_shuffled_dataloader([states.view(-1, *states.shape[-3:]), old_action_probs, rewards.view(-1), old_values.view(-1)], batch_size=self.aux_batch_size)
        return dl
    
    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        policy_ds = ExperienceSourceDataset(self.generate_trajectory_samples)
        policy_dl = DataLoader(dataset=policy_ds, batch_size=self.batch_size)
                
        return policy_dl

    def _dummy_dataloader(self) -> DataLoader:
        """Dummy dataloader for validation/ testing
        """
        return DataLoader([0])

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()
    
    def val_dataloader(self) -> DataLoader:
        return self._dummy_dataloader()
    
    def test_dataloader(self) -> DataLoader:
        return self._dummy_dataloader()


def cli_main() -> None:
    model = PPO(world=1, stage=1, num_workers=6, )

    trainer = Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()