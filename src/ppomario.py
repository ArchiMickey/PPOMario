from collections import deque
from typing import Any, List, Tuple
import numpy as np

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from torch.distributions import Categorical
import torch.nn.functional as F
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
import wandb

from .models import PPO
from .env import make_mario
from .multienv import MultiActor
from .log import log_video
from .loss import clipped_value_loss, normalize, cal_actor_loss, cal_critic_loss, calc_advantage
from tqdm import tqdm
from loguru import logger

from .dataloader import AuxMemory, Memory, create_shuffled_dataloader

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
        self.save_hyperparameters()
        self.alpha = 1.0

        # self.env = gym.make(env)
        self.demo_env = make_mario(world, stage)
        self.env_actor = MultiActor(world, stage, num_workers, num_envs)
        
        self.automatic_optimization = False
        self.model = PPO(self.demo_env.observation_space.shape, self.hidden_size, self.demo_env.action_space.n)
        
        self.memories = deque([])
        
        self.ep_scores = deque([0 for _ in range(100)], maxlen=100)
        self.ep_steps = deque([0 for _ in range(100)], maxlen=100)
        self.total_episodes = 0
        
        self.curr_episode = 0
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
                action_prob, _ = self.model(state.cuda())
                dist = Categorical(action_prob)
                action = dist.sample()
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
                    curr_episodes=self.curr_episode,
                    episode_reward=episode_reward,
                    fps=24,
                    is_test=is_test,
                    use_ppg=False,
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
        
        for _ in tqdm(range(self.steps_per_epoch), leave=False, desc="Sampling Batch"):
            self.state = self.state.to(device=self.device)

            with torch.no_grad():
                action_probs, value = self.model(self.state.view(-1, *self.state.shape[2:]))
                action_probs = action_probs.reshape(self.num_workers, self.num_envs, -1)
                value = value.reshape(self.num_workers, self.num_envs)
                
                dist = Categorical(action_probs)
                action = dist.sample()
                action = action.squeeze(0)
                log_prob = dist.log_prob(action)
                                
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
        qval, adv = calc_advantage(self.gamma, self.lam, ep_rewards, ep_dones, ep_values, last_value, self.device, False)
        batch_qval += qval
        batch_adv += adv
        batch_qval = torch.stack(batch_qval)
        batch_adv = torch.stack(batch_adv)
        
        train_data = list(zip(
            batch_states.view(-1, *batch_states.shape[-3:]), batch_actions.view(-1), batch_logp.view(-1), batch_qval.view(-1), batch_adv.view(-1)
        ))
        
        for state, action, old_logp, qval, adv in train_data:
            memory = Memory(state, action, old_logp, qval, adv)
            self.memories.append(memory)
        
        self.state = torch.from_numpy(self.env_actor.reset_all())
        self.curr_scores -= self.curr_scores
        self.curr_steps -= self.curr_steps
        
        # train_data = list(zip(
        #     batch_states.view(-1, *batch_states.shape[-3:]), batch_actions.view(-1), batch_logp.view(-1), batch_qval.view(-1), batch_adv.view(-1)
        # ))
        
        # for _ in range(self.batch_epoch):
        #     random.shuffle(train_data)
        #     for data in train_data:
        #         state, action, old_logp, qval, adv = data
        #         yield state, action, old_logp, qval, adv
    
    def _update_network(self, loss: Tensor, optimizer: Optimizer):
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
        opt = self.optimizers()
        sch = self.lr_schedulers()
        states, actions, old_logps, qvals, advs = batch
        action_probs , values = self.model(states)
        actor_loss = cal_actor_loss(action_probs, actions, old_logps, advs, self.clip_ratio, self.alpha)
        entropy_loss = torch.mean(Categorical(action_probs).entropy())
        policy_loss = actor_loss - self.beta * entropy_loss
        critic_loss = cal_critic_loss(qvals, values)
        total_loss = policy_loss + critic_loss
        self.alpha = max(1 - (self.global_step / self.trainer.max_steps), 0)
        self._update_network(total_loss, opt)
        sch.step()
            
        self.log_dict(
            {
                "episode": self.curr_episode,
                "policy_loss": policy_loss,
                "critic_loss": critic_loss,
            },
            logger=False,
            prog_bar=True,
        )
        
        self.log_dict(
            {
                "train/alpha": self.alpha,
                "loss/entropy_loss": entropy_loss,
                "loss/actor_loss": actor_loss,
                "loss/policy_loss": policy_loss,
                "loss/critic_loss": critic_loss,
                "loss/total_loss": total_loss,
            },
        )
        
    def on_train_epoch_end(self) -> None:
        self.log_dict(
            {
                "train/episode": self.curr_episode,
                "train/num_games": self.total_episodes,
                "train/avg_ep_len": np.mean(self.ep_steps),
                "train/avg_ep_score": np.mean(self.ep_scores),
            },
        )
    
    def validation_step(self, *args, **kwargs):
        val_scores = []
        if self.render:
            clips = []
             
        for _ in tqdm(range(self.val_episodes), desc="Validating in episode", leave=False):
            clip, val_score = self.eval_1_episode()
            if clip is not None:
                clips.append(clip)
            val_scores.append(val_score)
            
        avg_score = sum(val_scores) / len(val_scores)
        self.log("benchmark/avg_score", avg_score)
        print('')
        logger.info(f"Episode {self.curr_episode}: Average score: {avg_score:.2f}")
        
        if self.render:
            wandb.log({"video/gameplay": clips[val_scores.index(max(val_scores))]})
            
    def test_step(self, *args, **kwargs):
        return self.eval_1_episode(is_test=True)
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, capturable=True)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda x: self.alpha])
        
        return [optimizer], [lr_scheduler]
            
    
    def aux_dataloader(self, aux_memories):        
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
        # policy_ds = ExperienceSourceDataset(self.generate_trajectory_samples)
        # policy_dl = DataLoader(dataset=policy_ds, batch_size=self.batch_size)
        self.generate_trajectory_samples()
        states = []
        actions = []
        old_logps = []
        qvals = []
        advs = []
        
        for state, action, old_logp, qval, adv in self.memories:
            states.append(state)
            actions.append(action)
            old_logps.append(old_logp)
            qvals.append(qval)
            advs.append(adv)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_logps = torch.stack(old_logps)
        qvals = torch.stack(qvals)
        advs = torch.stack(advs)
        policy_dl = create_shuffled_dataloader([states, actions, old_logps, qvals, advs], batch_size=self.batch_size)
        self.memories.clear()
        self.curr_episode += 1
        
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