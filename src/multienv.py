from collections import deque
from typing import List
from loguru import logger
import numpy as np
import ray
from tqdm import tqdm


try:
    from .env import make_mario
except ImportError:
    from env import make_mario

from icecream import ic

@ray.remote
class MultiEnvActor:
    def __init__(self, world:int, stage:int, num_envs: int = 1) -> None:
        self.num_envs = num_envs
        self.envs = [make_mario(world, stage) for _ in range(num_envs)]
    
    def reset_all_envs(self):
        states = []
        for env_idx in range(self.num_envs):
            state = self.envs[env_idx].reset()
            states.append(np.array(state))
        return states
    
    def step(self, actions):
        try:
            assert len(actions) == self.num_envs
        except AssertionError:
            logger.error(f"actions length {len(actions)} != {self.num_envs}")
            raise AssertionError
        
        next_states = []
        rewards = []
        dones = []
        infos = []
        for i in range(self.num_envs):
            next_state, reward, done, info = self.envs[i].step(actions[i])
            next_state = np.array(next_state)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
            if done:
                self.envs[i].reset()
                
        return (np.array(next_states), np.array(rewards), np.array(dones), infos)
class MultiActor:
    def __init__(self, world:int = 1, stage: int = 1, num_workers: int = 1, num_envs: int = 1) -> None:
        self.num_workers = num_workers
        self.num_envs = num_envs
        self.actors = [MultiEnvActor.remote(world, stage, num_envs) for _ in range(self.num_workers)]
    
    def reset_all(self):
        returns = ray.get([self.actors[i].reset_all_envs.remote() for i in range(self.num_workers)])
        return np.array(returns)
    
    def step(self, action: List[int] or int):
        if isinstance(action, int):
                action = [action]
        
        returns = ray.get([self.actors[i].step.remote(action[i]) for i in range(self.num_workers)])
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        for actor_return in returns:
            # ic(actor_return)
            next_state, reward, done, info = actor_return
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
                    
        return (np.array(next_states), np.array(rewards), np.array(dones), infos)

def main(num_workers: int = 2, num_envs: int = 3) -> None:
    actor = MultiActor(num_workers=num_workers, num_envs=num_envs)
    ic(actor.num_envs, actor.num_workers)
    curr_scores = np.zeros((actor.num_workers, actor.num_envs))
    ep_scores = deque([0 for _ in range(100)], maxlen=100)
    ic(actor.reset_all())
    # for _ in tqdm(range(500)):
    #     test_action = np.random.randint(0, 7, size=(num_workers, num_envs))
    #     next_states, rewards, dones = actor.step(test_action)
    #     # ic(next_states.shape, rewards, dones)
    #     done_idx = np.where(dones)
    #     curr_scores += rewards
    #     for score in curr_scores[done_idx]:
    #         ep_scores.append(score)
    #     curr_scores[done_idx] = 0
                    
if __name__ == "__main__":
    main(num_workers=2, num_envs=2)