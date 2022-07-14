from typing import List
import numpy as np
import ray
from .env import make_mario

from icecream import ic

@ray.remote
class RemoteAgent:
    def __init__(self, world:int, stage:int) -> None:
        self.env = make_mario(world, stage)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action: int):
        next_state, reward, done, info = self.env.step(action)
        return (next_state, reward, done, info)

class MultiEnv:
    def __init__(self, world:int = 1, stage: int = 1, num_workers: int = 1) -> None:
        self.num_workers = num_workers
        self.envs = [RemoteAgent.remote(world, stage) for _ in range(num_workers)]
    
    def reset_all(self):
        states = []
        for i in range(self.num_workers):
            states.append(ray.get(self.envs[i].reset.remote()))
        return np.array(states)
    
    def step(self, action: List[int] or int):
        next_states = []
        rewards = []
        dones = []
        infos = []
        if isinstance(action, int):
                next_state, reward, done, info = ray.get(self.envs[0].step.remote(action))
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
        else:
            for i in range(self.num_workers):
                next_state, reward, done, info = ray.get(self.envs[i].step.remote(action[i]))
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
        for index, done in enumerate(dones):
            if done:
                self.envs[index].reset.remote()
        return (np.array(next_states), np.array(rewards), np.array(dones), infos)

def main(num_workers: int = 1) -> None:
    env = MultiEnv(num_workers)
    states = env.reset_all()
    ic(states.shape)
    next_states, rewards, dones, infos = env.step([])
    ic(next_states.shape)
    
if __name__ == "__main__":
    main(32)