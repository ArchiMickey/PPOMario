from re import L
from typing import Callable, List
import numpy as np
import ray
from torch import nn
from wandb import agent
from env import make_mario

from icecream import ic

@ray.remote
class Remote_env:
    def __init__(self, world:int, stage:int) -> None:
        self.env = make_mario(world, stage)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action: int):
        next_state, reward, done, info = self.env.step(action)
        return (next_state, reward, done, info)

class MultiEnv:
    def __init__(self, num_workers: int = 1) -> None:
        self.num_workers = num_workers
        self.envs = [Remote_env.remote(1, 1) for _ in range(num_workers)]
        ic(self.envs)
    
    def reset(self):
        for i in range(self.num_workers):
            self.envs[i].reset.remote()
    
    def step(self, action: List[int]):
        rt = [[] for _ in range(self.num_workers)]
        for i in range(self.num_workers):
            next_state, reward, done, info = ray.get(self.envs[i].step.remote(np.random.randint(0, 7)))
            exp = (next_state, reward, done, info)
            rt[i] = exp
        return rt

def main(num_workers: int = 1) -> None:
    env = MultiEnv(num_workers)
    env.reset()
    ic(env.step([]))
    
if __name__ == "__main__":
    main(4)