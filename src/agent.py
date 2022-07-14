from re import L
from typing import Callable
import numpy as np
import ray
from torch import nn
from wandb import agent
from .env import make_mario

from icecream import ic

@ray.remote
class Agent:
    def __init__(self, world:int, stage:int) -> None:
        self.env = make_mario(world, stage)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action: int):
        next_state, reward, done, info = self.env.step(action)
        return (next_state, reward, done, info)

def main():
    agents = [Agent.remote(1, 1) for _ in range(6)]
    ic(agents)
    for i in range(6):
        agents[i].reset.remote()
    while True:
        for i in range(6):
            ic(ray.get(agents[i].step.remote(np.random.randint(0, 7))))
        for i in range(6):
            agents[i].reset.remote()
    
    
if __name__ == "__main__":
    main()