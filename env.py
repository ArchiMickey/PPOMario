from icecream import ic
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import ResizeObservation, FrameStack, GrayScaleObservation, TransformObservation
from pl_bolts.models.rl.common.gym_wrappers import MaxAndSkipEnv
from gym import Wrapper
import numpy as np

class CustomReward(Wrapper):
    def __init__(self, env) -> None:
        super(CustomReward, self).__init__(env)
        self.curr_score = 0
        self.current_x = 40
    
    def step(self, action):
        next_state, reward, done, info = super().step(action)
        reward += (info['score'] - self.curr_score) / 40.
        self.curr_score = info['score']
        
        if done:
            if info['flag_get']:
                reward += 50
            else:
                reward -= 50
        
        self.current_x = info['x_pos']
        return next_state, reward / 10., done, info

def make_mario(world: int = 1, stage: int = 1, action_space: str = 'simple'):
    if action_space == 'right':
        actions = RIGHT_ONLY
    elif action_space == 'complex':
        actions = COMPLEX_MOVEMENT
    else:
        actions = SIMPLE_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-{}-{}-v0'.format(world, stage))
    env = CustomReward(env)
    env = JoypadSpace(env, actions=actions)
    env = ResizeObservation(env, 84)
    env = GrayScaleObservation(env, keep_dim=False)
    env = TransformObservation(env, lambda x: x.astype(np.float32) / 255.)
    env = FrameStack(env, num_stack=4)
    env = MaxAndSkipEnv(env, skip=4)
    
    return env

def main():
    env = make_mario(world=1, stage=1, action_space='simple')
    env.reset()
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        ic(env.observation_space.shape, env.action_space.n, next_state.shape, reward, done, info)
        # env.observation_space.shape: (4, 84, 84)
        # env.action_space.n: 7
        # next_state.shape: (4, 84, 84)
        # reward: -16.0
        # done: True
        # info: {'coins': 0,
        #     'flag_get': False,
        #     'life': 2,
        #     'score': 0,
        #     'stage': 1,
        #     'status': 'small',
        #     'time': 363,
        #     'world': 1,
        #     'x_pos': 715,
        #     'y_pos': 79}
    
if __name__ == '__main__':
    main()