from asyncio.log import logger
from typing import List
import wandb
from moviepy.editor import *
from pathlib import Path


def log_video(env_name: str, frames: List, durations: List, curr_episodes: int, episode_reward: int, fps: int, is_test: bool):
    if is_test:
        dir_name = "test"
    else:
        dir_name = "train"
        
    Path(f"{dir_name}_video/{env_name}").mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(frames, durations=durations)
    clip.write_videofile(f"{dir_name}_video/{env_name}/mario_episode{curr_episodes}_reward{episode_reward:.3f}.mp4",
                            fps=fps,
                            logger=None,
                            )
    return wandb.Video(f"{dir_name}_video/{env_name}/mario_episode{curr_episodes}_reward{episode_reward:.3f}.mp4", caption=f"reward: {episode_reward:.3f}")