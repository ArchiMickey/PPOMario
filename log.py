from typing import List
import wandb
from moviepy.editor import *


def log_video(env_name: str, frames: List, durations: List, curr_steps: int, episode_reward: int, fps: int):
    clip = ImageSequenceClip(frames, durations=durations)
    clip.write_videofile(f"train_video/{env_name}/mario_step{curr_steps}_reward{episode_reward}.mp4",
                            fps=fps,
                            )
    # wandb.log({f"gameplay": wandb.Video(f"train_video/{env_name}/mario_step{curr_steps}_reward{episode_reward}.mp4",
    #                                     caption=f"reward: {episode_reward}")})