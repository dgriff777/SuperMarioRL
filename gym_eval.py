from __future__ import division
import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from torch.autograd import Variable
from environment import mario_env
from utils import read_config, setup_logger
from model import MarioNET
from player_util import Agent
import gym
import logging
import time
from collections import OrderedDict
import torch.autograd.profiler as profiler
import time
from pprint import pformat, pprint
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import crayons
import numpy as np


gym.logger.set_level(40)


parser = argparse.ArgumentParser(description="MARIO_EVAL")
parser.add_argument(
    "-ev",
    "--env",
    default="SuperMarioBros-v0",
    help="environment to train on (default: SuperMarioBros-v0)",
)
parser.add_argument(
    "-ne",
    "--num-episodes",
    type=int,
    default=1,
    help="how many episodes in evaluation (default: 1)",
)
parser.add_argument(
    "-lmr",
    "--load-model-dir",
    default="trained_models/",
    help="folder to load trained models from",
)
parser.add_argument(
    "-ld", "--log-dir", default="logs/", metavar="LG", help="folder to save logs"
)
parser.add_argument(
    "-r", "--render", action="store_true", help="Watch game as it being played"
)
parser.add_argument(
    "-rf",
    "--render-freq",
    type=int,
    default=1,
    help="Frequency to watch rendered game play",
)
parser.add_argument(
    "-mel",
    "--max-episode-length",
    type=int,
    default=100000,
    help="maximum length of an episode (default: 100000)",
)
parser.add_argument(
    "-gp",
    "--gpu-id",
    type=int,
    default=-1,
    help="GPU to use [-1 CPU only] (default: -1)",
)
parser.add_argument(
    "-sr", "--skip-rate", type=int, default=4, help="frame skip rate (default: 4)"
)
parser.add_argument(
    "-s", "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "-nge",
    "--new-gym-eval",
    action="store_true",
    help="Create a gym evaluation for upload",
)
parser.add_argument(
    "-hs",
    "--hidden-size",
    type=int,
    default=512,
    help="LSTM Cell number of features in the hidden state h",
)
parser.add_argument(
    "-tps",
    "--time-per-stage",
    type=int,
    default=400,
    help="time allowed for agent to complete stage",
)
parser.add_argument(
    "-tss",
    "--test-single-stages",
    action="store_true",
    help="test agent on single stages",
)
parser.add_argument(
    "-es",
    "--episode-start",
    type=int,
    default=0,
    help="Used if testing on single stages, parameter value is stage number for agent to run on first (where World: 1, Stage: 1 would be defalut: 0 and World: 8, Stage 4 would be represented with int 31 if want to run first episode on that stage",
)
parser.add_argument(
    "-lrs",
    "--load-rms-stats",
    action="store_true",
    help="load saved running mean stats for observations, running mean is no longer updated",
)
parser.add_argument(
    "-rv",
    "--record-video",
    action="store_true",
    help="record and save video of episode run",
)


args = parser.parse_args()

gpu_id = args.gpu_id

torch.manual_seed(args.seed)
if gpu_id >= 0:
    torch.cuda.manual_seed(args.seed)

saved_state = torch.load(
    f"{args.load_model_dir}{args.env}.dat", map_location=lambda storage, loc: storage
)


setup_logger(f"{args.env}_mon_log", rf"{args.log_dir}{args.env}_mon_log")
log = logging.getLogger(f"{args.env}_mon_log")
env_id = args.env
env = mario_env(env_id, args)
d_args = vars(args)
for k in d_args.keys():
    log.info(f"{crayons.yellow(f'{k}: {d_args[k]}', bold=True)}")


num_tests = 0
start_time = time.time()
reward_total_sum = 0
player = Agent(None, env, args, None)
player.model = MarioNET(
    player.env.observation_space.shape[0], player.env.action_space, args
)

player.gpu_id = gpu_id
if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()
if args.new_gym_eval:
    player.env = gym.wrappers.Monitor(player.env, f"{args.env}_monitor", force=True)

if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model.load_state_dict(saved_state)
        state_to_save = player.model.state_dict()

else:
    player.model.load_state_dict(saved_state)


player.model.eval()
tempList = []
try:
    for i_episode in range(args.num_episodes):
        if args.test_single_stages:
            player.env.close()
            env_id = f"SuperMarioBros-{((args.episode_start+i_episode)//4)+1}-{((args.episode_start+i_episode)%4)+1}-v0"
            player.env = mario_env(env_id, args)
        if args.load_rms_stats:
            player.env.load("test_env_data")
        player.env.set_training_off()
        player.state = player.env.reset()
        if args.record_video:
            video_recorder = VideoRecorder(
                player.env, f"vid_log/{env_id}{i_episode}.mp4", enabled=True
            )

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.state = torch.from_numpy(player.state).cuda()
        else:
            player.state = torch.from_numpy(player.state)
        player.eps_len = 0
        reward_sum = 0
        while 1:
            if args.render:
                if i_episode % args.render_freq == 0:
                    player.env.render()
            if args.record_video:
                video_recorder.capture_frame()

            player.action_test()
            reward_sum += player.reward

            if player.done:
                if (
                    not player.env.unwrapped.is_single_stage_env
                    or not player.env.was_real_done
                ):  # player.info["life"]!=255: #
                    if args.load_rms_stats:
                        player.env.load("test_env_data")
                        player.env.set_training_off()
                    player.state = player.env.reset()

                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            player.state = torch.from_numpy(player.state).cuda()
                    else:
                        player.state = torch.from_numpy(player.state)
                else:
                    num_tests += 1
                    reward_total_sum += reward_sum
                    reward_mean = reward_total_sum / num_tests
                    log.info(
                        "{}".format(
                            crayons.yellow(
                                f"Time {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))}, episode reward {reward_sum}, episode length {player.eps_len}, reward mean {reward_mean:.4f}",
                                bold=True,
                            )
                        )
                    )
                    player.eps_len = 0
                    if args.record_video:
                        video_recorder.close()
                        video_recorder.enabled = False

                    break
except KeyboardInterrupt:
    print("KeyboardInterrupt exception is caught")

finally:
    player.env.close()
