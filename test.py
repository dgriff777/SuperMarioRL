from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import torch
from environment import mario_env, STAGE_LIST
from utils import setup_logger
from model import MarioNET
from player_util import Agent
from torch.autograd import Variable
import time
import logging
import random


def test(args, shared_model):
    ptitle("Test Agent")
    gpu_id = args.gpu_ids[-1]
    setup_logger(f"{args.env}_log", rf"{args.log_dir}{args.env}_log")
    log = logging.getLogger(f"{args.env}_log")
    d_args = vars(args)
    for k in d_args.keys():
        log.info(f"{k}: {d_args[k]}")
    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env_id = args.env
    if args.train_stages:
        env_stage = random.choice(STAGE_LIST)
        env_id = f"SuperMarioBros-{env_stage}-v0"
    env = mario_env(env_id, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = MarioNET(player.env.observation_space.shape[0],
                           player.env.action_space, args)
    if args.tensorboard_logger:
        from torch.utils.tensorboard import SummaryWriter
        dummy_input = (
            torch.zeros(1, player.env.observation_space.shape[0], 80, 80),
            torch.zeros(1, args.hidden_size),
            torch.zeros(1, args.hidden_size),
        )
        writer = SummaryWriter(f"runs/{args.env}_training")
        writer.add_graph(player.model, dummy_input, False)
        writer.close()
    player.state = player.env.reset()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = torch.from_numpy(player.state).cuda()
    else:
        player.state = torch.from_numpy(player.state)
    flag = True
    player.env.set_training_on()
    if args.load_rms_stats:
        player.env.load("test_env_data")
        player.env.set_training_off()
    try:
        while 1:
            if player.done:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.model.load_state_dict(shared_model.state_dict())
                else:
                    player.model.load_state_dict(shared_model.state_dict())
            player.action_test()
            reward_sum += player.reward
            if player.done and not player.env.was_real_done:
                state = player.env.reset()
                player.state = torch.from_numpy(state)
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()
            elif player.done and player.env.was_real_done:
                num_tests += 1
                reward_total_sum += reward_sum
                reward_mean = reward_total_sum / num_tests
                log.info(
                    f"Time {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start_time))}, episode reward {reward_sum}, episode length {player.eps_len}, reward mean {reward_mean:.4f}"
                )
                if args.tensorboard_logger:
                    writer.add_scalar(
                        f"{args.env}_Episode_Rewards", reward_sum, num_tests
                    )
                    for name, weight in player.model.named_parameters():
                        writer.add_histogram(name, weight, num_tests)
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(
                            state_to_save, f"{args.save_model_dir}{args.env}.dat"
                        )
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(
                        state_to_save, f"{args.save_model_dir}{args.env}.dat"
                    )
                reward_sum = 0
                player.eps_len = 0
                if args.train_stages:
                    env_stage = random.choice(STAGE_LIST)
                    env_id = f"SuperMarioBros-{env_stage}-v0"
                player.env = mario_env(env_id, args)
                if args.load_rms_stats:
                    player.env.load("test_env_data")
                    player.env.set_training_off()
                else:
                    player.env.save("test_env_data")
                    player.env.set_training_on()
                state = player.env.reset()
                time.sleep(60)
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = torch.from_numpy(state).cuda()
                else:
                    player.state = torch.from_numpy(state)
    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")
    finally:
        print("test agent process finished")
        if args.tensorboard_logger:
            writer.close()
