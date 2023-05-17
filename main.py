from __future__ import print_function, division
import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import mario_env
from utils import read_config
from model import MarioNET
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
import time


parser = argparse.ArgumentParser(description="MARIORL")
parser.add_argument(
    "-l", "--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)"
)
parser.add_argument(
    "-g",
    "--gamma",
    type=float,
    default=0.99,
    help="discount factor for rewards (default: 0.99)",
)
parser.add_argument(
    "-t", "--tau", type=float, default=1.00, help="parameter for GAE (default: 1.00)"
)
parser.add_argument(
    "-ec",
    "--entropy-coef",
    type=float,
    default=0.01,
    help="parameter for entropy loss coefficient (default: 0.01)",
)
parser.add_argument(
    "-vc",
    "--value-coef",
    type=float,
    default=0.5,
    help="parameter for value loss coefficient (default: 0.5)",
)
parser.add_argument(
    "-s", "--seed", type=int, default=1, help="random seed (default: 1)"
)
parser.add_argument(
    "-w",
    "--workers",
    type=int,
    default=32,
    help="how many training processes to use (default: 32)",
)
parser.add_argument(
    "-ns",
    "--num-steps",
    type=int,
    default=20,
    help="number of forward steps per update (default: 20)",
)
parser.add_argument(
    "-mel",
    "--max-episode-length",
    type=int,
    default=100000,
    help="maximum length of an episode (default: 100000)",
)
parser.add_argument(
    "-ev",
    "--env",
    default="SuperMarioBros-v0",
    help="environment to train on (default: SuperMarioBros-v0)",
)
parser.add_argument(
    "-so",
    "--shared-optimizer",
    default=True,
    help="use an optimizer without shared statistics.",
)
parser.add_argument("-ld", "--load", action="store_true", help="load a trained model")
parser.add_argument(
    "-o",
    "--optimizer",
    default="Adam",
    choices=["Adam", "RMSprop"],
    help="optimizer choice of Adam or RMSprop",
)
parser.add_argument(
    "-lmd",
    "--load-model-dir",
    default="trained_models/",
    help="folder to load trained models from",
)
parser.add_argument(
    "-smd",
    "--save-model-dir",
    default="trained_models/",
    help="folder to save trained models",
)
parser.add_argument("-lg", "--log-dir", default="logs/", help="folder to save logs")
parser.add_argument(
    "-gp",
    "--gpu-ids",
    type=int,
    default=[-1],
    nargs="+",
    help="GPUs to use [-1 CPU only] (default: -1)",
)
parser.add_argument(
    "-a", "--amsgrad", action="store_true", help="Adam optimizer amsgrad parameter"
)
parser.add_argument(
    "-sr",
    "--skip-rate",
    type=int,
    default=4,
    help="frame skip rate (default: 4)",
)
parser.add_argument(
    "-hs",
    "--hidden-size",
    type=int,
    default=512,
    help="LSTM Cell number of features in the hidden state h",
)
parser.add_argument(
    "-tl",
    "--tensorboard-logger",
    action="store_true",
    help="Creates tensorboard logger to see graph of model, view model weights and biases, and monitor test agent reward progress",
)
parser.add_argument(
    "-tps",
    "--time-per-stage",
    type=int,
    default=300,
    help="time allowed for agent to complete stage",
)
parser.add_argument(
    "-lrs",
    "--load-rms-stats",
    action="store_true",
    help="load saved running mean stats for observations, running mean is no longer updated",
)
parser.add_argument(
    "-ts",
    "--train_stages",
    action="store_true",
    help="train on stages with a agent training on individual stage. Will need to set workers parameter to 32 or greater to train on all stages",
)


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.gpu_ids != [-1]:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")
    env = mario_env(args.env, args)
    shared_model = MarioNET(env.observation_space.shape[0], env.action_space, args)
    if args.load:
        saved_state = torch.load(
            f"{args.load_model_dir}{args.env}.dat",
            map_location=lambda storage, loc: storage,
        )
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()

    if args.shared_optimizer:
        if args.optimizer == "RMSprop":
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == "Adam":
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad
            )
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    p = mp.Process(target=test, args=(args, shared_model))
    p.start()
    processes.append(p)
    time.sleep(0.001)
    for rank in range(0, args.workers):
        p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
        p.start()
        processes.append(p)
        time.sleep(0.001)
    for p in processes:
        time.sleep(0.001)
        p.join()
