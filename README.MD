# RL Mario Pytorch

<p align="center"> Mario Playing All 32 Levels</p>
<p align="center">
    <img src="https://github.com/dgriff777/SuperMarioRL/blob/main/demo/demo.gif" alt="Mario Playing all 32 Levels">
</p>

<p align="center"> Mario Playing The Whole Game Straight Through</p>
<p align="center">
    <img src="https://github.com/dgriff777/SuperMarioRL/blob/main/demo/demo2.gif" alt="Mario Playing Whole Game Straight Through">
</p>

# Train One Model to Play All 32 Levels!!
In honor of the recent Super Mario Bros movie, I decided to create a repo to train an agent to play the original Super Mario Bros. What separates this from other successful implementations is that you can train this model to win each level quickly, often under an hour, with just one or two GPUs with a 20-core CPU. You can also train one model to play all the levels successfully as opposed to other implementations I have seen which just train one specific model to play one specific stage successfully. I used an implementation of A3C that utilizes GPU for increased speed in training, which I call **A3G**. In A3G, as opposed to other versions that try to utilize GPU with A3C algorithm, each agent has its own network maintained on GPU while the shared model is on CPU. The A3G agent models are transferred to CPU to update the shared model allowing updates to be frequent and fast using Hogwild Training and making updates to the shared model asynchronously and without locks. This method increases training speed.

*Note on Reward setup: The reward is set up so that the training agent receives points for any new right point, further right than previously attained, that Mario travels to across the stage. For stage 2 on world 2 and 4, I have a custom penalty to deter agent from using the warp zone as I wanted the training agent to play each stage. However, you can comment out those lines in environment.py code if you do not mind the training agent learning to use warp zones and then jumping to new worlds which it can then successfully learn. For world 7 stage 4, there is a puzzle Mario must solve in terms of the correct path he must travel. When Mario successfully completes that puzzle, there is a sound in the game that tells Mario he has solved it. I replace this sound with reward points, but the training agent must still learn what path is required to get the reward with no hints.

## Requirements

- Python 2.7+
- Openai Gym and Universe
- Pytorch (Pytorch 2.0 has a bug where it incorrectly occupies GPU memory on all GPUs being used when backward() is called on training processes. This does not slow down training but it does unnecesarily take up a lot of gpu memory. If this is problem for you and running out of gpu memory downgrade pytorch)


## Training
*When training model it is important to limit number of worker processes to number of cpu cores available as too many processes (e.g. more than one process per cpu core available) will actually be detrimental in training speed and effectiveness*

Example to train agent on a single stage such as SuperMarioBros-1-1-v0, which is stage 1 of world 1 in the game, with 8 different worker processes:

```
python main.py --env SuperMarioBros-1-1-v0 --workers 8 -lrs --amsgrad
```
#A3G


Example to train agent on a single stage such as SuperMarioBros-1-1-v0, which is stage 1 of world 1 in the game, with 36 different worker processes on 3 GPUs with A3G:

```
python main.py --env SuperMarioBros-1-1-v0 --workers 36 --gpu-ids 0 1 2 --amsgrad -lrs
```


Example to train agent on a whole game SuperMarioBros-v0, with 36 different worker processes on 3 GPUs with A3G:

```
python main.py --env SuperMarioBros-v0 --workers 36 --gpu-ids 0 1 2 --amsgrad -lrs
```

Example to train agent on with one worker process training on just one stage of the 32 stages in game with 4 additional worker processes training on the whole game as well with 36 different worker processes on 3 GPUs with A3G:

```
python main.py --env SuperMarioBros-v0 --workers 36 --gpu-ids 0 1 2 --amsgrad -lrs -ts
```

Hit Ctrl C to end training session properly


## Evaluation
To see the one trained model I uploaded with repo playing each of the 32 stages in game
```
python gym_eval.py --env SuperMarioBros-v0 --num-episodes 32 -r -lrs -tps 400 -es 0 -tss
```



## Project Reference

- https://github.com/Kautenja/gym-super-mario-bros
- https://github.com/Kautenja/nes-py
- https://github.com/DLR-RM/stable-baselines3
