# RL Mario Pytorch

![Mario Playing All 32 Levels](https://github.com/dgriff777/SuperMarioRL/blob/master/demo/demo.gif) ![Mario Playing Whole Game Straight Through](https://github.com/dgriff777/SuperMarioRL/blob/master/demo/demo2.gif) 

# Train One Model to Play All 32 Levels!!
In honor of the recent Super Mario Bros movie I decided to create a repo to train an agent to play the original Super Mario Bros. What seperates this from other implementations successful implementations is that you can a train a model to win every level rather quickly, often under an hour with just one or two GPUs with a 20-core cpu. You can also train one model to play all the levels succesfully opposed to other implementations I have seen which just train one specific model to play one specific stage successful. We use implementation of A3C that utilizes GPU for speed increase in training. Which we can call **A3G**. A3G as opposed to other versions that try to utilize GPU with A3C algorithm, with A3G each agent has its own network maintained on GPU but shared model is on CPU and agent models are quickly converted to CPU to update shared model which allows updates to be frequent and fast by utilizing Hogwild Training and make updates to shared model asynchronously and without locks. This method greatly increase training speed.

*Note on Reward setup: For training agent the reward is setup so that agent receives points for any new further right point mario travels across stage. For stage 2 on world 2 and 4 I have a custom penalty to deter agent to use the warp zone as I wanted agent to play each stage but one can comment out those lines in environment.py code if you don't mind agent learning to use warp zones and jumping to new worlds which it can successfully learn. For world 7 stage 4 there is a puzzle mario must solve in terms of correct path he must travel and when mario successful completes that puzzle there is a sound in game that tells mario he completed which I give in the form of a reward but the agen must still learn what that path is to get reward with no hints to which paths to take only the reward for completing all the paths he must take.

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
