from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def MaxMinFilter(x, mn_d, mx_d):
    new_maxd = 1.0
    new_mind = -1.0
    obs = max(min(x, mx_d), mn_d)
    new_obs = (((obs - mn_d) * (new_maxd - new_mind)) / (mx_d - mn_d)) + new_mind
    new_obs = max(min(new_obs, new_maxd), new_mind)
    return new_obs


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.hidden_size = args.hidden_size


    def action_train(self):
        value, logit, self.hx, self.cx = self.model(self.state, self.hx, self.cx)
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, action)
        state, reward, self.done, self.info = self.env.step(action.item())
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = torch.from_numpy(state).cuda()
        else:
            self.state = torch.from_numpy(state)
        self.eps_len += 1
        reward = MaxMinFilter(reward, -30.0, 30.0)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self


    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = torch.zeros(1, self.hidden_size).cuda()
                        self.hx = torch.zeros(1, self.hidden_size).cuda()
                else:
                    self.cx = torch.zeros(1, self.hidden_size)
                    self.hx = torch.zeros(1, self.hidden_size)
            value, logit, self.hx, self.cx = self.model(self.state, self.hx, self.cx)
            prob = F.softmax(logit, dim=1)
            action = prob.cpu().numpy().argmax()
        state, self.reward, self.done, self.info = self.env.step(action)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = torch.from_numpy(state).cuda()
        else:
            self.state = torch.from_numpy(state)
        self.eps_len += 1
        return self


    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self
