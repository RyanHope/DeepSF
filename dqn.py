#!/usr/bin/env python

"""
Ideas:
  * Reward should scale with rarity (i.e. habituation?)
  * Reward magnitude reflects learning priority and timespan
  * Replay buffer filling up is like a warm-up after taking a break
    - Should buffer be filled/pruned intelligently (e.g. local maxima/minima)
  * Inferring the timing/reset rule is hard because it conflicts with the kill rule
  * thrusting = avoiding negative reward, shooting = gaining positive reward
  * predictive eye movements: saccade to border where impact is imminent
"""

from __future__ import print_function

import os

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from autoturn_sf import AutoturnSF_Env, TrainEpisodeFileLogger

def get_activation(activation):
    if activation == "relu":
        return Activation('relu')
    elif activation == "elu":
        return ELU(alpha=1.0)
    return None

def main(args):

    base = os.path.expanduser(args.data)
    if not os.path.exists(base):
        os.makedirs(base)
    if not os.path.isdir(base):
        raise Exception("Specified data directory is not a directory.")

    alias = "dqn-%s-%s-%d" % (args.activation, args.policy, args.interval)

    logfile = None
    weightfile = None
    logfile = os.path.join(base, "%s_log.tsv" % alias)
    weightsfile = os.path.join(base, "%s_weights.h5f" % alias)

    env = AutoturnSF_Env(alias, 4, 2, visualize=args.visualize)

    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(64))
    model.add(get_activation(args.activation))
    model.add(Dense(64))
    model.add(get_activation(args.activation))
    model.add(Dense(64))
    model.add(get_activation(args.activation))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    STEPS_PER_EPISODE = 5455

    memory = SequentialMemory(limit=STEPS_PER_EPISODE*500)
    if args.policy == "eps":
        # policy = EpsGreedyQPolicy(eps=.1)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.4, value_min=.01, value_test=.005, nb_steps=STEPS_PER_EPISODE*1000)
    elif args.policy == "tau":
        policy = BoltzmannQPolicy(tau=.85)
        # policy = LinearAnnealedPolicy(BoltzmannQPolicy(), attr='tau', value_max=2, value_min=.01, value_test=.01, nb_steps=STEPS_PER_EPISODE*1000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, window_length=1, memory=memory,
        nb_steps_warmup=0, gamma=.99, #delta_range=(-200000., 200000.),
        target_model_update=10*STEPS_PER_EPISODE, train_interval=args.interval)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if args.weights != None and os.path.isfile(args.weights):
        dqn.load_weights(args.weights)

    if args.mode == "train":
        log = TrainEpisodeFileLogger(env, dqn, logfile, weightsfile)
        dqn.fit(env, nb_steps=STEPS_PER_EPISODE*4000, visualize=True, verbose=2, callbacks=[log])#, action_repetition=3)
    elif args.mode == "test":
        dqn.test(env, nb_episodes=20, visualize=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Autoturn SF DQN Model')
    parser.add_argument('-m','--mode', nargs=1, choices=['train', 'test'], default=['train'])
    parser.add_argument('-p','--policy', nargs=1, choices=['eps', 'tau'], default=['eps'])
    parser.add_argument('-a','--activation', nargs=1, choices=['relu', 'elu'], default=['relu'])
    parser.add_argument('-d','--data', default="data")
    parser.add_argument('-w','--weights', default=None)
    parser.add_argument('-v','--visualize', action='store_true')
    parser.add_argument('-i','--interval', default=4, type=int)
    args = parser.parse_args()
    args.mode = args.mode[0]
    args.policy = args.policy[0]
    args.activation = args.activation[0]
    main(args)
