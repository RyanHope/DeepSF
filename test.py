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
from keras.layers import Dense, Activation, Flatten, LocallyConnected1D
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU

import sys,os
sys.path.insert(0, os.path.expanduser("~/workspace/keras-rl"))

from rl.agents.dqn import DQNAgent
from rl.policy import LambdaGreedyQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint

from autoturn_sf import AutoturnSF_Env, SFLogger

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

    alias = "dqn_%s_%s_%s_%d_%d_%d_%d" % (args.reward, args.activation, args.policy, args.neurons, args.interval, args.memlength, args.statehistory)

    logfile_train = os.path.join(base, "%s_train_log.tsv" % alias)
    logfile_test = os.path.join(base, "%s_test_log.tsv" % alias)
    weightsfile = os.path.join(base, "%s_weights.h5f" % alias)

    env = AutoturnSF_Env(alias, args.statehistory, visualize=args.visualize, reward=args.reward, port=args.port)

    nb_actions = env.action_space.n

    model = Sequential()
    #model.add(LocallyConnected1D(15, 4, input_shape=(1,60)))
    #model.add(Flatten())
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(args.neurons))
    model.add(get_activation(args.activation))
    model.add(Dense(args.neurons))
    model.add(get_activation(args.activation))
    model.add(Dense(args.neurons))
    model.add(get_activation(args.activation))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    STEPS_PER_EPISODE = env.get_max_frames() / args.frameskip

    log_train_cb = SFLogger(env, logfile_train)
    log_test_cb = SFLogger(env, logfile_test)
    save_model_cb = ModelIntervalCheckpoint(weightsfile, STEPS_PER_EPISODE*100)

    memory = SequentialMemory(STEPS_PER_EPISODE*args.memlength, window_length=1)
    if args.policy == "eps":
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=.01, value_test=.01, nb_steps=STEPS_PER_EPISODE*1000)
        #policy = EpsGreedyQPolicy(.05)
    elif args.policy == "lam":
        policy = LinearAnnealedPolicy(LambdaGreedyQPolicy(), attr='lam', value_max=1, value_min=30, value_test=30, nb_steps=STEPS_PER_EPISODE*1000)
        #policy = LambdaGreedyQPolicy(8)
    agent = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
        train_interval=args.interval, nb_steps_warmup=STEPS_PER_EPISODE*1, gamma=.999, target_model_update=STEPS_PER_EPISODE*1)
    agent.compile(Adam(lr=.0001), metrics=['mae'])
    if args.weights != None and os.path.isfile(args.weights):
        print("Loading weights from file: %s" % args.weights)
        agent.load_weights(args.weights)
    agent.fit(env, nb_steps=STEPS_PER_EPISODE*args.games, verbose=0, action_repetition=args.frameskip, callbacks=[save_model_cb, log_train_cb])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Autoturn SF DQN Model')
    parser.add_argument('-p','--policy', nargs=1, choices=['eps', 'lam'], default=['eps'])
    parser.add_argument('-a','--activation', nargs=1, choices=['relu', 'elu'], default=['elu'])
    parser.add_argument('-r','--reward', nargs=1, choices=['pnts', 'rawpnts', 'rmh'], default=['pnts'])
    parser.add_argument('-d','--data', default="data")
    parser.add_argument('-w','--weights', default=None)
    parser.add_argument('-v','--visualize', action='store_true')
    parser.add_argument('-f','--frameskip', default=2, type=int)
    parser.add_argument('-i','--interval', default=4, type=int)
    parser.add_argument('-s','--statehistory', default=4, type=int)
    parser.add_argument('-n','--neurons', default=64, type=int)
    parser.add_argument('-m','--memlength', default=200, type=int)
    parser.add_argument('-G','--games', default=10000, type=int)
    parser.add_argument('-P','--port', default=3000, type=int)
    args = parser.parse_args()
    args.policy = args.policy[0]
    args.activation = args.activation[0]
    args.reward = args.reward[0]
    main(args)
