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
from keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from keras.optimizers import SGD, Adam, Adagrad, Adadelta, Nadam
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM

import sys,os
sys.path.insert(0, os.path.expanduser("~/workspace/keras-rl"))

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
#from rl.policy import LambdaImpendingFailureQPolicy, AccumulatingImpendingFailureQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint

from autoturn_sf import AutoturnSF_Env, SFLogger

def get_activation(activation):
    if activation == "relu":
        return Activation('relu')
    elif activation == "elu":
        return ELU(alpha=1.0)
    return None

def make_sf_dqn_model(args, env):
    inshape = (1,) + env.observation_space.shape

    # Input layer
    model = Sequential()
    if args.lstm:
        model.add(Reshape((args.statehistory, 15), input_shape=inshape))
    else:
        model.add(Flatten(input_shape=inshape))

    # Hidden layer 1
    if args.lstm:
        model.add(LSTM(args.neurons))
    else:
        model.add(Dense(args.neurons))
    if args.batchnorm:
        model.add(BatchNormalization())
    model.add(get_activation(args.activation))
    if args.dropout > 0:
        model.add(Dropout(args.dropout))

    # Hidden layer 2
    model.add(Dense(args.neurons/2))
    if args.batchnorm:
        model.add(BatchNormalization())
    model.add(get_activation(args.activation))
    if args.dropout > 0:
        model.add(Dropout(args.dropout))

    # Hidden layer 3
    model.add(Dense(args.neurons/4))
    if args.batchnorm:
        model.add(BatchNormalization())
    model.add(get_activation(args.activation))
    if args.dropout > 0:
        model.add(Dropout(args.dropout))

    # Output layer
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))

    return model

def main(args):

    base = os.path.expanduser(args.data)
    if not os.path.exists(base):
        os.makedirs(base)
    if not os.path.isdir(base):
        raise Exception("Specified data directory is not a directory.")

    aargs = [
        "dqn",
        args.reward,
        args.activation,
        args.policy,
        "lstm" if args.lstm else None,
        "bn" if args.batchnorm else None,
        args.neurons,
        args.interval,
        args.memlength,
        args.statehistory,
        args.dropout,
        args.learningrate]
    alias = "_".join([str(a) for a in aargs if a!=None])

    logfile_train = os.path.join(base, "%s_train_log.tsv" % alias)
    logfile_test = os.path.join(base, "%s_test_log.tsv" % alias)
    weightsfile = os.path.join(base, "%s_weights.h5f" % alias)

    env = AutoturnSF_Env(alias, args.statehistory, visualize=args.visualize, reward=args.reward, port=args.port)

    model = make_sf_dqn_model(args, env)
    print(model.summary())

    STEPS_PER_EPISODE = env.get_max_frames() / args.frameskip

    log_train_cb = SFLogger(env, logfile_train)
    log_test_cb = SFLogger(env, logfile_test)
    save_model_cb = ModelIntervalCheckpoint(weightsfile, STEPS_PER_EPISODE*100)

    memory = SequentialMemory(STEPS_PER_EPISODE*args.memlength, window_length=1)
    if args.policy == "eps":
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.4, value_min=.01, value_test=.01, nb_steps=STEPS_PER_EPISODE*1000)
    # elif args.policy == "lam":
    #     policy = LinearAnnealedPolicy(LambdaImpendingFailureQPolicy(), attr='lam', value_max=1, value_min=30, value_test=30, nb_steps=STEPS_PER_EPISODE*1000)
    # elif args.policy == "alpha":
    #     policy = LinearAnnealedPolicy(AccumulatingImpendingFailureQPolicy(), attr='alpha', value_max=.05, value_min=.00005, value_test=.00005, nb_steps=STEPS_PER_EPISODE*1000)
    agent = DQNAgent(model=model, nb_actions=env.action_space.n, policy=policy, memory=memory,
        train_interval=args.interval, nb_steps_warmup=STEPS_PER_EPISODE*1, gamma=.999, target_model_update=STEPS_PER_EPISODE*1)
    opt = Nadam(lr=args.learningrate)
    agent.compile(opt, metrics=['mean_absolute_error'])
    if args.weights != None and os.path.isfile(args.weights):
        print("Loading weights from file: %s" % args.weights)
        agent.load_weights(args.weights)
    agent.fit(env, nb_steps=STEPS_PER_EPISODE*args.games, verbose=0, action_repetition=args.frameskip, callbacks=[save_model_cb, log_train_cb])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Autoturn SF DQN Model')
    parser.add_argument('-p','--policy', nargs=1, choices=['eps', 'lam', 'alpha'], default=['eps'])
    parser.add_argument('-a','--activation', nargs=1, choices=['relu', 'elu'], default=['elu'])
    parser.add_argument('-r','--reward', nargs=1, choices=['pnts', 'rawpnts', 'rmh'], default=['pnts'])
    parser.add_argument('-d','--data', default="data")
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('-D','--dropout', default=0, type=float)
    parser.add_argument('-w','--weights', default=None)
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('-v','--visualize', action='store_true')
    parser.add_argument('-l','--learningrate', default=1, type=float)
    parser.add_argument('-f','--frameskip', default=2, type=int)
    parser.add_argument('-i','--interval', default=4, type=int)
    parser.add_argument('-s','--statehistory', default=4, type=int)
    parser.add_argument('-n','--neurons', default=64, type=int)
    parser.add_argument('-m','--memlength', default=200, type=int)
    parser.add_argument('-G','--games', default=100000, type=int)
    parser.add_argument('-P','--port', default=3000, type=int)
    args = parser.parse_args()
    args.policy = args.policy[0]
    args.activation = args.activation[0]
    args.reward = args.reward[0]
    main(args)
