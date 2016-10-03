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

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.agents import ContinuousDQNAgent
from rl.random import OrnsteinUhlenbeckProcess

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

    alias = "dqn-%s-%s-%d-%d" % (args.activation, args.policy, args.interval, args.memlength)

    logfile = None
    weightfile = None
    logfile = os.path.join(base, "%s_log.tsv" % alias)
    weightsfile = os.path.join(base, "%s_weights.h5f" % alias)

    env = AutoturnSF_Env(alias, 4, 2, visualize=args.visualize)

    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Build all necessary models: V, mu, and L networks.
    V_model = Sequential()
    V_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(16))
    V_model.add(Activation('relu'))
    V_model.add(Dense(1))
    V_model.add(Activation('linear'))
    print(V_model.summary())

    mu_model = Sequential()
    mu_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(16))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(nb_actions))
    mu_model.add(Activation('linear'))
    print(mu_model.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    x = merge([action_input, Flatten()(observation_input)], mode='concat')
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) / 2))(x)
    x = Activation('linear')(x)
    L_model = Model(input=[action_input, observation_input], output=x)
    print(L_model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    STEPS_PER_EPISODE = 5455

    memory = SequentialMemory(limit=STEPS_PER_EPISODE*args.memlength)
    # if args.policy == "eps":
    #     #policy = EpsGreedyQPolicy(eps=.001)
    #     policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.9, value_min=.025, value_test=.05, nb_steps=STEPS_PER_EPISODE*20)
    # elif args.policy == "tau":
    #     policy = BoltzmannQPolicy(tau=.85)
    #     # policy = LinearAnnealedPolicy(BoltzmannQPolicy(), attr='tau', value_max=2, value_min=.01, value_test=.01, nb_steps=STEPS_PER_EPISODE*1000)
    # dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, window_length=1, memory=memory,
    #     nb_steps_warmup=0, gamma=.99, #delta_range=(-200000., 200000.),
    #     target_model_update=2*STEPS_PER_EPISODE, train_interval=args.interval)
    random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
    agent = ContinuousDQNAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                               memory=memory, nb_steps_warmup=100, random_process=random_process,
                               gamma=.99, target_model_update=2*STEPS_PER_EPISODE)
    agent.compile(Adam(lr=.001), metrics=['mae'])

    if args.weights != None and os.path.isfile(args.weights):
        agent.load_weights(args.weights)

    if args.mode == "train":
        log = TrainEpisodeFileLogger(env, agent, logfile, weightsfile)
        agent.fit(env, nb_steps=STEPS_PER_EPISODE*4000, visualize=True, verbose=2, callbacks=[log])#, action_repetition=3)
    elif args.mode == "test":
        agent.test(env, nb_episodes=20, visualize=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Autoturn SF DQN Model')
    parser.add_argument('-M','--mode', nargs=1, choices=['train', 'test'], default=['train'])
    parser.add_argument('-p','--policy', nargs=1, choices=['eps', 'tau'], default=['eps'])
    parser.add_argument('-a','--activation', nargs=1, choices=['relu', 'elu'], default=['relu'])
    parser.add_argument('-d','--data', default="data")
    parser.add_argument('-w','--weights', default=None)
    parser.add_argument('-v','--visualize', action='store_true')
    parser.add_argument('-i','--interval', default=4, type=int)
    parser.add_argument('-m','--memlength', default=200, type=int)
    args = parser.parse_args()
    args.mode = args.mode[0]
    args.policy = args.policy[0]
    args.activation = args.activation[0]
    main(args)
