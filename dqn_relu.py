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

import sys, os
import numpy as np
import timeit
import warnings

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import TrainEpisodeLogger

from autoturn_sf import AutoturnSF_Env

class TrainEpisodeFileLogger(TrainEpisodeLogger):
    def __init__(self, env, dqn, logfile, weightsfile):
        super(TrainEpisodeFileLogger, self).__init__()
        self.env = env
        self.dqn = dqn
        self.logfile = logfile
        self.weightsfile = weightsfile

    def on_train_begin(self, logs):
        self.metrics_names = self.model.metrics_names
        self.file = open(self.logfile, "w")
        header = [
            'step',
            'nb_steps',
            'episode',
            'duration',
            'episode_steps',
            'sps',
            'episode_reward',
            'reward_mean',
            'reward_min',
            'reward_max',
            'action_mean',
            'action_min',
            'action_max',
            'obs_mean',
            'obs_min',
            'obs_max'
        ] + self.metrics_names + ['maxscore', 'outer_deaths', 'inner_deaths', 'shell_deaths', 'resets',
                                  'reset_vlners', 'isi_pre', 'isi_post', 'fortress_kills', 'raw_pnts', 'total']
        self.file.write("%s\n" % "\t".join(header))
        self.file.flush()
        self.train_start = timeit.default_timer()

    def on_train_end(self, logs):
        duration = timeit.default_timer() - self.train_start
        self.file.write("\n")
        self.file.close()

    def on_episode_end(self, episode, logs):
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        metrics = np.array(self.metrics[episode])
        metric_values = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                try:
                    value = np.nanmean(metrics[:, idx])
                except Warning:
                    value = '--'
                metric_values.append(value)

        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        variables = [
            self.step,
            self.params['nb_steps'],
            episode + 1,
            duration,
            episode_steps,
            float(episode_steps) / duration,
            np.sum(self.rewards[episode]),
            np.mean(self.rewards[episode]),
            np.min(self.rewards[episode]),
            np.max(self.rewards[episode]),
            np.mean(self.actions[episode]),
            np.min(self.actions[episode]),
            np.max(self.actions[episode]),
            np.mean(self.observations[episode]),
            np.min(self.observations[episode]),
            np.max(self.observations[episode])
        ] + metric_values + [self.env.maxscore, self.env.outer_deaths, self.env.inner_deaths, self.env.shell_deaths,
                             self.env.resets, np.mean(self.env.reset_vlners) if len(self.env.reset_vlners)>0 else "NA",
                             np.mean(self.env.shot_intervals[0]) if len(self.env.shot_intervals[0])>0 else "NA",
                             np.mean(self.env.shot_intervals[1]) if len(self.env.shot_intervals[1])>0 else "NA",
                             self.env.fortress_kills, self.env.world["raw-pnts"], self.env.world["total"]]
        self.file.write("%s\n" % "\t".join(map(str,map(lambda x: "NA" if x=="--" else x, variables))))
        self.file.flush()

        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

        if episode % 10 == 0:
            self.dqn.save_weights(self.weightsfile, overwrite=True)

def main(args):

    base = os.path.expanduser(args.data)
    if not os.path.exists(base):
        os.makedirs(base)
    if not os.path.isdir(base):
        raise Exception("Specified data directory is not a directory.")

    logfile = None
    weightfile = None
    if args.log != None:
        logfile = os.path.join(base, "%s_log.tsv" % args.log)
        weightsfile = os.path.join(base, "%s_weights.h5f" % args.log)

    env = AutoturnSF_Env("DQN-SF-RELU", 4, 2, visualize=args.visualize)

    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    STEPS_PER_EPISODE = 5455

    memory = SequentialMemory(limit=STEPS_PER_EPISODE*200)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.4, value_min=.01, value_test=.01, nb_steps=STEPS_PER_EPISODE*1000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, window_length=1, memory=memory,
        nb_steps_warmup=0, gamma=.99, delta_range=(-200000., 200000.),
        target_model_update=10000, train_interval=4)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if args.weights != None and os.path.isfile(args.weights):
        dqn.load_weights(args.weights)

    if args.mode == "train":
        log = TrainEpisodeFileLogger(env, dqn, logfile, weightsfile)
        dqn.fit(env, nb_steps=STEPS_PER_EPISODE*2000, visualize=True, verbose=2, callbacks=[log])#, action_repetition=3)
    elif args.mode == "test":
        dqn.test(env, nb_episodes=20, visualize=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Autoturn SF DQN Model')
    parser.add_argument('-m','--mode', nargs=1, choices=['train', 'test'], default=['train'])
    parser.add_argument('-d','--data', default="data")
    parser.add_argument('-l','--log', default=None)
    parser.add_argument('-w','--weights', default=None)
    parser.add_argument('-v','--visualize', action='store_true')
    args = parser.parse_args()
    args.mode = args.mode[0]
    main(args)
