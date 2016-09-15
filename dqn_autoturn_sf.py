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
    def __init__(self, env, dqn, filename):
        super(TrainEpisodeFileLogger, self).__init__()
        self.env = env
        self.dqn = dqn
        self.filename = os.path.expanduser(filename)

    def on_train_begin(self, logs):
        self.metrics_names = self.model.metrics_names
        self.file = open(self.filename, "w")
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

        # Format all metrics.
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

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

        if episode % 10 == 0:
            self.dqn.save_weights('dqn_autoturn_sf_weights.h5f', overwrite=True)

def main(args):
    env = AutoturnSF_Env("DQN-SF", 4, 2)
    #env.reset()

    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    #model.add(Flatten(input_shape=(env.historylen,1,env.num_features)))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    if args.mode[0] == "plot":
        from keras.utils.visualize_util import plot
        plot(model, to_file='dqn_autoturn_sf_model.png', show_shapes=True)
        sys.exit(1)
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!

    STEPS_PER_EPISODE = 5455

    # memory = SequentialMemory(limit=STEPS_PER_EPISODE*184)
    # policy = BoltzmannQPolicy(tau=10000, clip=(-200000., 200000.))
    # #policy = EpsGreedyQPolicy(eps=.1)
    # #policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
    # #                          nb_steps=STEPS_PER_EPISODE*917)
    # dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
    #                target_model_update=1e-2, policy=policy)
    # optimizer = Adam(lr=.003)
    # dqn.compile(optimizer, metrics=['mae'])

    memory = SequentialMemory(limit=STEPS_PER_EPISODE*200)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.4, value_min=.01, value_test=.01, nb_steps=STEPS_PER_EPISODE*1000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, window_length=1, memory=memory,
        #nb_steps_warmup=50000, gamma=.99, delta_range=(-200000., 200000.),
        nb_steps_warmup=0, gamma=.99, delta_range=(-200000., 200000.),
        target_model_update=10000, train_interval=4)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if os.path.isfile('dqn_autoturn_sf_weights.h5f'):
        dqn.load_weights('dqn_autoturn_sf_weights.h5f')

    if args.mode[0] == "train":
        log = TrainEpisodeFileLogger(env, dqn, "dqn_autoturn_sf_log.tsv")
        dqn.fit(env, nb_steps=STEPS_PER_EPISODE*2000, visualize=True, verbose=2, callbacks=[log])#, action_repetition=3)
    elif args.mode[0] == "test":
        dqn.test(env, nb_episodes=20, visualize=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Autoturn SF DQN Model')
    parser.add_argument('--mode', nargs=1, choices=['train', 'test', 'plot'], default='plot')
    args = parser.parse_args()
    main(args)
