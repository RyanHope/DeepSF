from __future__ import print_function

import sys
import warnings
import json
import socket
import itertools
import time
import timeit

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import keras
from rl.callbacks import TrainEpisodeLogger

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
            'id',
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
            self.env.sid,
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
                             len(self.env.reset_vlners), np.mean(self.env.reset_vlners) if len(self.env.reset_vlners)>0 else "NA",
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

class AutoturnSF_Env(gym.Env):

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 30
    }

    def __init__(self, sid, historylen=8, game_time=178200, visualize=False):
        self.sid = sid

        self._seed()

        self.s = None
        self.sf = None

        self.historylen = historylen
        self.game_time = game_time
        self.state = []

        self.visualize = visualize

        """
        Discrete actions:
        0: NOOP - thrust_up, shoot_up
        1: THRUST - thrust_down, shoot_up
        2: SHOOT - thrust_up, shoot_down
        3: THRUSTSHOOT - thrust_down, thrust_shoot (disabled)
        """
        self.action_space = spaces.Discrete(3)

        self.num_features = 15
        high = np.array([np.inf]*self.num_features*self.historylen)
        self.observation_space = spaces.Box(-high, high)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        self.s.connect(("localhost", 3000))
        self.sf = self.s.makefile()
        self.config = json.loads(self.sf.readline())
        self.config = self.__send_command("id", self.sid, ret=True)
        if not self.visualize:
            self.__send_command("config", "display_level", 0, ret=True)
        self.__send_command("config", "game_time", self.game_time, ret=True)

    def get_max_frames(self):
        return self.game_time / int(1./self.metadata["video.frames_per_second"]*1000)

    def __send_command(self, cmd, *args, **kwargs):
        out = {"command": cmd}
        if len(args) > 0:
            out["args"] = args
        out = "%s\r\n" % json.dumps(out)
        self.sf.write(out.encode("utf-8"))
        self.sf.flush()
        if "ret" in kwargs and kwargs["ret"]:
            return json.loads(self.sf.readline())

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __make_state(self, world, done):
        ret = [0.0] * self.num_features
        if not done:
            ret = list(map(float, [
                world["ship"]["alive"],
                world["ship"]["x"],
                world["ship"]["y"],
                world["ship"]["vx"],
                world["ship"]["vy"],
                world["ship"]["orientation"],
                world["ship"]["vdir"],
                world["ship"]["distance-from-fortress"],
                world["fortress"]["alive"],
                len(world["missiles"]),
                len(world["shells"]),
                world["vlner"],
                world["pnts"],
                self.thrusting,
                self.shooting
            ]))
        return ret

    def __reshape_state(self):
        return list(itertools.chain.from_iterable(self.state))
        return state

    def _reset(self):
        self._destroy()
        self.world = self.__send_command("continue", ret=True)
        self.state = [self.__make_state(self.world, False)] * self.historylen
        return self.__reshape_state()

    def _render(self, mode='human', close=False):
        return True

    def _destroy(self):
        self.thrusting = -1
        self.shooting = -1
        self.score = 0
        self.maxscore = 0
        self.outer_deaths = 0
        self.inner_deaths = 0
        self.shell_deaths = 0
        self.fortress_kills = 0
        self.shot_intervals = [[],[]]
        self.shot_interval = 0
        self.steps = 0
        self.reset_vlners = []
        self.world = {}

    def _reward_log(self, world):
        reward = 0
        if world["ship"]["alive"]:
            #reward += self.steps * .5 # survival
            if not world["fortress"]["alive"] and self.world["fortress"]["alive"]:
                reward += 10000 # fortress kill
            else:
                if world["vlner"] > self.world["vlner"]:
                    if world["vlner"] < 12:
                        reward += world["vlner"]**2 # vlner increment
                    else:
                        reward -= 2 # failed kill
                elif world["vlner"] == 0 and self.world["vlner"] != 0:
                    reward -= self.world["vlner"] # reset
        else:
            if self.world["ship"]["alive"]:
                if "big-hex" in world["collisions"]:
                    reward -= 1000 # outer death
                if "small-hex" in world["collisions"]:
                    reward -= 1000 # inner death
                if "shell" in world["collisions"]:
                    reward -= 100 # shell death
        return reward

    def _reward_simple(self, world):
        reward = 0
        # ship death
        if not world["ship"]["alive"] and self.world["ship"]["alive"]:
            reward -= 1
        # reset
        if world["vlner"] == 0 and self.world["vlner"] > 0 and self.world["vlner"] < 11 and world["fortress"]["alive"]:
            reward -= 1
        # failed kill
        if world["vlner"] > self.world["vlner"] and world["fortress"]["alive"] and world["vlner"] > 11:
            reward -= 1
        # vlner increment < 11
        if "fortress" in world["collisions"] and not "fortress" in self.world["collisions"] and world["vlner"] < 12:
            reward += 1
        # kill
        if not world["fortress"]["alive"] and self.world["fortress"]["alive"]:
            reward += 1
        return max(min(reward, 1), -1)

    def _reward_simple2(self, world):
        reward = 0.0
        # ship death
        if not world["ship"]["alive"] and self.world["ship"]["alive"]:
            reward -= 1.0
        # reset
        if world["vlner"] == 0 and self.world["vlner"] > 0 and self.world["vlner"] < 11 and world["fortress"]["alive"]:
            reward -= 0.1
        # failed kill
        if world["vlner"] > self.world["vlner"] and world["fortress"]["alive"] and world["vlner"] > 11:
            reward -= 1.0
        # vlner increment < 11
        if "fortress" in world["collisions"] and not "fortress" in self.world["collisions"] and world["vlner"] < 12:
            reward += 0.1
        # kill
        if not world["fortress"]["alive"] and self.world["fortress"]["alive"]:
            reward += 10
        return reward

    def _update_stats(self, world):
        if not world["ship"]["alive"]:
            self.steps = 0
        if not world["ship"]["alive"] and self.world["ship"]["alive"] and "big-hex" in world["collisions"]:
            self.outer_deaths += 1
        if not world["ship"]["alive"] and self.world["ship"]["alive"] and "small-hex" in world["collisions"]:
            self.inner_deaths += 1
        if not world["ship"]["alive"] and self.world["ship"]["alive"] and "shell" in world["collisions"]:
            self.shell_deaths += 1
        if not world["fortress"]["alive"] and self.world["fortress"]["alive"]:
            self.fortress_kills += 1
        if world["vlner"] == 0 and self.world["vlner"] > 0:
            self.reset_vlners.append(self.world["vlner"])
        self.score = world["rawpnts"]
        if world["pnts"] > self.maxscore:
            self.maxscore = world["pnts"]

    def _step(self, action):
        done = False
        reward = 0
        self.steps += 1
        thrusting = self.thrusting
        shooting = self.shooting
        if action == 0:
            if self.thrusting > 0:
                self.__send_command("keyup", "thrust")
                thrusting = 0
            thrusting -= 1
            if self.shooting > 0:
                self.__send_command("keyup", "fire")
                shooting = 0
            shooting -= 1
        elif action == 1:
            if self.thrusting < 0:
                self.__send_command("keydown", "thrust")
                thrusting = 0
            thrusting += 1
            if self.shooting > 0:
                self.__send_command("keyup", "fire")
                shooting = 0
            shooting -= 1
        elif action == 2:
            if self.thrusting > 0:
                self.__send_command("keyup", "thrust")
                thrusting = 0
            thrusting -= 1
            if self.shooting < 0:
                self.__send_command("keydown", "fire")
                shooting = 0
            shooting += 1
        elif action == 3:
            if self.thrusting < 0:
                self.__send_command("keydown", "thrust")
                thrusting = 0
            thrusting += 1
            if self.shooting < 0:
                self.__send_command("keydown", "fire")
                shooting = 0
            shooting += 1

        if shooting < 0:
            self.shot_interval += 1
        elif self.shooting < 0:
            if self.world["vlner"] < 11:
                self.shot_intervals[0].append(self.shot_interval)
            else:
                self.shot_intervals[1].append(self.shot_interval)
            self.shot_interval = 0

        world = self.__send_command("continue", ret=True)
        if "rawpnts" in world:
            reward = self._reward_log(world)
            self._update_stats(world)
        else:
            done = True
        self.world = world
        self.shooting = shooting
        self.thrusting = thrusting
        self.state.pop(0)
        self.state.append(self.__make_state(self.world, done))
        return self.__reshape_state(), reward, done, {}
