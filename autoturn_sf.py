from __future__ import print_function

import sys
import warnings
import simplejson as json
import socket
import itertools
import time
import timeit

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import keras
from rl.callbacks import Callback, TrainEpisodeLogger

class SFLogger(TrainEpisodeLogger):
    def __init__(self, env, logfile):
        super(SFLogger, self).__init__()
        self.env = env
        self.logfile = logfile
        self.last_step = 0

    def on_train_begin(self, logs):
        self.metrics_names = self.model.metrics_names
        self.file = open(self.logfile, "w")
        header = [
            'id',
            'step',
            'nb_steps',
            'episode',
            'episode_reward',
            'reward_mean'
        ] + self.metrics_names + ['maxscore', 'outer_deaths', 'inner_deaths', 'shell_deaths', 'resets',
                                  'reset_vlners', 'max_vlner', 'isi_pre', 'isi_post', 'fortress_kills',
                                  'raw_pnts', 'total', 'thrust_durations', 'shoot_durations',
                                  'kill_vlners', 'action_noop', 'action_thrust', 'action_shoot']
        if self.env.action_space.n == 4:
            header.append("action_thrustshoot")

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
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}'
        variables = [
            self.env.sid,
            self.step,
            self.params['nb_steps'],
            episode + 1,
            np.sum(self.rewards[episode]),
            np.mean(self.rewards[episode])
        ] + metric_values + [
            self.env.maxscore, self.env.outer_deaths, self.env.inner_deaths, self.env.shell_deaths,
            len(self.env.reset_vlners), np.mean(self.env.reset_vlners) if len(self.env.reset_vlners)>0 else "NA", self.env.max_vlner,
            np.mean(self.env.shot_intervals[0]) if len(self.env.shot_intervals[0])>0 else "NA",
            np.mean(self.env.shot_intervals[1]) if len(self.env.shot_intervals[1])>0 else "NA",
            self.env.fortress_kills, self.env.world["raw-pnts"], self.env.world["total"]
        ] + [np.mean(self.env.thrust_durations), np.mean(self.env.shoot_durations), np.mean(self.env.kill_vlners)] + self.env.actions_taken
        self.file.write("%s\n" % "\t".join(map(str,map(lambda x: "NA" if x=="--" else x, variables))))
        self.file.flush()
        dump = {
            'step': self.step,
            'nb_steps': self.last_step + self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode])
        }
        print(template.format(**dump))

        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]


class AutoturnSF_Env(gym.Env):

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 30
    }

    def __init__(self, sid, historylen=8, game_time=178200, visualize=1, write_logs=0, reward="pnts", port=3000, actions=3, flat=True):
        self.sid = sid

        self._seed()

        self.s = None

        self.flat = flat

        self.historylen = historylen
        self.game_time = game_time
        self.write_logs = write_logs
        self.visualize = visualize
        self.state = []

        self.visualize = visualize

        if reward=="rmh":
            self._reward = self._reward_log
        elif reward=="rawpnts":
            self._reward = self._reward_rawpnts
        elif reward=="pnts":
            self._reward = self._reward_pnts
        else:
            self._reward = lambda x: 0

        """
        Discrete actions:
        0: NOOP - thrust_up, shoot_up
        1: THRUST - thrust_down, shoot_up
        2: SHOOT - thrust_up, shoot_down
        3: THRUSTSHOOT - thrust_down, thrust_shoot
        """
        self.action_space = spaces.Discrete(actions)

        self.num_features = 15
        high = np.array([np.inf]*self.num_features*self.historylen)
        self.observation_space = spaces.Box(-high, high)

        self.buffer = None
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        self.s.connect(("localhost", port))

    def __read_response(self):
        buf = self.s.recv(4096)
        buffering = True
        while buffering:
            if "\n" in buf:
                (line, buf) = buf.split("\n", 1)
                return line
            else:
                more = self.s.recv(4096)
                if not more:
                    buffering = False
                else:
                    buf += more
        if buf:
            return buf

    def get_max_frames(self):
        return self.game_time / int(1./self.metadata["video.frames_per_second"]*1000)

    def __send_command(self, cmd, *args, **kwargs):
        out = {"command": cmd}
        if len(args) > 0:
            out["args"] = args
        out = "%s\n" % json.dumps(out)
        self.s.send(out.encode("utf-8"))
        if "ret" in kwargs and kwargs["ret"]:
            return json.loads(self.__read_response())

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __make_state(self, world, done):
        ret = [0.0] * self.num_features
        if not done:
            ret = list(map(float, [
                world["ship"]["alive"],
                np.around(world["ship"]["x"]),
                np.around(world["ship"]["y"]),
                np.around(world["ship"]["vx"],2),
                np.around(world["ship"]["vy"],2),
                np.around(world["ship"]["orientation"]),
                np.around(world["ship"]["vdir"]),
                np.around(world["ship"]["distance-from-fortress"]),
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

    def _reset(self):
        self._destroy()
        self.config = self.__send_command("continue", ret=True)
        self.config = self.__send_command("id", self.sid, ret=True)
        self.__send_command("config", "game_time", self.game_time, ret=True)
        self.__send_command("config", "display_level", self.visualize, ret=True)
        self.__send_command("config", "write_logs", self.write_logs, ret=True)
        self.world = self.__send_command("continue", ret=True)
        self.state = [self.__make_state(self.world, False)] * self.historylen
        if self.flat:
            s = self.__reshape_state()
        else:
            s = np.array(self.state)
        return s

    def _render(self, mode='human', close=False):
        return True

    def _destroy(self):
        self.thrusting = -1
        self.shooting = -1
        self.score = 0
        self.score2 = 0
        self.maxscore = 0
        self.outer_deaths = 0
        self.inner_deaths = 0
        self.shell_deaths = 0
        self.fortress_kills = 0
        self.shot_intervals = [[],[]]
        self.shot_interval = 0
        self.shoot_durations = []
        self.thrust_durations = []
        self.steps = 0
        self.reset_vlners = []
        self.kill_vlners = []
        self.max_vlner = 0
        self.actions_taken = [0] * self.action_space.n
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

    def _reward_pnts(self, world):
        return world["pnts"] - self.score2

    def _reward_rawpnts(self, world):
        return world["rawpnts"] - self.score

    def _update_stats(self, world):
        if len(world["events"]) > 0:
            if world["vlner"] > self.max_vlner:
                self.max_vlner = world["vlner"]
            if "ship-destroyed" in world["events"]:
                self.steps = 0
            if "explode-bighex" in world["events"]:
                self.outer_deaths += 1
            if "explode-smallhex" in world["events"]:
                self.inner_deaths += 1
            if "shell-hit-ship" in world["events"]:
                self.shell_deaths += 1
            if "fortress-destroyed" in world["events"]:
                self.fortress_kills += 1
                self.kill_vlners.append(self.world["vlner"])
            if "vlner-reset" in world["events"]:
                self.reset_vlners.append(self.world["vlner"])
        self.score = world["rawpnts"]
        self.score2 = world["pnts"]
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
                self.thrust_durations.append(self.thrusting)
                thrusting = 0
            thrusting -= 1
            if self.shooting > 0:
                self.__send_command("keyup", "fire")
                self.shoot_durations.append(self.shooting)
                shooting = 0
            shooting -= 1
        elif action == 1:
            if self.thrusting < 0:
                self.__send_command("keydown", "thrust")
                thrusting = 0
            thrusting += 1
            if self.shooting > 0:
                self.__send_command("keyup", "fire")
                self.shoot_durations.append(self.shooting)
                shooting = 0
            shooting -= 1
        elif action == 2:
            if self.thrusting > 0:
                self.__send_command("keyup", "thrust")
                self.thrust_durations.append(self.thrusting)
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

        if shooting > 0 and self.shooting < 0:
            if self.world["vlner"] < 11:
                self.shot_intervals[0].append(self.shot_interval)
            else:
                self.shot_intervals[1].append(self.shot_interval)
            self.shot_interval = 0
        else:
            self.shot_interval += 1

        self.actions_taken[action] += 1
        world = self.__send_command("continue", ret=True)
        if "rawpnts" in world:
            reward = self._reward(world)
            self._update_stats(world)
        else:
            self.__send_command("continue", ret=False)
            done = True
        self.world = world

        self.shooting = shooting
        self.thrusting = thrusting
        self.state.pop(0)
        self.state.append(self.__make_state(self.world, done))
        if self.flat:
            s = self.__reshape_state()
        else:
            s = np.array(self.state)
        return s, reward, done, {}
