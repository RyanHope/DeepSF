from __future__ import print_function

import json
import socket
import itertools
import time

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class AutoturnSF_Env(gym.Env):

    metadata = {
        'render.modes': ['human'],
        'video.frames_per_second' : 30
    }

    def __init__(self, sid, historylen=8):
        self.sid = sid

        self._seed()

        self.s = None
        self.sf = None

        self.historylen = historylen
        self.state = []

        """
        Discrete actions:
        0: NOOP - thrust_up, shoot_up
        1: THRUST - thrust_down, shoot_up
        2: SHOOT - thrust_up, shoot_down
        3: THRUSTSHOOT - thrust_down, thrust_shoot (disabled)
        """
        self.action_space = spaces.Discrete(3)

        self.num_features = 15
        # low = [0] * (self.num_features-2) + [-np.inf, -np.inf]
        # high = [1, 710, 626, 360, 360, 200, 1, 100, 20, 20, np.inf, np.inf, np.inf]
        # low = np.array(low * self.historylen)
        # high = np.array(high * self.historylen)
        high = np.array([np.inf]*self.num_features*self.historylen)
        self.observation_space = spaces.Box(-high, high)

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

    def _reset(self):
        self._destroy()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("localhost", 3000))
        self.sf = self.s.makefile()
        self.config = json.loads(self.sf.readline())
        self.config = self.__send_command("id", self.sid, ret=True)
        self.world = self.__send_command("continue", ret=True)
        self.state = [self.__make_state(self.world, False)] * self.historylen
        return list(itertools.chain.from_iterable(self.state))

    def _render(self, mode='human', close=False):
        return True

    def _destroy(self):
        if self.sf != None:
            self.sf.close()
            self.sf = None
        if self.s != None:
            self.s.close()
            self.s = None
        self.config = None
        self.config = None
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
        self.resets = 0
        self.reset_vlners = []
        self.world = {}

    def _step(self, action):
        done = False
        reward = 0
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
            if self.world["vlner"] < 10:
                self.shot_intervals[0].append(self.shot_interval)
            else:
                self.shot_intervals[1].append(self.shot_interval)
            self.shot_interval = 0

        world = self.__send_command("continue", ret=True)
        if "raw_pnts" in world:
            now = time.time()
            reward += world["raw_pnts"] - self.score
            if not world["ship"]["alive"] and self.world["ship"]["alive"] and "big-hex" in world["collisions"]:
                self.outer_deaths += 1
                reward -= 900
            if not world["ship"]["alive"] and self.world["ship"]["alive"] and "small-hex" in world["collisions"]:
                self.inner_deaths += 1
                reward -= 900
            if not world["ship"]["alive"] and self.world["ship"]["alive"] and "shell" in world["collisions"]:
                self.shell_deaths += 1
            if "fortress" in world["collisions"] and not "fortress" in self.world["collisions"]:
                reward += world["vlner"]**2
            if not world["fortress"]["alive"] and self.world["fortress"]["alive"]:
                self.fortress_kills += 1
                reward += 99900
            if world["vlner"] == 0 and self.world["vlner"] > 0:
                if world["fortress"]["alive"]:
                    self.resets += 1
                    reward -= self.world["vlner"]
                self.reset_vlners.append(self.world["vlner"])
            if world["vlner"] > self.world["vlner"] and world["fortress"]["alive"] and world["vlner"] > 10:
                reward -= 2
            if shooting and not self.shooting:
                reward += 2
            self.score = world["raw_pnts"]
            if world["pnts"] > self.maxscore:
                self.maxscore = world["pnts"]
        else:
            done = True
        self.world = world
        self.shooting = shooting
        self.thrusting = thrusting
        self.state.pop(0)
        self.state.append(self.__make_state(world, done))
        return np.array(list(itertools.chain.from_iterable(self.state))), reward, done, {}
