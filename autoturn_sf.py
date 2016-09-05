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
        3: THRUSTSHOOT - thrust_down, thrust_shoot
        """
        self.action_space = spaces.Discrete(3)

        self.num_features = 11
        low = [0] * self.num_features
        high = [1, 360, 360, 200, 1, 100, 20, 20, np.inf, 1, 1]
        low = np.array(low * self.historylen)
        high = np.array(high * self.historylen)
        self.observation_space = spaces.Box(low, high)

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
                world["ship"]["orientation"],
                world["ship"]["vdir"],
                world["ship"]["distance-from-fortress"],
                world["fortress"]["alive"],
                world["vlner"],
                len(world["missiles"]),
                len(world["shells"]),
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

    def _step(self, action):
        self.__do_action(action)
        world = self.__send_command("continue", ret=True)
        state, reward, done = self.__process_world(world)
        self.state.pop(0)
        self.state.append(state)
        return np.array(list(itertools.chain.from_iterable(self.state))), reward, done, {}

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
        self.thrusting = False
        self.shooting = False
        self.config = None
        self.thrusting = False
        self.shooting = False
        self.score = 0
        self.maxscore = 0
        self.outer_deaths = 0
        self.inner_deaths = 0
        self.fortress_kills = 0
        self.resets = 0
        self.world = {}

    def __do_action(self, action):
        done = False
        reward = 0
        thrusting = self.thrusting
        shooting = self.shooting
        if action == 0:
            if self.thrusting:
                self.__send_command("keyup", "thrust")
                thrusting = False
            if self.shooting:
                self.__send_command("keyup", "fire")
                shooting = False
        elif action == 1:
            if not self.thrusting:
                self.__send_command("keydown", "thrust")
                thrusting = True
            if self.shooting:
                self.__send_command("keyup", "fire")
                shooting = False
        elif action == 2:
            if self.thrusting:
                self.__send_command("keyup", "thrust")
                thrusting = False
            if not self.shooting:
                self.__send_command("keydown", "fire")
                shooting = True
        # elif action == 3:
        #     if not self.thrusting:
        #         self.__send_command("keydown", "thrust")
        #         thrusting = True
        #     if not self.shooting:
        #         self.__send_command("keydown", "fire")
        #         shooting = True

    def __process_world(self, world):
        reward = 0
        if "raw_pnts" in world:
            now = time.time()
            reward += world["raw_pnts"] - self.score
            if not world["ship"]["alive"] and self.world["ship"]["alive"] and "big-hex" in world["collisions"]:
                self.outer_deaths += 1
            if not world["ship"]["alive"] and self.world["ship"]["alive"] and "small-hex" in world["collisions"]:
                self.inner_deaths += 1
            if "fortress" in world["collisions"] and not "fortress" in self.world["collisions"]:
                reward += min(world["vlner"],10)**2
            if not world["fortress"]["alive"] and self.world["fortress"]["alive"]:
                self.fortress_kills += 1
                reward += 1000
            if world["vlner"] == 0 and self.world["vlner"] > 0 and world["fortress"]["alive"]:
                self.resets += 1
                reward -= self.world["vlner"]
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
        return self.__make_state(world, done), reward, done
