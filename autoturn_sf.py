from __future__ import print_function

import json
import socket
import itertools

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class AutoturnSF_Env(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 33
    }

    def __init__(self, sid):
        self.sid = sid

        self.historylen = 8
        self._seed()

        self.s = None
        self.sf = None

        self._reset()

        """
        Discrete actions:
        0: NOOP - thrust_up, shoot_up
        1: THRUST - thrust_down, shoot_up
        2: SHOOT - thrust_up, shoot_down
        3: THRUSTSHOOT - thrust_down, thrust_shoot
        """
        self.action_space = spaces.Discrete(4)
        high = np.array([np.inf]*13*self.historylen)
        self.observation_space = spaces.Box(-high, high)
        self.state = []

    def __send_command(self, cmd, *args, **kwargs):
        out = {"command": cmd}
        if len(args) > 0:
            out["args"] = args
        out = "%s\r\n" % json.dumps(out)
        self.sf.write(out.encode("utf-8"))
        self.sf.flush()
        if "ret" in kwargs and kwargs["ret"]:
            return json.loads(self.sf.readline())

    def __make_state(self, world, done):
        ret = [0.0] * 13
        if not done:
            ret = list(map(float, [
                world["ship"]["alive"],
                world["ship"]["x"],
                world["ship"]["y"],
                world["ship"]["vx"],
                world["ship"]["vy"],
                world["ship"]["vdir"],
                world["ship"]["distance-from-fortress"],
                world["fortress"]["alive"],
                world["vlner"],
                len(world["missiles"])>0,
                len(world["shells"])>0,
                self.thrusting,
                self.shooting
            ]))
        return ret

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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

    def _reset(self):
        self._destroy()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("localhost", 3000))
        self.sf = self.s.makefile()
        self.config = json.loads(self.sf.readline())
        self.config = self.__send_command("id", self.sid, ret=True)
        self.state = [self.__make_state(self.__send_command("continue", ret=True), False)] * self.historylen
        return list(itertools.chain.from_iterable(self.state))

    def _render(self, mode='human', close=False):
        return True

    def _step(self, action):
        done = False
        reward = 0
        if action == 0:
            if self.thrusting:
                self.__send_command("keyup", "thrust")
                self.thrusting = False
            if self.shooting:
                self.__send_command("keyup", "fire")
                self.shooting = False
        elif action == 1:
            if not self.thrusting:
                self.__send_command("keydown", "thrust")
                self.thrusting = True
            if self.shooting:
                self.__send_command("keyup", "fire")
                self.shooting = False
        elif action == 2:
            if self.thrusting:
                self.__send_command("keyup", "thrust")
                self.thrusting = False
            if not self.shooting:
                self.__send_command("keydown", "fire")
                self.shooting = True
        elif action == 3:
            if not self.thrusting:
                self.__send_command("keydown", "thrust")
                self.thrusting = True
            if not self.shooting:
                self.__send_command("keydown", "fire")
                self.shooting = True
        world = self.__send_command("continue", ret=True)
        if "raw_pnts" in world:
            reward = world["raw_pnts"] - self.score
            self.score = world["raw_pnts"]
        else:
            done = True
        self.state.pop(0)
        self.state.append(self.__make_state(world, done))
        return np.array(list(itertools.chain.from_iterable(self.state))), reward, done, {}