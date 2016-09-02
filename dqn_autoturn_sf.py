#!/usr/bin/env python

from __future__ import print_function

import json
import socket

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class AutoturnSF_Env(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 33
    }

    def __init__(self, sid):
        self.sid = sid

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
        high = np.array([np.inf]*11)
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

    def __make_state(self, world):
        return list(map(float, [
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
            len(world["shells"])>0
        ]))

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
        return self.__make_state(self.__send_command("continue", ret=True))

    def _render(self, mode='human', close=False):
        return True

    def _step(self, action):
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
        while True:
            world = self.__send_command("continue", ret=True)
            if "pnts" in world:
                break
            else:
                self.score = 0
        reward = world["raw_pnts"] - self.score
        self.score = world["raw_pnts"]
        state = self.__make_state(world)
        #print(state)
        done = False
        return np.array(state), reward, done, {}

if __name__ == '__main__':

    env = AutoturnSF_Env("DeefSF")
    env.reset()

    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_autoturn_sf_weights.h5f', overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)
