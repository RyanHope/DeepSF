from __future__ import division
import os
import sys
import time
import random

from spacefortress import game, screen, config

class GymGame(game.Game, screen.Screen):
    def __init__(self, conf, game_name, game_number):
        screen.Screen.__init__(self, 'game')
        game.Game.__init__(self, conf, game_name, game_number)
        self.steps = 0
        self.objects = None

    def open_logs(self):
        self.log.open_gamelogs(self.config)

    def now(self):
        return time.time()

    def play_sound(self, sound_id):
        pass

    def draw_world(self):
        pass

    def process_input_events(self):
        pass

    def set_objects(self, objects):
        self.objects = objects

    def step(self):
        self.steps += 1
        self.gameTimer.tick(33)
        self.step_one_tick()
        return self.cur_time >= int(self.config["game_time"])

    def run(self):
        self.start()
        while not self.step(): continue
        self.finish()

if __name__ == '__main__':
    gc = config.get_global_config()
    gc.fill_in_missing_keys()

    gnum = gc.get_next_game()
    c = gc.snapshot("fortress")
    g = GymGame(c, "fortress", gnum)
    g.run()
