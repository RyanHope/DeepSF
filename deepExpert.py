from __future__ import print_function, division
import csv
from tables import *
import numpy as np

import sys,os
sys.path.insert(0, os.path.expanduser("~/workspace/keras-rl"))

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

from autoturn_sf import AutoturnSF_Env, SFLogger

def onehotaction(action):
    onehot = [0,0,0,0]
    onehot[action] = 1
    return onehot

def main(args):

    alias = "deepExpert"

    env = AutoturnSF_Env(alias, args.statehistory, reward=None, port=args.port, actions=4)
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Dense(64, input_dim=15*args.statehistory, activation="relu"))
    model.add(Dense(64, activation="relu"))
    #model.add(Dense(32, activation="relu"))
    #model.add(Dense(16, activation="relu"))
    model.add(Dense(nb_actions, activation="sigmoid"))
    print(model.summary())

    #opt = SGD(lr=0.00001)
    opt = Adam(lr=.00001)
    model.compile(loss="mean_absolute_error", optimizer=opt, metrics=['accuracy','mean_absolute_error'])

    if args.mode == 'train':
        train = []
        target = []
        h5file = open_file("all_state_action_pairs.h5", mode="r")
        table = h5file.root.saps
        for r in table.iterrows():
            if r["game"] > 1 and r["sid"] == "sfsa28":
                train.append([
                    r["shipalive"], r["shipx"], r["shipy"], r["shipvx"], r["shipvy"],
                    r["shipo"], r["vdir"], r["dist"], r["fortressalive"], r["missiles"],
                    r["shells"], r["vlner"], r["pnts"], r["thrusting"], r["shooting"]
                ])
                target.append(r["action"])
        h5file.close()

        train[0] = [1.0, 245.0, 315.0, 0.0, 1.0, 0.0, 90.0, 110.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0] * (args.statehistory-1) + train[0]
        target[0] = onehotaction(target[0])
        for i in xrange(1, len(train)):
            train[i] = train[i-1][15:] + train[i]
            target[i] = onehotaction(target[i])

        train = np.array(train)
        target = np.array(target)

        epochs_remaining = args.epochs - 100
        while epochs_remaining > 0:
            model.fit(train, target, validation_split=.1, verbose=1, nb_epoch=100)
            model.save_weights("deepExpert_weights.h5", overwrite=T)
            epochs_remaining -= 100
            
    elif args.mode == 'test':
        model.load_weights("deepExpert_weights.h5")
        while True:
            observation = env.reset()
            done = False
            while not done:
                action = model.predict_on_batch(np.array([observation]))[0]
                observation, r, done, _ = env.step(np.argmax(action))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Autoturn SF Expert Trained Model')
    parser.add_argument('-m','--mode', nargs=1, choices=['train', 'test'], default=['train'])
    parser.add_argument('-d','--data', default="data")
    parser.add_argument('-s','--statehistory', default=4, type=int)
    parser.add_argument('-e','--epochs', default=1000, type=int)
    parser.add_argument('-P','--port', default=3000, type=int)
    args = parser.parse_args()
    args.mode = args.mode[0]
    main(args)
