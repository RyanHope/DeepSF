from __future__ import print_function, division
import csv
from tables import *
import numpy as np

import sys,os
sys.path.insert(0, os.path.expanduser("~/workspace/keras-rl"))

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM

from autoturn_sf import AutoturnSF_Env, SFLogger

def onehotaction(action):
    onehot = [0,0,0,0]
    onehot[action] = 1
    return onehot

def main(args):

    alias = "deepExpert"

    env = AutoturnSF_Env(alias, args.statehistory, reward=None, port=args.port, actions=4, flat=False)
    nb_actions = env.action_space.n

    model = Sequential()

    model.add(LSTM(128, input_shape=(args.statehistory, 15)))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(nb_actions))
    model.add(Activation('softmax'))

    print(model.summary())

    opt = Adadelta(lr=1, epsilon=.00001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["precision","recall","fmeasure"])

    if args.mode == 'train':
        if args.weights != None and os.path.isfile(args.weights):
            model.load_weights(args.weights)

        train = []
        target = []

        h5file = open_file("all_state_action_pairs.h5", mode="r")
        table = h5file.root.saps
        for r in table.iterrows():
            if r["game"] == 10 and r["frame"] > 15 and r["sid"] in ["sfsa43","sfsa30","sfsa20"]:
                train.append([
                    r["shipalive"], r["shipx"], r["shipy"], r["shipvx"], r["shipvy"],
                    r["shipo"], r["vdir"], r["dist"], r["fortressalive"], r["missiles"],
                    r["shells"], r["vlner"], r["pnts"], r["thrusting"], r["shooting"]
                ])
                target.append(r["action"])
        h5file.close()

        train[0] = [[1.0, 245.0, 315.0, 0.0, 1.0, 0.0, 90.0, 110.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0]] * (args.statehistory-1) + [train[0]]
        target[0] = onehotaction(target[0])
        for i in xrange(1, len(train)):
            train[i] = train[i-1][1:] + [train[i]]
            target[i] = onehotaction(target[i])

        train = np.array(train)
        target = np.array(target)

        mcp = ModelCheckpoint("deepExpert_weights.h5", save_best_only=False,  save_weights_only=True)
        model.fit(train, target, validation_split=.2, verbose=1, nb_epoch=args.epochs, batch_size=32, callbacks=[mcp])

    elif args.mode == 'test':
        while True:
            try:
                model.load_weights("deepExpert_weights.h5")
                observation = env.reset()
                done = False
                while not done:
                    print(observation)
                    action = model.predict_on_batch(np.array([observation]))[0]
                    print(action,np.argmax(action))
                    observation, r, done, _ = env.step(np.argmax(action))
            except KeyError:
                pass
            except IOError:
                pass


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Autoturn SF Expert Trained Model')
    parser.add_argument('-m','--mode', nargs=1, choices=['train', 'test'], default=['train'])
    parser.add_argument('-d','--data', default="data")
    parser.add_argument('-s','--statehistory', default=4, type=int)
    parser.add_argument('-w','--weights', default=None)
    parser.add_argument('-e','--epochs', default=1000000, type=int)
    parser.add_argument('-P','--port', default=3000, type=int)
    args = parser.parse_args()
    args.mode = args.mode[0]
    main(args)
