from __future__ import print_function, division
import csv
import numpy as np
from tables import *

class SFStateActionInfo(IsDescription):
    sid = StringCol(64)
    game = UInt16Col()
    frame = UInt16Col()
    action = UInt8Col()
    final_score = UInt16Col()
    max_score = UInt16Col()
    shipalive = UInt8Col()
    shipx = Float32Col()
    shipy = Float32Col()
    shipvx = Float32Col()
    shipvy = Float32Col()
    shipo = Float32Col()
    vdir = Float32Col()
    dist = Float32Col()
    fortressalive = UInt8Col()
    missiles = UInt8Col()
    shells = UInt8Col()
    vlner = UInt16Col()
    pnts = UInt16Col()
    thrusting = Int32Col()
    shooting = Int32Col()

if __name__ == '__main__':

    h5file = open_file("all_state_action_pairs.h5", mode="w", title="AutorTurn SF Subject Data")
    table = h5file.create_table("/", "saps", SFStateActionInfo, "State/Action Pairs")
    saps = table.row
    with open('all_state_action_pairs.txt', 'rb') as sap:
        sapreader = csv.reader(sap, delimiter='\t')
        for row in sapreader:
            saps["sid"] = row[0]
            saps["game"], saps["frame"], saps["action"], saps["final_score"], saps["max_score"] = map(int, row[1:6])
            state = map(float, row[6:])
            saps["shipalive"] = state[0]
            saps["shipx"] = state[1]
            saps["shipy"] = state[2]
            saps["shipvx"] = state[3]
            saps["shipvy"] = state[4]
            saps["shipo"] = state[5]
            saps["vdir"] = state[6]
            saps["dist"] = state[7]
            saps["fortressalive"] = state[8]
            saps["missiles"] = state[9]
            saps["shells"] = state[10]
            saps["vlner"] = state[11]
            saps["pnts"] = state[12]
            saps["thrusting"] = state[13]
            saps["shooting"] = state[14]
            saps.append()
    table.flush()
    h5file.close()
