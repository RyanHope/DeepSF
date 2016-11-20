#!/usr/bin/env python

"""
Parse a SF 1.5 or 1.6 logfile into something that R can load.
"""

from __future__ import division
import sys, os
import csv
import json
import math
from Vector2D import Vector2D

def velocity_angle_to_object(ship_position, ship_velocity, fortress_position):
    o_angle = math.degrees(((fortress_position.copy()-ship_position)*Vector2D(1,-1)).angle())
    v_angle = math.degrees(ship_velocity.angle())
    diff = v_angle - o_angle
    if diff > 180.0:
        diff = diff - 360.0
    if diff < -180.0:
        diff = diff + 360.0
    return diff

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: parse.py [LOG_FILE]"
        sys.exit(-1)

    sys.stderr.write("log file: %s\n" % sys.argv[1])
    base = os.path.splitext(sys.argv[1])[0]
    try:
        sid, session, game = os.path.split(base)[1].split("-")
        sinfo = [sid, session, game]
    except ValueError:
        sys.exit(-2)

    fin = open(sys.argv[1], "r")
    version = fin.readline().strip().split(" ")
    sys.stderr.write("log version: %s\n" % version[3])
    if not version[3] in ["1.5","1.6"]:
        raise Exception("Log version %s is not supported" % (version[3]))

    if "mturk" in fin.readline().split():
        fin.readline()

    header = fin.readline().strip().replace("[",'"[').replace("]",']"')
    if version[3] == "1.6":
        header = [row for row in csv.reader([header], delimiter=' ')][0][1:-1]
    else:
        header = [row for row in csv.reader([header], delimiter=' ')][0][1:]
    header[14] = "missiles"
    header[15] = "shells"
    R = len(header)
    f = fin.read().replace("[",'"[').replace("]",']"').splitlines()
    rows = []
    for row in csv.reader(f, delimiter=' '):
        if row[0] == '#': continue
        if version[3] == "1.6":
            del row[-1]
        for i in xrange(R):
            if i in [14,15]:
                if row[i] != '[]': row[i] = "[%s]" %  ",".join(row[i].replace("[","").replace("]","").strip().split())
            elif row[i] == "-": row[i] = "NA"
        rows.append(sinfo + row)

    final_score = int(rows[-1][20])
    max_score = max([int(r[20]) for r in rows])

    fortress_position = Vector2D(355,315)

    thrusting = 0
    shooting = 0
    for i in xrange(len(rows)):
        if rows[i][28] == 'y':
            if thrusting < 0:
                thrusting = 0
            thrusting += 1
        else:
            if thrusting > 0:
                thrusting = 0
            thrusting -= 1
        if rows[i][31] == 'y':
            if shooting < 0:
                shooting = 0
            shooting += 1
        else:
            if shooting > 0:
                shooting = 0
            shooting -= 1
        row = [rows[i][j] for j in [6,7,8,9,10,11,15,17,18,23,20]]
        if row[0] == 'y':
            row[0] = 1
        else:
            row[0:6] = [0] * 6
        row[6] = 1 if row[0] == 'y' else 0
        row[7] = len(json.loads(row[7]))
        row[8] = len(json.loads(row[8]))
        row = map(float, row)
        ship_position = Vector2D(row[1],row[2])
        ship_velocity = Vector2D(row[3],row[4])
        vdir = round(velocity_angle_to_object(ship_position, ship_velocity, fortress_position),2)
        ship_distance = round((ship_position.copy()-fortress_position).norm(),2)
        row.insert(6, ship_distance)
        row.insert(6, vdir)
        action = 0
        if rows[i][28] == 'y' and rows[i][31] == 'n':
            action = 1
        elif rows[i][28] == 'n' and rows[i][31] == 'y':
            action = 2
        elif rows[i][28] == 'y' and rows[i][31] == 'y':
            action = 3
        rows[i] = [sinfo[0], sinfo[2], i, action, final_score, max_score] + row + [thrusting, shooting]

    rl = "%s.rl" % base
    with open(rl, "wb") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerows(rows)

    sys.stderr.write("rl file: %s\n" % rl)
