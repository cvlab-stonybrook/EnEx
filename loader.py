#!python3

import os
import json
import csv
import math
import random
import time
import copy

import numpy as np
import sklearn.utils
# import sklearn.preprocessing


def load_phone(L, T, L2):
    desc = 'phone_pick_L_%s_T_%s_L2_%s' % (L, T, L2)
    csvfile, npzfile = map(lambda f: os.path.join('datasets', 'phonepickup', f), ['202003171530.csv', '202003171530.mp4_resnet34.npz'])

    with open(csvfile, 'r') as fp:
        csvreader = csv.reader(fp, delimiter=',')
        segs = list(csvreader)[1 :]
    for anno in segs:
        for i in range(1, 5):
            ts = anno[i].split(':')
            anno[i] = 60 * float(ts[0]) + float(ts[1])

    npz = np.load(npzfile)
    frame_features = npz['features']
    FPS = npz['FPS']
    npz.close()

    durations = []
    for anno in segs:
        durations.append(anno[2] - anno[1])
        durations.append(anno[4] - anno[3])
    durations = np.array(durations)
    print('phone pick: %d m=%.2f std=%.2f [%.2f - %.2f] sec' % (len(durations), durations.mean(), durations.var() ** 0.5, durations.min(), durations.max()))
    assert durations.max() < L2, 'anticipation window %.2f shorter than action duration %.2f' % (L2, durations.max())

    for anno in segs:
        for i in range(1, 5):
            anno[i] = int(FPS  * anno[i])

    L_neg_sample = 10
    L, T, L2, L_neg_sample = map(lambda x: int(FPS * x), [L, T, L2, L_neg_sample])
    X, y = [], []

    for idx in range(0, len(segs) - 1):
        X.append(frame_features[segs[idx][1] - T - L : segs[idx][1] - T].max(axis=0).reshape(1, -1))
        y.append(1)

        for fidx in range(segs[idx + 1][1] - L_neg_sample, segs[idx][4] + L_neg_sample, -1 * L_neg_sample):
            X.append(frame_features[fidx - T - L : fidx - T].max(axis=0).reshape(1, -1))
            y.append(0)

    X = np.concatenate(X, axis=0)
    y = np.array(y, dtype=int)
    print('[%s] X: %s y: %s positive rate: %.5f' % (desc, X.shape, y.shape, y.mean()))
    return X, y, desc


def load_door(L, T, L2):
    desc = 'door_L_%s_T_%s_L2_%s' % (L, T, L2)
    csvfile, npzfile = map(lambda f: os.path.join('datasets', 'opendoor', f), ['202005261550.csv', '202005261550.mp4_resnet34.npz'])

    with open(csvfile, 'r') as fp:
        csvreader = csv.reader(fp, delimiter=',')
        segs = list(csvreader)[1 :]
    for anno in segs:
        for i in (1, 2):
            ts = anno[i].split(':')
            anno[i] = 60 * float(ts[0]) + float(ts[1])

    durations = []
    for anno in segs:
        durations.append(anno[2] - anno[1])
    durations = np.array(durations)
    print('door open: %d m=%.2f std=%.2f [%.2f - %.2f] sec' % (len(durations), durations.mean(), durations.var() ** 0.5, durations.min(), durations.max()))

    npz = np.load(npzfile)
    frame_features = npz['features']
    FPS = npz['FPS']
    npz.close()

    for anno in segs:
        for i in (1, 2):
            anno[i] = int(FPS  * anno[i])

    L_neg_sample = 10
    L, T, L2, L_neg_sample = map(lambda x: int(FPS * x), [L, T, L2, L_neg_sample])
    X, y = [], []

    for idx in range(0, len(segs) - 1):
        X.append(frame_features[segs[idx][1] - T - L : segs[idx][1] - T].reshape(1, L, -1))
        y.append(1)

        for fidx in range(segs[idx + 1][1] - L_neg_sample, segs[idx][2] + L_neg_sample, -1 * L_neg_sample):
            X.append(frame_features[fidx - T - L : fidx - T].reshape(1, L, -1))
            y.append(0)

    X = np.concatenate(X, axis=0).max(axis=1)
    y = np.array(y, dtype=int)
    print('[%s] X: %s y: %s positive rate: %.5f' % (desc, X.shape, y.shape, y.mean()))
    return X, y, desc


def load_epic(action_id, L, T, L2):
    L, T, L2 = map(int, [L, T, L2])
    L_neg_sample = 15
    assert L_neg_sample > (T + L2), 'leading time %.2f or anticipation window %.2f too short' % (T, L2)
    assert (L_neg_sample * 2) > L, 'input window %.2f too long' % L

    with open(os.path.join('datasets', 'epic_kitchens', 'labels.json'), 'r') as fp:
        actions = {v: k.replace(':', '') for k, v in json.load(fp).items()}
    npz = np.load(os.path.join('datasets', 'epic_kitchens', 'heatmaps.npz'))
    heatmaps = {k: npz[k] for k in npz}
    npz.close()

    desc = 'epic_%s_L_%d_T_%d_L2_%d' % (actions[action_id], L, T, L2)

    vid_fps = 60
    X, y = [], []
    L_chunk = L + T + L2

    durations, segs_dict, eps = [], {}, 1e-4
    for vid in heatmaps:
        npz = os.path.join('datasets', 'epic_kitchens', vid + '_i3d.npz')
        if not os.access(npz, os.R_OK):
            continue
        heatmaps[vid] = heatmaps[vid][:, action_id]
        segs_dict[vid] = []
        # print('video %s - %s - %s' % (vid, npz, heatmaps[vid].shape))
        i1 = 0
        while i1 < heatmaps[vid].shape[0] - 1:
            if heatmaps[vid][i1] < eps:
                i1 += 1
            else:
                i2 = i1
                while i2 < heatmaps[vid].shape[0] and heatmaps[vid][i2] > eps:
                    i2 += 1
                durations.append((i2 - i1) / vid_fps)
                segs_dict[vid].append([i1, i2])
                i1 = i2
    durations = np.array(durations)
    print('%s: %d m=%.2f std=%.2f [%.2f - %.2f] sec' % (actions[action_id], len(durations), durations.mean(), durations.var() ** 0.5, durations.min(), durations.max()))

    for vid in heatmaps:
        frame_features = np.load(os.path.join('datasets', 'epic_kitchens', vid + '_i3d.npz'))['features']
        action_segs = copy.deepcopy(segs_dict[vid])
        action_segs.append([frame_features.shape[0] - 1, frame_features.shape[0] - 1])
        action_segs.insert(0, [0, 0])

        for idx in range(0, len(action_segs) - 1):
            action_second = math.floor(action_segs[idx][0] / vid_fps)
            if action_second - T - L >= 0:
                X.append(frame_features[action_second - T - L : action_second - T].max(axis=0).reshape(1, -1))
                y.append(1)

            for second in range(math.ceil(action_segs[idx][1] / vid_fps), math.floor(action_segs[idx + 1][0] / vid_fps) - L_neg_sample * 2, L_neg_sample):
                X.append(frame_features[second : second + L].max(axis=0).reshape(1, -1))
                y.append(0)

    X = np.concatenate(X, axis=0)
    y = np.array(y, dtype=int)
    print('[%s] X: %s y: %s positive rate: %.5f' % (desc, X.shape, y.shape, y.mean()))
    return X, y, desc


if __name__ == '__main__':
    # load_phone(2, 0.5, 2)
    # load_door(5, 2, 5)
    load_epic(0, 5, 1, 5)
    # pass
