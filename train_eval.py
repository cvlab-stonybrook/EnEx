#!python3

# suppress warnings
def deaf_warn(*args, **kwargs):
    pass
import warnings
warnings.warn = deaf_warn
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=3)

import os
import random
import json
import datetime
import math
import argparse
import sklearn.utils

import loader
from utils import metrics
from models import M_EnEx


def train_eval(args):
    if args.dataset == 'phone':
        X, y, desc = loader.load_phone(args.L, args.T, args.L2)
    elif args.dataset == 'door':
        X, y, desc = loader.load_door(args.L, args.T, args.L2)
    elif args.dataset == 'epic':
        X, y, desc = loader.load_epic(args.action_id, args.L, args.T, args.L2)
    else:
        print('unknown dataset:', args.dataset)
        return

    X, y = sklearn.utils.shuffle(X, y)
    print('[%s] #total: %d, #positive: %d, %%positive: %.2f' % (desc, y.shape[0], y.sum(), y.mean() * 100), flush=True)

    enex = M_EnEx(ex_cmb_methods=['RankPool'])

    max_eval_idx  = X.shape[0]
    max_train_idx = np.int(np.ceil(max_eval_idx * 0.75))
    shld_train = [False] * max_train_idx
    shld_train[-1] = True # always update at the end

    for i in range(0, max_train_idx):
        if y[i]:
            shld_train[i] = True

    recall_thres = [0.1, 0.3, 1.0]
    APs = []
    pred_scores = enex.prediction_scores(X[:max_eval_idx]) # initial scores, model not trained yet

    train_cnt = 0
    for idx in range(0, max_train_idx - 1):
        # Reveal the label at the current time step, add to the classifier
        X_cur, y_cur = X[idx : idx + 1], y[idx]
        enex.data.add_data(X_cur, y_cur)

        # Train, update predictions for future events, and evaluate
        if shld_train[idx]:
            print('idx %d train #%d' % (idx, train_cnt), end=' ')
            print('training: +%d/-%d, heldout: +%d/-%d' % enex.data.counts(), end=' ')

            # training
            enex.train()

            # predict for future time steps
            scores = enex.prediction_scores(X[idx + 1 : max_eval_idx])  # for ALL future events
            pred_scores[idx + 1 :, :] = scores # update the prediction scores for future events only

            # evaluate the model
            ap_valid, _, _, ap_thres_valid, _ = metrics(y[idx + 1 : max_eval_idx], pred_scores[idx + 1 :, 0], plot=False, verbose=False, recall_thres=recall_thres)
            APs.append([train_cnt] + ap_thres_valid)

            print(np.array(ap_thres_valid), flush=True)
            train_cnt += 1

    log_json, logs = 'log_%s.json' % (desc), {}
    if os.path.isfile(log_json):
        with open(log_json, 'r') as fp:
            logs = json.load(fp)
    log_data = {'recall_thres': recall_thres, 'p_rate': y.mean(), 'enex': APs}
    logs[str(datetime.datetime.now())] = log_data
    with open(log_json, 'w') as fp:
        json.dump(logs, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EnEx Training & Evaluation')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--action_id', type=int)
    parser.add_argument('--L', type=float)
    parser.add_argument('--T', type=float)
    parser.add_argument('--L2', type=float)
    parser.add_argument('--runs', type=int)
    args = parser.parse_args()
    print(args)

    for _ in range(0, args.runs):
        train_eval(args)
