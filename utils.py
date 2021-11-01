#!python3

import math
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


'''Helper function to concatenate two numpy arrays'''
def concat(a, b):
    if a.shape[0] == 0:
        return b
    else:
        return np.concatenate((a, b), axis=0)


def metrics(y, y_score, plot=False, baseline=None, plot_save=None, title=None, verbose=True, recall_thres=[]):
    if type(y) == type([]):
        y = np.array(y)
    if type(y_score) == type([]):
        y_score = np.array(y_score)
    ap = average_precision_score(y, y_score)
    curve_p, curve_r, thres = precision_recall_curve(y, y_score)
    curve_p, curve_r = curve_p[0 : -1], curve_r[0 : -1]

    ap_thres, score_thres = [], []
    for r in recall_thres:
        t = thres[(curve_r >= r).sum() - 1]
        mask = (y_score >= t)
        ap = average_precision_score(y[mask], y_score[mask])
        ap_thres.append(ap)
        score_thres.append(t)

    if verbose:
        print('precision %.5f recall %.5f' % (precision, recall))
        print('AP = %.6f' % ap, flush=True)
    if plot:
        plt.figure(figsize=(8, 8))
        plt.title('%s Precision-Recall Curve\nAP = %.6f' % (title, ap))
        plt.plot(curve_r, curve_p, color='red', lw=3)
        if baseline is not None:
            plt.plot([0.0, 1.0], [baseline, baseline], color='green', lw=1)
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.tight_layout()
        if plot_save is None:
            plt.show()
        else:
            plt.savefig(plot_save)
        plt.close()

    return ap, curve_p, curve_r, ap_thres, score_thres


if __name__ == '__main__':
    pass
