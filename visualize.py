#!python3

import os
import glob
import json
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot(json_file):
    print('log JSON: %s' % json_file)
    try:
        with open(json_file, 'r') as fp:
            logs = json.load(fp)
        print('timestamps:', sorted(logs.keys()))
    except:
        print('reading JSON failed:', json_file)
        return

    logs = list(logs.values())
    thres = logs[0]['recall_thres']
    assert len(thres) == 3
    max_iter = 1000000
    for r in logs:
        max_iter = min(max_iter, r['enex'][-1][0])
    baseline = 0
    for l in logs:
        baseline += l['p_rate']
    baseline /= len(logs)
    print('%d runs thresholds=%s max training count %d baseline=%.1f\n' % (len(logs), thres, max_iter, baseline * 100))

    all_aps = [[] for i in range(max_iter +1)]
    for r in logs:
        for aps in r['enex'][: max_iter + 1]:
            if not math.isnan(sum(aps)):
                all_aps[aps[0]].append(aps[1 :])
    all_aps = list(map(lambda x: np.array(x), all_aps))
    mAPs = np.zeros((len(all_aps), len(thres)), dtype=float) - 1

    for it in range(0, len(all_aps)):
        mAPs[it, :] = all_aps[it].mean(axis=0)

    with PdfPages(json_file[: -5] + '_runs_%d.pdf' % (len(logs))) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for thres_idx in range(0, len(thres)):
            ax = axes[thres_idx]

            x = 1 + np.array(range(0, mAPs.shape[0]))
            y = mAPs[:len(x), thres_idx]
            ax.plot(x, y, color='red', linestyle='-', lw=2, alpha=0.75)

            ax.legend(['EnEx %.4f' % mAPs[:, thres_idx].mean()])
            ax.set_xlabel('# Positives')
            ax.set_ylim([0, 1.05])
            xmax = mAPs.shape[0] + 2
            xtick_shift = 5
            if xmax > 30:
                xtick_shift = 20
            if xmax > 100:
                xtick_shift = 50
            ax.set_xlim([0, xmax])
            ax.set_xticks(list(range(0, xmax, xtick_shift)))
            ax.set_title('AP@%s' % thres[thres_idx])
            ax.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Experiment Results')
    parser.add_argument('--files', default='*.json', help='log files generated by training & evaluation')
    parser.add_argument('--check', help='check log files')
    args = parser.parse_args()

    if not args.check is None:
        for file in glob.glob(args.check):
            with open(file, 'r') as fp:
                print('%s: %s' % (file, len(json.load(fp))))
    else:
        for f in glob.glob(args.files):
            plot(f)