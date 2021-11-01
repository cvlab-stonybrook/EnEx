#!python3


import random
import time
import copy
import numpy as np
import sklearn.utils
from scipy.optimize import linprog
from scipy.interpolate import interp1d

from ml_ridge import KerRidgeExemplars
from utils import concat


class M_Ensemble(object):
    """A class for combining multiple exemplar classifiers. Based on Orderly Weighted Averaging"""
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        :param X: n*d matrix
        :param y: n*1 vector
        minimze mean_{i: y_i > 0.5 } xi_i + mean_{i: y_i < 0.5} xi_i
        s.t. <w, X[i,:]> + b >=  margin - xi_i if y_i = 1
             <w, X[i,:]> + b <= -margin + xi_i if y_i = 0 or -1
             w_1 >= w_2 >= ... >= w_d >= 0
             sum_i w_i = 1
             xi_i >= 0
        """

        n = X.shape[0]
        d = X.shape[1]
        m = d + 1 + n # number of variables, order: w, b, xi

        # linear term for: mean_{i: y_i > 0.5 } xi_i + mean_{i: y_i < 0.5} xi_i
        f = np.zeros(m)
        pos_weight = 1.0/sum(y > 0.5)
        neg_weight = 1.0/sum(y < 0.5)
        for i in range(n):
            if y[i] > 0.5:
                f[d+1+i] = pos_weight
            else:
                f[d+1+i] = neg_weight

        Aub = np.zeros(shape=(n + d -1 , m))
        bub = np.zeros(n+d -1)

        xm = np.mean(X, axis=1)
        margin = np.abs(np.mean(xm[y > 0.5]) - np.mean(xm[y<0.5]))/2

        # <w, X[i, :] > + b >= margin - xi_i if y_i = 1
        # <w, X[i,:]> + b <= -margin + xi_i if y_i = 0 or -1
        for i in range(n):
            if y[i] < 0.5: # negative data point
                Aub[i,:d] = X[i,:] # for x
                Aub[i,d] = 1.0 # for b
            else: # positive data point
                Aub[i,:d] = - X[i,:] # -x*w
                Aub[i,d] = -1.0 # -b
            Aub[i,d+1+i] = -1.0 # for - xi_i
            bub[i] = -margin

        for i in range(d-1):
            Aub[n + i, i] = -1.0 # -w[i] + w[i+1] <= 0
            Aub[n + i, i + 1] = 1.0

        Aeq = np.zeros(shape=(1,m))
        Aeq[0,:d] = 1.0
        beq = np.ones(1)

        bounds = [[0, None] for i in range(m)]
        bounds[d][0] = None

        sol = linprog(f, Aub, bub, Aeq, beq, bounds)
        self.w = sol.x[:d]
        self.b = sol.x[d]

    def decision_function(self, X_t):
        return np.matmul(X_t, self.w) + self.b


class M_Ensemble2(object):
    """A class for combining multiple exemplar classifiers. Based on softmax with tuned temperature parameter.
    Tuning criteria is based on either loglikelihood or aps
    """
    def __init__(self, method="loglikelihood"):
        self.softmax_temperature = 1.0 # value 0.1, behave like mean, value 10 behaves like max
        self.method = method

    def fit(self, X, y):
        """
        :param X: n*d matrix, non-negative values between 0 and 1
        :param y: n*1 vector
        """

        # temperature values to decide.
        # Lower than 0.5 is not worth considering, almost like 0
        # Bigger value than 2 is not robust, bigger than 10 is much like max
        temp_values = [0, 0.5, 1]

        Z = X - np.expand_dims(X.max(axis=1), axis=1)

        if self.method == "loglikelihood":
            loglikelihood = np.zeros(len(temp_values))
            for idx, temp in enumerate(temp_values):
                weights = np.exp(temp * Z)
                norm_weights = weights / np.expand_dims(np.sum(weights, axis=1), axis=1)
                prob = np.sum(X*norm_weights, axis=1)
                prob[y < 0.5] = 1 - prob[y < 0.5]
                #loglikelihood[idx] = np.log(prob).mean()
                loglikelihood[idx] = np.log(prob[y > 0.5]).mean() + np.log(prob[y < 0.5]).mean()

            max_idx = np.argmax(loglikelihood)
        elif self.method == "aps":
            aps = np.zeros(len(temp_values))
            for idx, temp in enumerate(temp_values):
                weights = np.exp(temp * Z)
                norm_weights = weights / np.expand_dims(np.sum(weights, axis=1), axis=1)
                prob = np.sum(X * norm_weights, axis=1)
                _, _, _, ap_thresh, _ = metrics(y, prob, plot=False, verbose=False, recall_thres=[0.1, 1])
                aps[idx] = ap_thresh[0] + 0.1*ap_thresh[1] # for tie breaking

            max_idx = np.argmax(aps)
            print(aps)

        self.softmax_temperature = temp_values[max_idx]
        print("====> {}".format(self.softmax_temperature))

    def decision_function(self, X_t):
        Z = X_t - np.expand_dims(X_t.max(axis=1), axis=1)
        weights = np.exp(self.softmax_temperature*Z)
        norm_weights = weights/np.expand_dims(np.sum(weights, axis=1) , axis=1)
        return np.sum(X_t*norm_weights, axis=1)


class M_Calibrator(object):
    """Obtain the calibration scores and probability values based on the positive and negative scores"""

    def __init__(self, scores_p, scores_n, method='sorted'):
        """
        scores_p: 1D numpy array for the scores of positive instances
        scores_n: 1D numpy array for the scores of negative instances"""

        self.method = method

        dtype = [('score', float), ('isNeg', bool)]
        values = [(sp, False) for sp in scores_p] + [(sn, True) for sn in scores_n]
        sorted_scores = np.array(values, dtype=dtype)
        sorted_scores[::-1].sort(order=['score', 'isNeg']) # sort bassed on decreasing order of score and negative data first

        n_instance = len(sorted_scores)

        probs = [0] * n_instance
        # probs[i], the probability of being positive if the threshold is sorted_scores[i][0]
        # probs is essentially the list of precision values (might not be decreasing)

        idx_p = 0
        for idx_total in range(n_instance):
            if not sorted_scores[idx_total][1]: # positive instance
                idx_p += 1
            probs[idx_total] = float(idx_p) / (idx_total + 1)

        scores = np.array([sorted_scores[i][0] for i in range(n_instance)])
        probs = np.array(probs)

        # alpha = 0.001 # regularization
        # probs = (1-alpha)*probs + alpha*np.linspace(1.0, 0.0, n_instance)


        self.sorted_scores_ = scores
        self.pos_idxs_ = [i for i in range(n_instance) if not sorted_scores[i][1]]
        self.neg_idxs_ = [i for i in range(n_instance) if sorted_scores[i][1]]



        if method == 'unsorted': # raw precision, not sorted:
            self.scores_ = scores
            self.probs_ = probs
        elif method == 'sorted': # sort so that higher score corresponds to higher precision
            # because probs might not be decreasing, we perform an additional step to ensure that
            # by deleting some unnecessary thresholds
            cur_max_val = - np.Inf
            idx2delete = [False]*n_instance

            for i in range(n_instance-1, -1, -1):
                if probs[i] <= cur_max_val:
                    idx2delete[i] = True
                else:
                    cur_max_val = probs[i]

            # We maintain a mapping from raw classifier scores to probability values for being positive
            # Instead of using a parametric function such as Logistic Regression, we use non-parametric function by
            # simply keeping the list of scores and corresponding probability values
            # self.probs_ is strictly decreasing array
            self.scores_ = np.array([scores[i] for i in range(n_instance) if not idx2delete[i]])
            self.probs_ = np.array([probs[i] for i in range(n_instance) if not idx2delete[i]])

        elif method == 'unsorted-smooth': # using kernel interpolation for smoothing
            ref_scores = np.linspace(scores.min(), scores.max(), 100)

            dist = sklearn.metrics.pairwise_distances(ref_scores.reshape(-1,1), scores.reshape(-1,1), metric='euclidean')
            sqr_dist = np.square(dist)
            sorted_sqr_dist = np.sort(sqr_dist, axis=1)
            gammas = 1.0 / sorted_sqr_dist[:, 0:min(4, sqr_dist.shape[1])].mean(axis=1) # based on the average distance to 2 nearest neighbors

            K = np.exp(-np.repeat(gammas.reshape(-1,1), sqr_dist.shape[1], axis=1)*sqr_dist)
            ref_probs = np.matmul(K, probs)/K.sum(1)

            self.scores_ = ref_scores
            self.probs_ = ref_probs

        if len(self.scores_) == 1:
            self.interp_ = interp1d([self.scores_[0], self.scores_[0]], [self.probs_[0], self.probs_[0]],
                                    fill_value=(self.probs_[0],self.probs_[0]), bounds_error=False)
        else:
            self.interp_ = interp1d(self.scores_[::-1], self.probs_[::-1],
                                    fill_value=(self.probs_[-1],self.probs_[0]), bounds_error=False)

    def predict_proba(self, scores):
        return self.interp_(scores)

    def plot(self):
        plt.plot(self.scores_, self.probs_)
        plt.xlabel('score')
        plt.ylabel('probability')
        plt.ylim([0.0, 1.0])

    def plot_raw_score(self):
        plt.scatter(self.neg_idxs_, self.sorted_scores_[self.neg_idxs_], c='b', marker=".")
        plt.scatter(self.pos_idxs_, self.sorted_scores_[self.pos_idxs_], c='r', marker="+")
        plt.title('Raw input scores of positive and negative points')


class M_DataManager(object):
    """Data manager class. Divide data into positive, negative train, postive holdout, negative holdout for calibration"""

    def __init__(self, random_seed=None):
        self.X_p, self.X_n, self.X_ph, self.X_nh = np.array(()), np.array(()), np.array(()), np.array(())
        # hack to keep same random states for different data managers
        self.random_nums = []
        if random_seed is not None:
            random.seed(random_seed)
            for _ in range(0, 50000):
                self.random_nums.append(random.uniform(0, 1))

    def counts(self):
        return self.X_p.shape[0], self.X_n.shape[0], self.X_ph.shape[0], self.X_nh.shape[0]

    def functional(self):
        return self.X_p.shape[0] > 0 and self.X_n.shape[0] > 0 and self.X_nh.shape[0] > 0

    def add_data(self, X, y):
        if y > 0.5:
            self.X_p = concat(self.X_p, X)
        else:
            if len(self.random_nums) > 0:
                r = self.random_nums[-1]
                self.random_nums.pop()
            else:
                r = random.uniform(0, 1)
            if r < 0.5:
                self.X_n = concat(self.X_n, X)
            else:
                self.X_nh = concat(self.X_nh, X)

            if len(self.X_n) == 0:
                self.X_n = concat(self.X_n, X)
            if len(self.X_nh) == 0:
                self.X_nh = concat(self.X_nh, X)

    # Move some data from positive train set X_p to the positive holdout set X_ph
    def reduce_Xp(self, keep_idxs):
        move_data = np.delete(self.X_p, keep_idxs, axis=0)
        self.X_ph = concat(self.X_ph, move_data)
        self.X_p = self.X_p[keep_idxs,:]


class M_EnEx(object):
    """Combination of Examplar and Universal Ridge Regression. Use orderly weighted averaging for combination"""

    def __init__(self, data_manager=None, ex_cmb_methods=['SoftMax'], max_num_ex_clf=np.inf, gamma_factor=1.0, lambda_factor=1.0):
        super().__init__()
        self.ex_clf = None # exemplar classifier
        self.ex_cal = None # exemplar calibraror
        self.ex_cmbs = {} # exemplar combinator
        self.ex_cmb_methods = ex_cmb_methods # methods for combining exemplar classifiers
        self.lambda_factor = lambda_factor
        self.gamma_factor = gamma_factor

        # Maximum number of exemplar classifiers to keep
        # When X_p, the set of exemplars, grows bigger than this size, some exemplars will be moved to a holdout set
        # This would be the size of X_p after this data moving process
        self.max_num_ex_clf = max_num_ex_clf

        if data_manager is None:
            self.data = M_DataManager()
        else:
            self.data = data_manager

    def train(self):
        if not self.data.functional():
            print('Exemplar No trained')
            return

        self.ex_clf = KerRidgeExemplars(self.data.X_p, self.data.X_n, max_num_ex_clf=self.max_num_ex_clf, gamma_factor=self.gamma_factor, lambda_factor=self.lambda_factor)

        if self.max_num_ex_clf < self.data.X_p.shape[0]:
            self.data.reduce_Xp(self.ex_clf.keep_p_idxs)

        self._calibrate()
        self._learn_cmb()

    def prediction_scores(self, X):
        scores = np.random.rand(X.shape[0], len(self.ex_cmb_methods))
        if not self.data.functional() or self.ex_clf is None:
            return scores

        cal_scores = self._get_cal_scores(X)
        cal_scores[:, ::-1].sort(axis=1)  # sort from the second column onward, decreasing order

        for idx, method in enumerate(self.ex_cmb_methods):
            if method == "Max":
                scores[:, idx] = cal_scores[:, 0]
            elif method == "Mean":
                scores[:, idx] = cal_scores.mean(axis=1)
            elif method == "SoftMax":
                # softmax, between max and mean
                # alpha = 0 (or 0.1) mean, alpha=inf (or 10) max
                # range: 0.1 to 2 is good range, default alpha = 1 works well
                alpha = 1.0
                weights = np.exp(alpha * (cal_scores - cal_scores[:, 0:1]))
                norm_weights = weights / np.expand_dims(np.sum(weights, axis=1), axis=1)
                scores[:, idx] = np.sum(cal_scores * norm_weights, axis=1)
            elif method == "RankPool" or method == "TunedSoftMax":
                scores[:,idx] = self.ex_cmbs[method].decision_function(cal_scores)
            else:
                raise Exception("Unknown method")

        return scores

    def _calibrate(self):
        "calibrate exemplar classifier"

        n_p = self.data.X_p.shape[0]
        self.ex_cal = [None] * n_p

        scores_nhs = self.ex_clf.decision_function(self.data.X_nh)
        scores_ps = self.ex_clf.decision_function(self.data.X_p)

        # use hold out positives if available
        if self.data.X_ph.shape[0] > 0:
            scores_phs = self.ex_clf.decision_function(self.data.X_ph)
            scores_ps = concat(scores_ps, scores_phs)

        for idx in range(n_p):
            scores_nh = scores_nhs[:, idx]
            scores_p = scores_ps[:, idx]
            self.ex_cal[idx] = M_Calibrator(scores_p, scores_nh)

    def _learn_cmb(self):

        if "RankPool" in self.ex_cmb_methods or "TunedSoftMax" in self.ex_cmb_methods:
            n_p = self.data.X_p.shape[0]
            n_ph = self.data.X_ph.shape[0]
            n_nh = self.data.X_nh.shape[0]

            # learn the combinator
            feat_nh = self._get_cal_scores(self.data.X_nh)
            feat_p = self._get_cal_scores(self.data.X_p)

            if n_p > 1:
                for idx in range(n_p):
                    # the feature defined on the classifier trained on the positive instance idx
                    # so it is over-optimistic estimate. Treat it as missing value, and estimate based on the other values
                    feat_p[idx, idx] = 0
                    feat_p[idx, idx] = np.sum(feat_p[:,idx])/(n_p - 1)

            # Instead of using the maximum or mean of the calibrated scores
            # We learn a linear combination of the sorted scores (in decreasing order)
            # This is called Orderly Weighted Averaging
            if n_ph > 0:
                feat_ph = self._get_cal_scores(self.data.X_ph)
                feat_p_ph_nh = np.concatenate((feat_p, feat_ph, feat_nh), axis=0 ) # feature matrix
            else:
                feat_p_ph_nh = np.concatenate((feat_p, feat_nh), axis=0)  # feature matrix

            feat_p_ph_nh[:, ::-1].sort(axis=1)  # sort in decreasing order

            lb_p_ph_nh = np.array([1] * (n_p + n_ph) + [0] * n_nh, dtype=np.float)  # label vector

            if 'RankPool' in self.ex_cmb_methods:
                ex_cmb = M_Ensemble()
                ex_cmb.fit(feat_p_ph_nh, lb_p_ph_nh)
                self.ex_cmbs['RankPool'] = ex_cmb

            if 'TunedSoftMax' in self.ex_cmb_methods:
                ex_cmb2 = M_Ensemble2()
                ex_cmb2.fit(feat_p_ph_nh, lb_p_ph_nh)
                self.ex_cmbs['TunedSoftMax'] = ex_cmb2


    def _get_cal_scores(self, X):
        if not self.data.functional() or self.ex_clf is None:
            return np.random.rand(X.shape[0])

        raw_scores = self.ex_clf.decision_function(X)
        cal_scores = np.empty(shape=raw_scores.shape)
        for i in range(0, raw_scores.shape[1]):
            s = raw_scores[:,i]
            cal_scores[:, i] = self.ex_cal[i].predict_proba(s)

        return cal_scores


if __name__ == '__main__':
    pass
