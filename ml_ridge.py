#!python3


import numpy as np
from sklearn import metrics
from sklearn_extra.cluster import KMedoids


class KerRidgeExemplars(object):
    """    Kernel ridge regression for individual exemplars with RBF kernel
    """
    def __init__(self, Xp, Xn, max_num_ex_clf=np.inf, gamma_factor=1.0, lambda_factor=1.0):
        """
        :param Xp: p*d data matrix for negative data
        :param Xn: n*d data matrix for negative data
        :param max_num_ex_clf: maximum number of exemplar classifiers to keep. Default is np.inf, keeping all
        :param gamma_factor:
            gamma = gamma_factor*gamma_default for RBF kernel k(x,z) = exp(-gamma*||x-z||^2)
            gamma_default is the inverse of average distance between positive and negative data instances
        :param lambda_factor:
            lambda = lambda_factor*lambda_default, regularization parameter for ridge regression
            lambda_default is 1e-3*(n+1) where n is the number of negative example
        :return:
            alphas: p*n matrix, alpha = alphas[i,:] is the dual coefficients for the least square SVM of the i-th exemplar
            Suppose alpha is the coefficient for a positive exemplar z,
                the weight vector is: w = sum_j alpha_j(phi(z) - phi(x_j))
                the bias term is:     b = 1 - sum_j alpha_j [k(z,z) - k(z, x_j)]
                the decision value:
                    dec_val = sum_j alpha_j(k(u,z) - k(u,x_j)) + b = b + alpha.sum()*k(u,z) - sum_j alpha_j k(u,x_j)
                    the higher the dec_val, the more it is likely positive
                the square error for being positive:
                    sqr_err = (dec_val - 1)^2
                the square error for being negative:
                    sqr_err = dec_val^2
                one possible approach to turn in to probability to being positive is:
                    prob = exp( - (dec_val - 1)^2)
            The objective function is to minimize:
                lambda ||w||^2 + sum_j (w'*(phi(z) - phi(x_j)) -1)^2

        See m_paperDraft.tex in MyNotes for detail
        """

        n = Xn.shape[0]
        p = Xp.shape[0]

        lambda_default = 1e-3*(n+1)
        lbd = lambda_factor*lambda_default

        dist_p2n = metrics.pairwise_distances(Xn, Xp, metric='euclidean')
        sqr_dist_p2n = np.square(dist_p2n)
        dist = metrics.pairwise_distances(Xn, metric='euclidean')
        sqr_dist = np.square(dist)

        gamma_default = 1.0 / sqr_dist_p2n.mean()
        gamma = gamma_factor*gamma_default

        Kz = np.exp(-gamma * sqr_dist_p2n)
        K = np.exp(-gamma * sqr_dist)

        A = K + 1
        for i in range(n):
            A[i, i] += lbd

        Kz1 = np.concatenate((Kz, np.ones(shape=(n, 1))), axis=1)
        U1 = np.linalg.solve(A, Kz1)  # U = A\Kz1
        U = U1[:, :-1]
        u_star = U1[:, -1]
        u_star_sum = u_star.sum()

        alphas = np.zeros(shape=(p, n))
        bs = np.zeros(p)

        for i in range(p):
            Binv1n = u_star + u_star_sum / (1 - np.dot(Kz[:, i], u_star)) * U[:, i]
            alpha = Binv1n / (1 - np.dot(Kz[:, i], Binv1n))
            alphas[i,:] = alpha
            bs[i] = 1 - alpha.sum() + np.dot(alpha, Kz[:,i])

        self.alphas = alphas
        self.bs = bs
        self.gamma = gamma
        self.Xnp = np.concatenate((Xn, Xp))
        self.n = n
        self.p = p
        self.keep_p_idxs = np.arange(p)

        if max_num_ex_clf < p:
            dec_val = np.transpose(self.decision_function(self.Xnp)) # size p*(n+p)
            kmedoids = KMedoids(n_clusters=max_num_ex_clf, init='k-medoids++').fit(dec_val)
            keep_idxs = kmedoids.medoid_indices_

            self.Xnp = np.concatenate((Xn, Xp[keep_idxs,:]))
            self.alphas = self.alphas[keep_idxs]
            self.bs = self.bs[keep_idxs]
            self.p = max_num_ex_clf
            self.keep_p_idxs = keep_idxs


    def decision_function(self, Xt):
        dist_t = metrics.pairwise_distances(Xt, self.Xnp, metric='euclidean')
        sqr_dist_t = np.square(dist_t)
        Kt = np.exp(-self.gamma * sqr_dist_t)

        n = self.n
        p = self.p
        dec_val = np.zeros(shape=(Xt.shape[0], p))
        for idx in range(p):
            alpha = self.alphas[idx,:]
            b = self.bs[idx]
            dec_val[:,idx] = b + alpha.sum()*Kt[:, n+ idx] - np.matmul(Kt[:, :n], alpha)

        return dec_val


if __name__ == '__main__':
    pass
