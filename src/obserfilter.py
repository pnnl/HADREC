# Code in this file is copied and adapted from
# https://github.com/ray-project/ray/blob/master/python/ray/rllib/utils/filter.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):

    def __init__(self, shape=None):
        self._n = 0
        self._M = np.zeros(shape, dtype = np.float64)
        self._S = np.ones(shape,  dtype = np.float64)

    def copy(self):
        other = RunningStat()
        other._n = self._n
        other._M = np.copy(self._M)
        other._S = np.copy(self._S)
        return other

    def push(self, x):
        x = np.asarray(x)
        # Unvectorized update of the running statistics.
        assert x.shape == self._M.shape, ("x.shape = {}, self.shape = {}"
                                          .format(x.shape, self._M.shape))
        n1 = self._n
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            delta = x - self._M
            self._M[...] += delta / self._n
            self._S[...] += delta * delta * n1 / self._n
            

    def update(self, other):
        n1 = self._n
        n2 = other._n
        n = n1 + n2
        delta = self._M - other._M
        delta2 = delta * delta
        M = (n1 * self._M + n2 * other._M) / n
        S = self._S + other._S + delta2 * n1 * n2 / n
        self._n = n
        self._M = M
        self._S = S

    def __repr__(self):
        return '(n={}, mean_mean={}, mean_std={})'.format(
            self.n, np.mean(self.mean), np.mean(self.std))

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        aa = np.array(self.var, dtype=np.float64)
        return np.sqrt(aa)
        #return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape



