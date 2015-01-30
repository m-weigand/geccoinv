# -*- coding: utf-8 -*-
"""
Regularization functions
"""
import numpy as np
import scipy.sparse as sparse


class DifferenceWeighting(object):
    """
    Implement a difference weighting scheme, e.g. to weight the regularization
    matrix with time differences

    See T. Günther (2004) - Inversion Methods and Resolution Analysis for the
    2D/3D Reconstruction of Resistivity Structures from DC Measurements. PhD
    Thesis, Bergbauakademie Freiburg, page 20
    """
    def __init__(self, data):
        self.data = data

    def get_C(self):
        r"""
        Create a diagonal matrix with
        :math:`C_{i,i} = \frac{1}{\sqrt(t_{i+1} - t_{i})}`
        """
        diff = np.diff(self.data)
        weighting_factors = 1 / np.sqrt(diff)
        # weighting_factors = 1 / (diff)

        C = np.diag(weighting_factors)
        C = sparse.csc_matrix(C)
        return C


class BaseRegularization(object):
    """
    Inherit all regularizations from this class
    """
    def __init__(self, decouple=None, outside_first_dim=None,
                 weighting_obj=None):
        """
        Parameters
        ----------
        decouple : [None or list/tuple] set all axes to 0 in order decouple
                   those axes. Start with zero.
        outside_first_dim : [None or list/tuple] this is only used for the
                            aggregation to all dimensions. If this object is
                            used to regularize dimensions other than 0, this
                            list decides which base-dimensions will be
                            regularized
        weighting_obj :
        """
        self.decouple = decouple
        self.outside_first_dim = outside_first_dim
        self.weighting_obj = weighting_obj

    def WtWm(self, parsize):
        Wm = self.Wm(parsize)
        WtWm = Wm.T.dot(Wm)
        return WtWm


class Damping(BaseRegularization):
    """
    Return a diagonal unit matrix as a damping matrix
    """
    def Wm(self, parsize):
        Wm = np.identity(parsize)
        # remove last row
        # Wm = Wm[0:-1, :]
        if(self.decouple is not None):
            for axis in self.decouple:
                Wm[:, axis] = 0
                Wm[axis, :] = 0
        return Wm


class SmoothingFirstOrder(BaseRegularization):
    """
    Implement a first order smoothing regularization object

    See Aster, page 98+
    """
    def Wm(self, parsize):
        """
        Return a first order smoothing matrix.

        Parameters
        ----------
        parsize : size of parameter vector m to be regularized

        """
        R = np.zeros((parsize - 1, parsize))
        for i in range(0, parsize - 1):
            R[i, i] = -1
            R[i, i + 1] = 1

        if(self.decouple is not None):
            for axis in self.decouple:
                R[:, axis] = 0
                R[axis, :] = 0

        if(self.weighting_obj is not None):
            # get weighting matrix C
            C = self.weighting_obj.get_C()
            R = C.dot(R)
        return R


class SmoothingSecondOrder(BaseRegularization):
    def Wm(self, parsize):
        """
        Return a second order smoothing matrix.

        Parameters
        ----------
        parsize : size of parameter vector m to be regularized

        """
        first_order = SmoothingFirstOrder()
        R1 = first_order.Wm(parsize)
        if(self.weighting_obj is not None):
            # get weighting matrix C
            C = self.weighting_obj.get_C()
            W = C.dot(R1)
        else:
            W = R1
        R = W.T.dot(W)
        RR = R.T.dot(R)
        return RR

        R = np.zeros((parsize - 2, parsize))
        for i in range(0, parsize - 2):
            R[i, i] = 1
            R[i, i + 1] = -2
            R[i, i + 2] = 1

        if(self.decouple is not None):
            for axis in self.decouple:
                R[:, axis] = 0
                R[axis, :] = 0

        return R
