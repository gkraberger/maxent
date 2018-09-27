# TRIQS application maxent
# Copyright (C) 2018 Gernot J. Kraberger
# Copyright (C) 2018 Simons Foundation
# Authors: Gernot J. Kraberger and Manuel Zingl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
This file defines a bunch of functions that represent physical
functions in the MaxEnt formalism, in the matrix version.
"""
from __future__ import absolute_import, print_function

import numpy as np
from .maxent_util import *
from .functions import *


class GenericMatrix():
    r""" A function depending on matrices

    Attributes
    ----------
    sub : list of lists of GenericFunction
        a list of lists (with dimensions corresponding to the matrix
        dimensions) containing function objects
    l_only : bool
        only calculate the function for the lower triangle of the matrix,
        i.e., for matrix elements :math:`M_{ij}` where :math:`j \le i`.
        The default value is ``False``.
    """

    def _forevery(self, fun, H):
        """ Apply a function to all elements of the matrix ``H``

        Parameters
        ----------
        fun : str
            the name of the function; for each function object in ``sub``,
            this function is extracted (e.g., if ``sub`` contains
            :py:class:`.NormalChi2` objects, and ``fun`` is ``'f'``, for
            every element ``NormalChi2.f`` is evaluated.
        H : numpy array
            the first two dimensions are the matrix dimensions; the rest
            are the usual dimensions (e.g., for ``H``, typically the
            ``omega`` dimension)
        """
        l_only = getattr(self, "l_only", False)
        ret = None
        for i in range(H.shape[0]):
            for j in range(i + 1 if l_only else H.shape[1]):
                if ret is None:
                    val = getattr(self.sub[i][j], fun)(H[i, j, :])
                    ret = np.zeros((H.shape[0], H.shape[1]) + val.shape,
                                   dtype=val.dtype)
                    ret[i, j] = val
                else:
                    ret[i, j] = getattr(self.sub[i][j], fun)(H[i, j, :])
        return ret


class MatrixChi2(Chi2, GenericMatrix):
    r""" The misfit for matrix-values H

    In this function, an important (but usually valid) assumption is made
    that reduces computation time in a typical MaxEnt calculation; namely
    that

    .. math::

        \frac{\partial^2 \chi^2}{\partial A^{ab}_i \partial A^{cd}_j} \sim \delta_{ac} \delta_{bd},

    where :math:`a, \dots, d` are matrix indices and :math:`i, j` are
    other indices (typically the omega values).

    Parameters
    ----------
    base_chi2 : Chi2
        the class that shall be used for the diagonal elements (this should
        be the class type rather than a class instance); default is
        :py:class:`.NormalChi2`
    base_chi2_offd : Chi2
        the class that shall be used for the off-diagonal elements (this should
        be the class type rather than a class instance); default is the
        same as ``base_chi2``
    K : :py:class:`.Kernel`
        the kernel to use (only one kernel, this does not have the matrix
        structure of ``G``)
    G : array
        the Green function data (with matrix structure; the first two
        indices are the matrix indices)
    err : array
        the error of the Green function data (must have the same shape
        as G)
    """

    def __init__(self, base_chi2=None, base_chi2_offd=None,
                 K=None, G=None, err=None):
        if base_chi2 is None:
            base_chi2 = NormalChi2
        self.base_chi2 = base_chi2
        if base_chi2_offd is None:
            base_chi2_offd = base_chi2
        self.base_chi2_offd = base_chi2_offd
        Chi2.__init__(self, K=K, G=G, err=err)

    def parameter_change(self):
        # here we initialize the sub elements for the matrix elements
        if self.G is None or self.K is None or self.err is None:
            return
        self.sub = [[self.base_chi2() if i == j else self.base_chi2_offd()
                     for j in range(self.G.shape[1])]
                    for i in range(self.G.shape[0])]
        for i in range(self.G.shape[0]):
            for j in range(self.G.shape[1]):
                self.sub[i][j].K = self.K
                self.sub[i][j].G = self.G[i, j]
                self.sub[i][j].err = self.err[i, j]
                self.sub[i][j].parameter_change()

    @cached
    def f(self, H):
        ret = np.array(self._forevery('f', H))
        return np.sum(ret)

    @cached
    def d(self, H):
        return self._forevery('d', H)

    @cached
    def dd(self, H):
        return self._forevery('dd', H)
