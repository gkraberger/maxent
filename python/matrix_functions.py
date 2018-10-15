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
from itertools import product


def blowup_matrix(scalar, matrix_dims, fe):
    if scalar:
        ret = np.zeros(fe.shape[:3] + matrix_dims + (fe.shape[-1],),
                       dtype=fe.dtype)
        for i, j in product(*list(map(range, matrix_dims))):
            ret[i, j, :, i, j, :] = fe[i, j]
    else:
        ret = np.zeros(fe.shape[:3] + matrix_dims + (fe.shape[-1],) +
                       matrix_dims + (fe.shape[-1],), dtype=fe.dtype)
        for i, j in product(*list(map(range, matrix_dims))):
            ret[i, j, :, i, j, :, i, j, :] = fe[i, j]
    return ret


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

    def __init__(self):
        # for sphinx
        pass

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
    r""" The misfit for matrix-valued H

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


class MatrixEntropy(Entropy, GenericMatrix):
    r""" The entropy for matrix-valued H

    In this function, an important (but usually valid) assumption is made
    that reduces computation time in a typical MaxEnt calculation; namely
    that

    .. math::

        \frac{\partial^2 S}{\partial A^{ab}_i \partial A^{cd}_j} \sim \delta_{ac} \delta_{bd},

    where :math:`a, \dots, d` are matrix indices and :math:`i, j` are
    other indices (typically the omega values).

    Parameters
    ----------
    base_entropy : Entropy
        the class that shall be used for the diagonal elements (this should
        be the class type rather than a class instance); default is
        :py:class:`.NormalEntropy`
    base_entropy_offd : Entropy
        the class that shall be used for the off-diagonal elements (this should
        be the class type rather than a class instance); default is
        :py:class:`.PlusMinusEntropy`
    D : :py:class:`.MatrixDefaultModel`
        the default model to use
    """

    def __init__(self, base_entropy=None, base_entropy_offd=None, D=None):
        if base_entropy is None:
            base_entropy = NormalEntropy
        self.base_entropy = base_entropy
        if base_entropy_offd is None:
            base_entropy_offd = PlusMinusEntropy
        self.base_entropy_offd = base_entropy_offd
        Entropy.__init__(self, D)

    def parameter_change(self):
        if self.D is None:
            return
        self.sub = [[self.base_entropy() if i == j else self.base_entropy_offd()
                     for j in range(self.D.matrix_dims[1])]
                    for i in range(self.D.matrix_dims[0])]
        for i in range(self.D.matrix_dims[0]):
            for j in range(self.D.matrix_dims[1]):
                self.sub[i][j].D = self.D[i, j]
                self.sub[i][j].parameter_change()

    @cached
    def f(self, H):
        ret = self._forevery('f', H)
        return np.sum(ret)

    @cached
    def d(self, H):
        return self._forevery('d', H)

    @cached
    def dd(self, H):
        return self._forevery('dd', H)


class MatrixH_of_v_small(GenericH_of_v, GenericMatrix):
    r""" The mapping H(v) for matrix-valued H

    In this function, an important (but usually valid) assumption is made
    that reduces computation time in a typical calculation; namely
    that

    .. math::

        \frac{\partial^2 H^{ab}}{\partial v^{cd}_i \partial v^{ef}_j} \sim \delta_{ac} \delta_{ae} \delta_{bd} \delta_{bf},

    where :math:`a, \dots, f` are matrix indices and :math:`i, j` are
    other indices (typically the values in singular space).

    Parameters
    ----------
    base_H_of_v: GenericH_of_v
        the class that shall be used for the diagonal elements (this should
        be the class type rather than a class instance); default is
        :py:class:`.NormalH_of_v`
    base_H_of_v_offd : GenericH_of_v
        the class that shall be used for the off-diagonal elements (this should
        be the class type rather than a class instance); default is
        :py:class:`.PlusMinusH_of_v`
    D : :py:class:`.MatrixDefaultModel`
        the default model to use
    K : :py:class:`.Kernel`
        the kernel to use (only one kernel, this does not have the matrix
        structure of ``G``)
    """

    def __init__(self, base_H_of_v=None, base_H_of_v_offd=None,
                 D=None, K=None):
        if base_H_of_v is None:
            base_H_of_v = NormalH_of_v
        self.base_H_of_v = base_H_of_v
        if base_H_of_v_offd is None:
            base_H_of_v_offd = PlusMinusH_of_v
        self.base_H_of_v_offd = base_H_of_v_offd
        GenericH_of_v.__init__(self, D=D, K=K)

    def parameter_change(self):
        if self.D is None or self.K is None:
            return
        self.sub = [[self.base_H_of_v() if i == j else self.base_H_of_v_offd()
                     for j in range(self.D.matrix_dims[1])]
                    for i in range(self.D.matrix_dims[0])]
        for i in range(self.D.matrix_dims[0]):
            for j in range(self.D.matrix_dims[1]):
                self.sub[i][j].D = self.D[i, j]
                self.sub[i][j].K = self.K
                self.sub[i][j].parameter_change()

    @cached
    def f(self, v):
        return self._forevery('f', v.reshape(self.D.matrix_dims + (-1,)))

    @cached
    def d(self, v):
        fe = self._forevery('d', v.reshape(self.D.matrix_dims + (-1,)))
        return fe

    @cached
    def dd(self, v):
        fe = self._forevery('dd', v.reshape(self.D.matrix_dims + (-1,)))
        return fe

    @cached
    def inv(self, H):
        return self._forevery('inv', H).reshape(-1)


class MatrixH_of_v(MatrixH_of_v_small):
    r""" The mapping H(v) for matrix-valued H

    Parameters
    ----------
    base_H_of_v: GenericH_of_v
        the class that shall be used for the diagonal elements (this should
        be the class type rather than a class instance); default is
        :py:class:`.NormalH_of_v`
    base_H_of_v_offd : GenericH_of_v
        the class that shall be used for the off-diagonal elements (this should
        be the class type rather than a class instance); default is
        :py:class:`.PlusMinusH_of_v`
    D : :py:class:`.MatrixDefaultModel`
        the default model to use
    K : :py:class:`.Kernel`
        the kernel to use (only one kernel, this does not have the matrix
        structure of ``G``)
    """

    def __init__(self, base_H_of_v=None, base_H_of_v_offd=None,
                 D=None, K=None):
        super(MatrixH_of_v, self).__init__(base_H_of_v,
                                           base_H_of_v_offd, D, K)

    @cached
    def f(self, v):
        return super(MatrixH_of_v, self).f(v)

    @cached
    def d(self, v):
        fe = super(MatrixH_of_v, self).d(v)
        ret = np.zeros(fe.shape[:3] + self.D.matrix_dims + (fe.shape[-1],),
                       dtype=fe.dtype)
        for i, j in product(*list(map(range, self.D.matrix_dims))):
            ret[i, j, :, i, j, :] = fe[i, j]
        return ret.reshape(fe.shape[:3] + (-1,))

    @cached
    def dd(self, v):
        fe = super(MatrixH_of_v, self).dd(v)
        ret = blowup_matrix(False, self.D.matrix_dims, fe)
        for i, j in product(*list(map(range, self.D.matrix_dims))):
            ret[i, j, :, i, j, :, i, j, :] = fe[i, j]
        pind = self.D.matrix_dims[0] * self.D.matrix_dims[1] * fe.shape[-1]
        return ret.reshape(fe.shape[:3] + (pind, pind))

    @cached
    def inv(self, H):
        return super(MatrixH_of_v, self).inv(H)


class MatrixSquareH_of_v(MatrixH_of_v_small):

    def __init__(self, base_H_of_v=None, base_H_of_v_offd=None,
                 D=None, K=None, l_only=False):

        self.l_only = l_only
        super(MatrixSquareH_of_v, self).__init__(base_H_of_v,
                                                 base_H_of_v_offd, D, K)

    @cached
    def f(self, v):
        B = super(MatrixSquareH_of_v, self).f(v)
        H = np.einsum('abi,cbi->aci', B, B.conjugate())
        return H

    @cached
    def d(self, v):
        B = super(MatrixSquareH_of_v, self).f(v)
        dB_dv = super(MatrixSquareH_of_v, self).d(v)
        dH_dv = np.zeros(
            B.shape + B.shape[:2] + (dB_dv.shape[-1],), dtype=dB_dv.dtype)
        for m in range(B.shape[0]):
            h = np.einsum('bij,cbi->cibj', dB_dv[m], B.conjugate())
            dH_dv[m, :, :, m, :, :] += h
            dH_dv[:, m, :, m, :, :] += h.conjugate()
        return dH_dv.reshape(B.shape + (-1,))

    @cached
    def dd(self, v):
        B = super(MatrixSquareH_of_v, self).f(v)
        dB_dv = super(MatrixSquareH_of_v, self).d(v)
        ddB_dvdv = super(MatrixSquareH_of_v, self).dd(v)
        ddH_dvdv = np.zeros(B.shape + B.shape[:2] + (dB_dv.shape[-1],) + B.shape[
                            :2] + (dB_dv.shape[-1],), dtype=dB_dv.dtype)
        for a in range(B.shape[0]):
            for d in range(B.shape[1]):
                h = np.einsum('kjl,bk->bkjl',
                              ddB_dvdv[a, d, :, :, :], B[:, d, :].conjugate())
                ddH_dvdv[a, :, :, a, d, :, a, d, :] += h
                for b in range(B.shape[1]):
                    k = np.einsum('kj,kl->kjl',
                                  dB_dv[a, d, :, :],
                                  dB_dv[b, d, :, :].conjugate())
                    ddH_dvdv[a, b, :, a, d, :, b, d, :] += k
                    ddH_dvdv[a, b, :, b, d, :, a, d, :] += \
                        k.transpose([0, 2, 1])
                b = a
                ddH_dvdv[:, b, :, b, d, :, b, d, :] += h.conjugate()
        return ddH_dvdv.reshape(B.shape + v.shape + v.shape)

    @cached
    def inv(self, H):
        B = np.zeros(H.shape, dtype=H.dtype)
        for i_w in range(H.shape[2]):
            B[:, :, i_w] = np.linalg.cholesky(H[:, :, i_w])
        return super(MatrixSquareH_of_v, self).inv(B)
