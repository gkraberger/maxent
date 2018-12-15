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


from __future__ import absolute_import, print_function


import matplotlib
matplotlib.use('tkagg')

import numpy as np
from triqs_maxent import *
from triqs_maxent.matrix_functions import _MatrixH_of_v_small
from triqs_maxent.triqs_support import *
from itertools import product

import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno,
                        file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename,
                                     lineno, line))
warnings.showwarning = warn_with_traceback

# to make it reproducible
np.random.seed(658436166)

beta = 40
tau = np.linspace(0, beta, 100)
omega = HyperbolicOmegaMesh(omega_min=-10, omega_max=10, n_points=100)
K = TauKernel(tau=tau, omega=omega, beta=beta)
# here we construct the G(tau)
A = np.zeros((2, 2, len(omega)))
A[0, 0] = np.exp(-(omega - 1)**2 / 0.4**2)
A[1, 1] = np.exp(-(omega + 1)**2 / 0.4**2)
A[0, 0] /= np.trapz(A[0, 0], omega)
A[1, 1] /= np.trapz(A[1, 1], omega)

# construct random rotation matrix
H = np.random.rand(2, 2) + 1.0j * np.random.rand(2, 2)
H = H + np.conjugate(np.transpose(H))
e, v = np.linalg.eigh(H)
T = np.dot(v, np.dot(np.diag(-np.exp(1.0j * e)), v.conjugate().transpose()))
A = np.einsum('ab,bcx,cd', T, A, T.conjugate().transpose())

A_view = A.view(float).reshape(A.shape + (2,))

for i in range(2):
    for j in range(2):
        assert np.abs(np.trapz(A[i, j], omega) -
                      (1 if i == j else 0)) < 1.e-14, "A not normalized"

G = np.einsum('ij,abj', K.K_delta, A)
G += 1.e-4 * np.random.randn(*G.shape)
err = 1.e-4 * np.ones(G.shape)

chi2_elements = dict()
c2_elements = dict()
for i, j in product(range(2), range(2)):
    chi2_elements[i, j] = ComplexChi2(K=K, G=G[i, j], err=err[i, j])
    c2_elements[i, j] = chi2_elements[i, j](A_view[i, j])
    assert chi2_elements[i, j].check_derivatives(
        A_view[i, j] * 0.1,
        chi2_elements[i, j].f(A_view[i, j] * 0.1),
        prec=1.e-6)

chi2_matrix = MatrixChi2(base_chi2=ComplexChi2, K=K, G=G, err=err)
c2_matrix = chi2_matrix(A_view)
calc1 = c2_matrix.f()
calc2 = sum(map(lambda e: e[1].f(), c2_elements.iteritems()))
assert (np.abs(calc1 - calc2)) < 1.e-13

assert c2_matrix.d().shape == (2, 2, 100, 2)
d1 = c2_matrix.d()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(c2_elements[i, j].d() - d1[i, j])) < 1.e-13

assert (c2_matrix.dd().shape == (2, 2, 100, 2, 100, 2))
d2 = c2_matrix.dd()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(c2_elements[i, j].dd() - d2[i, j])) < 1.e-13

bu = lambda x: blowup_second_derivative(
    x, chi2_matrix.input_shape, chi2_matrix.output_shape)

assert chi2_matrix.check_derivatives(
    A_view * 0.1, chi2_matrix.f(A_view * 0.1), extra_func_dd=bu)

D = FlatDefaultModel(omega=omega)

entropy_elements = dict()
S_elements = dict()
for i, j in product(range(2), range(2)):
    if i == j:
        entropy_elements[i, j] = ComplexNormalEntropy(D=D)
    else:
        entropy_elements[i, j] = ComplexPlusMinusEntropy(D=D)
    S_elements[i, j] = entropy_elements[i, j](
        A[i, j].view(float).reshape(A[i, j].shape + (2,)))

entropy_matrix = MatrixEntropy(D=MatrixDefaultModel((2, 2), D),
                               base_entropy=ComplexNormalEntropy,
                               base_entropy_offd=ComplexPlusMinusEntropy)
S_matrix = entropy_matrix(A.view(float).reshape(A.shape + (2,)))

# TODO: in the complex case, the diag and offdiag has to be treated differently
calc1 = S_matrix.f()
calc2 = sum(map(lambda e: e[1].f(), S_elements.iteritems()))
assert (np.abs(calc1 - calc2)) < 1.e-13

assert S_matrix.d().shape == (2, 2, 100, 2)
d1 = S_matrix.d()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(S_elements[i, j].d() - d1[i, j])) < 1.e-13

assert (S_matrix.dd().shape == (2, 2, 100, 2, 100, 2))
d2 = S_matrix.dd()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(S_elements[i, j].dd() - d2[i, j])) < 1.e-13
