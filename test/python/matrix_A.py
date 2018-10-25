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
A[0, 0] = np.exp(-(omega - 1)**2)
A[1, 1] = np.exp(-(omega + 1)**2)
A[0, 0] /= np.trapz(A[0, 0], omega)
A[1, 1] /= np.trapz(A[1, 1], omega)

# construct random rotation matrix
phi = 2 * np.pi * np.random.rand()
T = np.array([[np.cos(phi), np.sin(phi)],
              [-np.sin(phi), np.cos(phi)]])
A = np.einsum('ab,bcx,cd', T, A, T.conjugate().transpose())

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
    chi2_elements[i, j] = NormalChi2(K=K, G=G[i, j], err=err[i, j])
    c2_elements[i, j] = chi2_elements[i, j](A[i, j])
    assert chi2_elements[i, j].check_derivatives(
        A[i, j] * 0.1, chi2_elements[i, j].f(A[i, j] * 0.1))

chi2_matrix = MatrixChi2(K=K, G=G, err=err)
c2_matrix = chi2_matrix(A)
calc1 = c2_matrix.f()
calc2 = sum(map(lambda e: e[1].f(), c2_elements.iteritems()))
assert (np.abs(calc1 - calc2)) < 1.e-13

assert c2_matrix.d().shape == (2, 2, 100)
d1 = c2_matrix.d()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(c2_elements[i, j].d() - d1[i, j])) < 1.e-13

assert (c2_matrix.dd().shape == (2, 2, 100, 100))
d2 = c2_matrix.dd()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(c2_elements[i, j].dd() - d2[i, j])) < 1.e-13

bu = lambda x: blowup_matrix(True, x)

assert chi2_matrix.check_derivatives(
    A * 0.1, chi2_matrix.f(A * 0.1), extra_func_dd=bu)

D = FlatDefaultModel(omega=omega)

entropy_elements = dict()
S_elements = dict()
for i, j in product(range(2), range(2)):
    if i == j:
        entropy_elements[i, j] = NormalEntropy(D=D)
    else:
        entropy_elements[i, j] = PlusMinusEntropy(D=D)
    S_elements[i, j] = entropy_elements[i, j](A[i, j])

entropy_matrix = MatrixEntropy(D=MatrixDefaultModel((2, 2), D))
S_matrix = entropy_matrix(A)

calc1 = S_matrix.f()
calc2 = sum(map(lambda e: e[1].f(), S_elements.iteritems()))
assert (np.abs(calc1 - calc2)) < 1.e-13

assert S_matrix.d().shape == (2, 2, 100)
d1 = S_matrix.d()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(S_elements[i, j].d() - d1[i, j])) < 1.e-13

assert (S_matrix.dd().shape == (2, 2, 100, 100))
d2 = S_matrix.dd()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(S_elements[i, j].dd() - d2[i, j])) < 1.e-13

v = np.random.rand(2, 2, len(K.S))

H_of_v_elements = dict()
H_elements = dict()
for i, j in product(range(2), range(2)):
    if i == j:
        H_of_v_elements[i, j] = NormalH_of_v(D=D, K=K)
    else:
        H_of_v_elements[i, j] = PlusMinusH_of_v(D=D, K=K)
    H_elements[i, j] = H_of_v_elements[i, j](v[i, j])

H_of_v_matrix = MatrixH_of_v(D=MatrixDefaultModel((2, 2), D), K=K)
H_matrix = H_of_v_matrix(v.reshape(-1))

assert H_matrix.f().shape == (2, 2, 100)
f1 = H_matrix.f()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(H_elements[i, j].f() - f1[i, j])) < 1.e-13

assert H_matrix.d().shape == (2, 2, 100, 400)
d1 = H_matrix.d()
for i, j in product(range(2), range(2)):
    linin = slice(200 * i + 100 * j, 200 * i + 100 * j + 100)
    assert np.max(np.abs(H_elements[i, j].d() -
                         d1[i, j, :, linin])) < 1.e-13

assert (H_matrix.dd().shape == (2, 2, 100, 400, 400))
d2 = H_matrix.dd()
for i, j in product(range(2), range(2)):
    linin = slice(200 * i + 100 * j, 200 * i + 100 * j + 100)
    assert np.max(np.abs(H_elements[i, j].dd() -
                         d2[i, j, :, linin, linin])) < 1.e-13


H_of_v_matrix_s = _MatrixH_of_v_small(D=MatrixDefaultModel((2, 2), D), K=K)
H_matrix_s = H_of_v_matrix_s(v.reshape(-1))

assert H_matrix_s.f().shape == (2, 2, 100)
f1 = H_matrix_s.f()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(H_elements[i, j].f() - f1[i, j])) < 1.e-13

assert H_matrix_s.d().shape == (2, 2, 100, 100)
d1 = H_matrix_s.d()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(H_elements[i, j].d() - d1[i, j])) < 1.e-13

assert (H_matrix_s.dd().shape == (2, 2, 100, 100, 100))
d2 = H_matrix_s.dd()
for i, j in product(range(2), range(2)):
    assert np.max(np.abs(H_elements[i, j].dd() - d2[i, j])) < 1.e-13


# TODO inv

# TODO check_der

class MatrixSquareH_of_v_slow(MatrixH_of_v):

    def __init__(self, base_H_of_v=None, base_H_of_v_offd=None,
                 D=None, K=None, l_only=False):

        self.l_only = l_only
        super(MatrixSquareH_of_v_slow, self).__init__(base_H_of_v,
                                                      base_H_of_v_offd, D, K)

    @cached
    def f(self, v):
        B = super(MatrixSquareH_of_v_slow, self).f(v)
        H = np.einsum('abi,cbi->aci', B, B.conjugate())
        return H

    @cached
    def d(self, v):
        B = super(MatrixSquareH_of_v_slow, self).f(v)
        dB_dv = super(MatrixSquareH_of_v_slow, self).d(v)
        dH_dv = np.einsum('abij,cbi->acij', dB_dv, B.conjugate())
        dH_dv += dH_dv.conjugate().transpose([1, 0, 2, 3])
        return dH_dv

    @cached
    def dd(self, v):
        B = super(MatrixSquareH_of_v_slow, self).f(v)
        dB_dv = super(MatrixSquareH_of_v_slow, self).d(v)
        ddB_dvdv = super(MatrixSquareH_of_v_slow, self).dd(v)
        ddH_dvdv = np.zeros(B.shape + v.shape + v.shape, dtype=dB_dv.dtype)
        ddH_dvdv = np.einsum('abijk,cbi->acijk', ddB_dvdv, B.conjugate())
        ddH_dvdv += np.einsum('abij,cbik->acijk', dB_dv, dB_dv.conjugate())
        ddH_dvdv += ddH_dvdv.conjugate().transpose([1, 0, 2, 3, 4])
        return ddH_dvdv

    @cached
    def inv(self, H):
        B = np.zeros(H.shape, dtype=H.dtype)
        for i_w in range(H.shape[2]):
            B[:, :, i_w] = np.linalg.cholesky(H[:, :, i_w])
        return super(MatrixSquareH_of_v_slow, self).inv(B)

H_of_v_matrix2 = MatrixSquareH_of_v_slow(D=MatrixDefaultModel((2, 2), D), K=K)
H_matrix2 = H_of_v_matrix2(v.reshape(-1))

assert H_matrix2.f().shape == (2, 2, 100)

assert H_matrix2.d().shape == (2, 2, 100, 400)

assert (H_matrix2.dd().shape == (2, 2, 100, 400, 400))

assert H_of_v_matrix2.check_derivatives(v.reshape(-1))


H_of_v_matrix3 = MatrixSquareH_of_v(D=MatrixDefaultModel((2, 2), D), K=K)
H_matrix3 = H_of_v_matrix3(v.reshape(-1))

assert H_matrix3.f().shape == (2, 2, 100)
assert np.max(np.abs(H_matrix2.f() - H_matrix3.f())) < 1.e-13

assert H_matrix3.d().shape == (2, 2, 100, 400)
assert np.max(np.abs(H_matrix2.d() - H_matrix3.d())) < 1.e-13

assert (H_matrix3.dd().shape == (2, 2, 100, 400, 400))
assert np.max(np.abs(H_matrix2.dd() - H_matrix3.dd())) < 1.e-13

assert H_of_v_matrix3.check_derivatives(v.reshape(-1))

#######################################################################
# MaxEnt with matrix G(tau)
#######################################################################
Q = MaxEntCostFunction(chi2=chi2_matrix,
                       S=entropy_matrix,
                       H_of_v=H_of_v_matrix3,
                       d_dv=True)
Q.set_alpha(1.0)

assert Q.dH(v).shape == (2, 2, 100)
assert Q.d(v).shape == (400,)

assert Q.ddH(v).shape == (2, 2, 100, 100)
assert Q.dd(v).shape == (400, 400)

assert Q.check_derivatives(v.reshape(-1), Q.f(v.reshape(-1)))

Q.d_dv = False
Q.dA_projection = 0

assert Q.check_dd(v.reshape(-1), Q.f(v.reshape(-1)))

Q.dA_projection = 2

logtaker = Logtaker()

minimizer = LevenbergMinimizer(maxiter=10000)

alpha_values = LogAlphaMesh(alpha_max=6000, alpha_min=8, n_points=5)

ml = MaxEntLoop(cost_function=Q, minimizer=minimizer,
                alpha_mesh=alpha_values, logtaker=logtaker)
# the following is just for the test, so that it does not take forever
ml.minimizer.convergence = (MaxDerivativeConvergenceMethod(1.e-4) |
                            RelativeFunctionChangeConvergenceMethod(1.e-6))
result_matrix = ml.run()

if not if_no_triqs():
    from pytriqs.archive import HDFArchive
    with HDFArchive('matrix_A.h5', 'a') as ar:
        ar['result_matrix'] = result_matrix.data
else:
    result_matrix.data

data = result_matrix.data
assert data.A.shape == (5, 2, 2, 100)
assert data.G.shape == (2, 2, 100)
assert data.G_orig.shape == (2, 2, 100)
assert data.G_rec.shape == (5, 2, 2, 100)
assert data.H.shape == (5, 2, 2, 100)
assert data.Q.shape == (5, )
assert data.S.shape == (5, )
assert data.alpha.shape == (5, )
assert data.chi2.shape == (5, )
assert data.complex_elements == False
assert data.data_variable.shape == (100, )
assert data.effective_matrix_structure == (2, 2)
assert data.element_wise == False
assert data.matrix_structure == (2, 2)
assert data.omega.shape == (100, )
assert data.probability.shape == (5, )
assert data.v.shape == (5, 4 * len(Q.chi2.K.S))
assert data.A_out.shape == (2, 2, 100)
