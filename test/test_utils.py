import numpy as np
import pytest

from lib.utils import make_hermitian, qubitize_price_system


def test_make_hermitian():
    A = np.random.rand(3, 4) + 1j * np.random.rand(3, 4)
    A_herm = make_hermitian(A)
    assert np.allclose(A_herm, A_herm.T.conj())


def test_qubitize_price_system():
    A = np.random.rand(3, 4)
    x = np.random.rand(4).reshape(-1, 1)
    b = A @ x

    A_ext, b_ext = qubitize_price_system(A, b)
    x_ext = np.concatenate((np.zeros((3, 1)), x, np.zeros((1, 1))), axis=0)

    assert np.allclose(A_ext @ x_ext, b_ext)

    n = int(np.log2(A_ext.shape[0]))
    assert A_ext.shape[0] == 2**n
