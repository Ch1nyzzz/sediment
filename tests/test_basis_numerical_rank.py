import numpy as np
import pytest

from sediment.compiler.basis import UpdateBasis


def test_basis_nonzero_extremely_wide_state_is_not_misclassified_as_zero():
    width = int(1.0 / np.finfo(np.float32).eps) + 1
    update = np.zeros(width, dtype=np.float32)
    update[0] = 1.0

    basis = UpdateBasis.fit([{"very_wide": update}], rank=1)

    assert basis.rank == 1
    assert basis.fit_samples == 1
    assert basis.singular_values[0] == pytest.approx(1.0)
