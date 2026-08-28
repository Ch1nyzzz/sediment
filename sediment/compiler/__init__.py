"""Parameter compiler: experience -> bounded update-basis coefficients.

The package is intentionally independent from the former text/lesson compiler.
Heavy training dependencies remain lazy so importing :mod:`sediment` requires
only NumPy and the standard library.
"""

from .basis import ParameterLayout, UpdateBasis
from .data import OracleRecord, ProjectedRecord

__all__ = ["OracleRecord", "ParameterLayout", "ProjectedRecord", "UpdateBasis"]
