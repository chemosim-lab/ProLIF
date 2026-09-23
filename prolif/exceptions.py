"""
Custom exceptions and error handling --- :mod:`prolif.exceptions`
=================================================================
"""

import warnings
from collections.abc import Callable
from typing import ClassVar, Literal, TypeAlias


class RunRequiredError(RuntimeError):
    """When a function requires the :meth:`~prolif.fingerprint.Fingerprint.run` method
    to have been called prior to execution."""


ErrorBehavior: TypeAlias = Callable[[str], None] | Literal["warn", "raise", "skip"]


class OptionalException(RuntimeError):
    """Used for runtime exceptions that can be converted to a warning or directly
    handled by the user"""

    on_error: ClassVar[ErrorBehavior] = "raise"


class FragmentedResidueError(OptionalException):
    """When a ResidueId maps to multiple Residue objects"""


def trigger(exc: type[OptionalException], msg: str) -> None:
    """Handles the runtime behavior when specific skippable exceptions happen.

    Examples
    --------
    By default, all the exceptions managed by the handler will raise when triggered.
    If you believe the reasons for the error don't apply to your specific case, you can
    convert it to a warning message instead::

        >>> FragmentedResidueError.on_error = "warn"

    You can also completely bypass the exception::

        >>> FragmentedResidueError.on_error = "skip"

    Or redirect them elsewhere::

        >>> FragmentedResidueError.on_error = lambda msg: logger.error(msg)


    .. versionadded:: 2.2.2

    """
    match on_error := exc.on_error:
        case "raise":
            raise exc(
                f"{msg}\nAlthough not recommended, you can also ignore this error by "
                f"setting `prolif.exceptions.{exc.__name__}.on_error = 'warn'."
            )
        case "warn":
            warnings.warn(msg, stacklevel=2)
        case on_error if callable(on_error):
            on_error(msg)
        case _:
            pass
