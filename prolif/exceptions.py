"""
Custom exceptions and error handling --- :mod:`prolif.exceptions`
=================================================================
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias


class RunRequiredError(RuntimeError):
    """When a function requires the :meth:`~prolif.fingerprint.Fingerprint.run` method
    to have been called prior to execution."""


ErrorBehavior: TypeAlias = Callable[[str], None] | Literal["warn", "raise"]


class OptionalException(RuntimeError):
    """Used for runtime exceptions that can be converted to a warning or directly
    handled by the user"""

    on_error: ErrorBehavior = "raise"

    def trigger(self, msg: str) -> None:
        if self.on_error == "raise":
            raise type(self)(msg)
        if self.on_error == "warn":
            warnings.warn(msg, stacklevel=2)
        else:
            self.on_error(msg)


class FragmentedResidueError(OptionalException):
    """When a ResidueId maps to multiple Residue objects"""


@(lambda cls: cls())
@dataclass(frozen=True)
class error_handler:
    """Handles the runtime behavior when specific skippable exceptions happen.

    Examples
    --------
    By default, all the exceptions managed by the handler will raise when triggered.
    If you believe the reasons for the error don't apply to your specific case, you can
    convert it to a warning message instead::

        >>> error_handler.fragmented_residue.on_error = "warn"

    You can also completely bypass the errors::    

        >>> error_handler.fragmented_residue.on_error = lambda msg: None

    Or redirect them elsewhere::

        >>> error_handler.fragmented_residue.on_error = lambda msg: logger.debug(msg)


    ..versionadded: 2.2.2

    """

    fragmented_residue: FragmentedResidueError = field(
        default_factory=FragmentedResidueError
    )
