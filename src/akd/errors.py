class AkdError(RuntimeError):
    """Base error for akd failures."""


class ContractError(AkdError):
    """A correctness contract was violated (devices, capture mode, output declarations)."""


class AkdWarning(UserWarning):
    """Non-fatal akd condition the user should know about."""
