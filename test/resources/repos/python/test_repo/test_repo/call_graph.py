"""Test file for call hierarchy testing."""


def leaf() -> None:
    """Leaf function that is called by mid."""
    pass


def mid() -> None:
    """Mid function that calls leaf and is called by entry_a and entry_b."""
    leaf()


def entry_a() -> None:
    """Entry point A that calls mid."""
    mid()


def entry_b() -> None:
    """Entry point B that calls mid."""
    mid()


def recursive_fn(n: int = 0) -> None:
    """Recursive function that calls itself."""
    if n > 0:
        recursive_fn(n - 1)
