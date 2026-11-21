from muna import compile


@compile(targets=["x86_64-unknown-linux-gnu"])
def area(radius: float) -> float:
    """Calculate the area of a circle."""
    if radius < 0.0:
        raise ValueError("Radius must be nonnegative")
    return 3.141592653589793 * radius ** 2
