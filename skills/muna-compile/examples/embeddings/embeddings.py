"""Protocol fixture: synthetic vectors, without an embedding model."""

from muna import compile, Parameter, Sandbox
from numpy import float32, ndarray, zeros
from typing import Annotated


@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox().pip_install("numpy")
)
def embeddings(texts: list[str]) -> Annotated[
    ndarray,
    Parameter.Embedding(description="Synthetic float32 embedding matrix.")
]:
    """Create zero vectors to exercise the embedding contract."""
    return zeros((len(texts), 8), dtype=float32)
