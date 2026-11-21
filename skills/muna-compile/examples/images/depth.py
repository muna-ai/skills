"""Protocol fixture: synthetic 16-by-16 maps, without a depth model."""

from muna import compile, Parameter, Sandbox
from muna.beta import Annotations
from numpy import float32, ndarray, zeros
from PIL import Image
from typing import Annotated


@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox().pip_install("numpy", "pillow")
)
def depth(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Reference images.")
    ],
    *,
    prompt: Annotated[str, Annotations.ImagePrompt(description="Depth instruction.")] = ""
) -> Annotated[
    list[ndarray],
    Parameter.DepthMap(description="Synthetic 16-by-16 zero depth maps; no physical units.")
]:
    """Create zero maps to exercise the depth output contract."""
    return [zeros((16, 16), dtype=float32) for image in images]
