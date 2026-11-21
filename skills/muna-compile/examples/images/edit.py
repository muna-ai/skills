"""Protocol fixture: return the original images unchanged."""

from muna import compile, Sandbox
from muna.beta import Annotations
from PIL import Image
from typing import Annotated


@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox().pip_install("pillow")
)
def edit(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Reference images.")
    ],
    *,
    prompt: Annotated[str, Annotations.ImagePrompt(description="Edit instruction.")] = ""
) -> list[Image.Image]:
    """Return reference images to exercise the image edit contract."""
    return images
