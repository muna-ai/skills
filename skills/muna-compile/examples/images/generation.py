"""Protocol fixture: a blank image, without a diffusion model."""

from muna import compile, Sandbox
from muna.beta import Annotations
from PIL import Image
from typing import Annotated


@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox().pip_install("pillow")
)
def generation(
    prompt: Annotated[str, Annotations.ImagePrompt(description="Image prompt.")]
) -> list[Image.Image]:
    """Create a blank image to exercise the generation contract."""
    return [Image.new("RGB", (32, 32))]
