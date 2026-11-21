"""Protocol fixture: an empty PLY, without a reconstruction model."""

from muna import compile, Parameter, Sandbox
from muna.beta import Annotations
from PIL import Image
from typing import Annotated


@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox().pip_install("pillow")
)
def model3d(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Reference images.")
    ],
    *,
    prompt: Annotated[str, Annotations.ImagePrompt(description="3D instruction.")] = ""
) -> Annotated[
    bytes,
    Parameter.Model3D(description="Empty ASCII PLY point cloud, for protocol testing.")
]:
    """Return an empty PLY to exercise the 3D asset contract."""
    return b"ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
