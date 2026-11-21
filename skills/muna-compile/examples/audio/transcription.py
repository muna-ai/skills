"""Protocol fixture: sample-count text and measured duration, not recognition."""

from muna import compile, Parameter, Sandbox
from numpy import ndarray
from pydantic import BaseModel
from typing import Annotated, Literal


# Return a separate Usage output to report processed audio seconds.
class Usage(BaseModel):
    type: Literal["duration"] = "duration"
    seconds: float


@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox().pip_install("numpy", "pydantic")
)
def transcription(
    audio: Annotated[
        ndarray,
        Parameter.Audio(description="Mono input PCM samples.", sample_rate=16000)
    ]
) -> tuple[str, Usage]:
    """Describe audio samples to exercise the transcription contract."""
    frames = len(audio)
    return f"Received {frames} audio frames.", Usage(seconds=frames / 16000.0)
