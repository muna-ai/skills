"""Protocol fixture: silence, not a text-to-speech model or HTTP endpoint."""

from muna import compile, Parameter, Sandbox
from muna.beta import Annotations
from numpy import float32, ndarray, zeros
from typing import Annotated


@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=Sandbox().pip_install("numpy")
)
def speech(
    text: str,
    voice: Annotated[str, Annotations.AudioVoice(description="Voice identifier.")]
) -> Annotated[
    ndarray,
    Parameter.Audio(description="One second of silent mono PCM.", sample_rate=24000)
]:
    """Create silence to exercise the speech SDK adapter contract."""
    return zeros((24000,), dtype=float32)
