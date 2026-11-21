# Audio workloads

Read for speech recognition or synthesis. Follow the guide for the direction of
conversion; each introduces its own input/output annotations.

- Audio to text: [transcription](transcription.md).
- Text to audio: [speech](speech.md).
- Audio attachments inside a conversation: [multimodal LLM inputs](../llm/multimodal-inputs.md).

## Signature examples

These examples show the audio interfaces, with model bodies and build
configuration omitted. Replace `...` before compiling. A different supported
signature can still be called through the Muna prediction API.

### Transcription

```python
from muna import compile, Parameter
from numpy import ndarray
from pydantic import BaseModel
from typing import Annotated, Literal

class Usage(BaseModel):
    type: Literal["duration"] = "duration"
    seconds: float

@compile()
def transcribe(
    audio: Annotated[ndarray, Parameter.Audio(
        description="Mono PCM samples.",
        sample_rate=16000
    )]
) -> tuple[str, Usage]:
    """Transcribe speech and report processed audio seconds."""
    ...
```

For Muna to recognize this transcription interface:

1. Declare a required float32 audio-array input annotated with `Parameter.Audio`
   and a positive `sample_rate`. The array-list form is also supported.
2. Return an unambiguous text output; the example uses `str`.
3. If reporting duration usage, return the example's Pydantic `Usage` as a
   separate tuple output. Set `seconds` to the audio actually processed,
   not prediction latency. Usage is optional in the signature.
4. Give additional inputs defaults unless the transcription interface supplies
   them through a supported annotation.

See [transcription](transcription.md) for usage types, controls and streaming
limitations, and the [signature-only predictor](../../examples/audio/transcription.py)
for an executable example without speech recognition.

### Speech synthesis

```python
from muna import compile, Parameter
from muna.beta import Annotations
from numpy import ndarray
from typing import Annotated

@compile()
def speech(
    text: str,
    voice: Annotated[
        str,
        Annotations.AudioVoice(description="Voice identifier.")
    ]
) -> Annotated[ndarray, Parameter.Audio(
    description="Mono PCM output.",
    sample_rate=24000
)]:
    """Synthesize speech with a selected voice."""
    ...
```

For Muna to recognize this speech interface:

1. Declare one required string for the input text.
2. Declare a separate required string annotated with `Annotations.AudioVoice`.
   Keep the text and voice inputs unambiguous.
3. Return one float32 PCM array annotated with `Parameter.Audio` and a positive
   `sample_rate`, shaped `[frames]` or `[frames, channels]`.
4. Give additional inputs defaults unless the speech interface supplies them
   through a supported annotation.

This signature supports the SDK speech adapter. The OpenAI-compatible HTTP
speech endpoint is currently unavailable; see [speech](speech.md) for how to
call the predictor and the [silence example](../../examples/audio/speech.py).

## Choose the execution path

Select [inference metadata](../inference-metadata/index.md) for the model object
and intended hardware. Transcription and speech annotations describe the audio
contract; they do not select NanoSGL or require a GPU. Use a compatible graph
backend or the documented native integration for the particular architecture.

## Sample representation

Model functions normally work with decoded audio samples rather than uploaded
file bytes. Use float32 PCM arrays: frames, or frames by channels as required by
the model. The audio annotation records the sample rate. Distinguish sample
frames from channel count, and do not assume metadata resamples arbitrary data
inside the function.

Client adapters prepare supported files or sample arrays for the declared rate.
If the task instead uses raw prediction payloads or forwards encoded audio, read
[prediction values](../prediction-values.md) for the serialization contract.

Whisper has a specific [NanoSGL model path](../inference-metadata/sglang.md) and
processor requirements. Other architectures need an execution backend that supports
their computation; a Python sample for an audio model alone is not proof of a
compatible hosted HTTP route.
