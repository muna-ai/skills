# Speech transcription

Read for `openai.audio.transcriptions.create`.

## Declare the interface

Use a required float32 array or array-list annotated with
`Parameter.Audio(sample_rate=<positive-rate>)`, which marks the input as audio
and requires a positive sample rate. Return text, or a supported
string-list result. Optional controls use `TranscriptionLanguage`,
`TranscriptionPrompt` and `Temperature`; unsupported required controls reject
adapter matching.

## Report processed audio usage

Report the audio actually processed. For duration usage, return a separate
Pydantic `Usage` output with `type="duration"` and `seconds`; follow the
[example](../../examples/audio/transcription.py) for the required type and fields.
Token-based usage instead reports the model's token counts.

For the current Python SDK, use the example's Pydantic `Usage` model for this
output. `Transcription.DurationUsage` and `Transcription.TokenUsage` do not work
as direct substitutes. Without a usage output, the response can fall back to the input
audio duration. Prediction latency is never audio duration.

Keep the content consumed by the model and the charged duration consistent when
cropping, chunking or padding. If only part of an input was processed, return
usage that expresses that fact instead of relying on input-length fallback.

## Streaming and model limits

The Python SDK currently rejects streaming transcription. Do not copy chat's
`Iterator[ChatCompletionChunk]` pattern onto this endpoint. Check the server's
current request/format options before promising parity with every OpenAI format.

NanoSGL Whisper does not support disaggregated prefill/decode. Follow its
processor recipe and verify sample rate, language/task tokens and long-audio
behavior against the reference model.

See the [duration protocol fixture](../../examples/audio/transcription.py).
