# Speech synthesis

Read for the SDK's `openai.audio.speech.create` adapter.

## Declare the interface

The current adapter requires two distinguishable strings: a text input and a
required voice input annotated with `Annotations.AudioVoice`. Optional speed
uses `Annotations.AudioSpeed`. Return a float32 array annotated with
`Parameter.Audio` and a positive sample rate.

The adapter encodes the PCM into supported response formats. The predictor
returns actual samples rather than a pre-framed HTTP response. Do not return
compressed audio bytes while declaring a float32 PCM output. Preserve whether
the model returns mono or frames-by-channels audio and validate sample rate.

The [silence fixture](../../examples/audio/speech.py) tests this contract only.
It is not a TTS model.

## When serving over HTTP

**Serving limit:** an OpenAI-compatible HTTP speech endpoint is not currently
available. Use the Muna SDK prediction interface for a deployed speech function;
an SDK adapter match alone does not make `openai.audio.speech.create` available
through the hosted HTTP API.
