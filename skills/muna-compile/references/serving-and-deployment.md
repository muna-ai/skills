# Serve a compiled predictor

Read after the model script is ready, when the user wants to deploy it or call a
hosted prediction. Compilation creates the native predictor; deployment prepares
it to serve on eligible hardware.

## Compile and deploy

```sh
muna compile model.py
muna deploy model.py

# Or deploy an already compiled predictor.
muna deploy @your-org/your-model
```

`muna deploy model.py` compiles the script, then watches deployment preparation.
Use an authenticated CLI environment. Availability also depends on eligible
hardware, pricing and account access. For platform requirements, see
[targets and hardware](targets-and-hardware.md).

## Call the deployed function

The Muna prediction API supports the function's typed inputs directly. For
example, a deployed circle-area function with a `radius` parameter can be called
remotely with the Python SDK:

```python
from muna import Muna
from os import environ

muna = Muna(environ["MUNA_ACCESS_KEY"])
prediction = muna.predictions.create(
    tag="@your-org/area",
    inputs={"radius": 2.0},
    acceleration="remote_auto"
)
print(prediction.results[0])
```

Use the actual deployed tag and signature. The SDK handles value serialization;
see [prediction values](prediction-values.md) for arrays, images and reuse of
encoded results.

## When using an OpenAI or Anthropic client

The function must match the intended adapter's interface. Follow its modality
guide for [chat](llm/generation-and-streaming.md), [embeddings](embeddings/authoring.md),
[images](images/authoring.md) or [audio](audio/authoring.md). The CLI can display a
client example when a supported adapter matches; no match means no such example
is displayed. The deployed function remains callable through the Muna SDK.

The inference base URL is `https://inference.muna.ai/v1`. The current HTTP surface
includes chat, embeddings, image generation/editing, transcription and Anthropic
messages. An SDK feature does not by itself establish a corresponding HTTP route;
see [speech](audio/speech.md) for that particular limitation.

## Readiness and updates

A completed deployment preparation means an eligible node has the required
files on disk. A first request can still incur model initialization. Compile and
deploy again to make a changed Python function live; editing the local file alone
does not update its deployed binary.

For preparation progress, first-request latency, timeouts or publishing an
update, read [deployment readiness](deployment-lifecycle.md). For a failing
request, use [troubleshooting](troubleshooting.md).
