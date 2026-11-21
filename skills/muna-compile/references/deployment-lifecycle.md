# Deployment readiness, cold starts and updates

Read when deployment progress stalls, a first request is slow, or you want to
publish a changed predictor. For commands and client examples, start with
[serving and deployment](serving-and-deployment.md).

## What deployment readiness means

`muna deploy` waits until the files for a compatible compiled model are staged
on eligible hardware. It does not run a test prediction or guarantee the model
is already initialized. The first request can still incur a cold start while
weights are loaded and the model is initialized.

If deployment is waiting for a serving node, check that the model's
[inference metadata](inference-metadata/index.md) and
[build target](targets-and-hardware.md) match available hardware. During resource
staging, large checkpoints can take time even when the predictor code is small.
Keep the CLI's last progress message and any error when reporting a stalled
deployment to Muna support.

## Understand request latency and timeouts

Separate input upload, queueing, cold-start initialization, inference and output
download when investigating a slow request. For streaming generation, measure
time to first token separately from the time to finish the response.

An omitted or null deployment queue-time limit uses the platform default; an
explicit limit overrides it. Increasing the limit lets a request wait longer,
but does not make model initialization faster. Keep queue-time settings in the
deployment configuration rather than the prediction function.

## Publish a changed function

Run `muna deploy model.py` to compile and deploy the updated script. Editing the
local file or compiling it alone does not replace the deployed model.

A new version can change the function's parameters, defaults or outputs. Update
client calls to match the deployed signature and test the new version with a
representative request. A successful deployment is separate from the
[correctness checks](validation.md) for model inference.
