# Batching and concurrent requests

Read when the predictor needs to combine requests or allow concurrent calls.
Several items inside one request and several independent requests are different
dimensions: an embedding call with ten texts is not ten concurrent callers.

## Start with the threading contract

Compiled predictors are not assumed to be thread safe. By default, all
predictions for a given predictor must be submitted on one thread, so function
invocations do not overlap. Static and dynamic batching preserve this serialized
entry while combining multiple requests into each invocation.

Continuous mode relaxes that requirement: multiple threads can submit requests
and execute separate invocations concurrently. The compiled predictor itself
must therefore be thread safe, including its shared state and model calls.
Most ordinary compiled operations can run concurrently, but shared mutable state
still needs coordination.

For model inference, only NanoSGL engines currently provide this thread-safe
entry contract. The compiled TensorRT, ONNX Runtime, Core ML and other model
integrations do not. Use sequential, static or dynamic entry for those backends;
a batching annotation cannot make their model calls thread safe.

## Choose a batching mode

`BatchConfig` is parameter metadata that tells Muna how requests enter a
prediction function:

| Mode | Request handling | Threading |
| --- | --- | --- |
| Sequential (default) | One request per function invocation, without merging requests. Omit `BatchConfig`. | Submit predictions on one thread; invocations are serialized. |
| `static` | Buffer requests until the configured `capacity` forms a fixed-size batch, then invoke the function with merged list-like inputs. | Batch invocations remain serialized on one thread. |
| `dynamic` | Form variable-size batches from queued requests, up to `capacity`, and invoke the function with merged list-like inputs. | Batch invocations remain serialized on one thread. |
| `continuous` | Keep a separate function invocation for each request; a shared engine can combine their work internally. | Allow concurrent invocations from multiple threads; the compiled code must be thread safe. |

Sequential is the behavior when no batch configuration is present, not a
`BatchConfig(mode="sequential")` value. Static and dynamic modes require
`capacity` and list-like annotated parameters. Continuous mode can annotate
another supported parameter type because callers do not share one merged
argument. Do not add a dynamic-batching layer solely because a model accepts a
list within a single request.

The annotation is attached through a parameter helper. For example, a
thread-safe predictor's list parameter can declare continuous entry with:

```python
from muna import BatchConfig, Parameter
from typing import Annotated

# Use as a parameter in the prediction function.
texts: Annotated[list[str], Parameter.Generic(
    description="Texts in this request.",
    batch=BatchConfig(mode="continuous")
)]
```

This permits concurrent entry; it does not implement a model scheduler or
promise unlimited concurrency. For NanoSGL inference, keep the engine/model
instance shared across calls and use the limits supported by that integration.

## Follow the model's execution pattern

- NanoSGL LLM requests need independent generation state and cleanup while
  sharing the manager: [LLM batching and KV cache](llm/batching-and-kv-cache.md).
- NanoSGL embedding engines can combine compatible queued calls internally:
  [embedding authoring](embeddings/authoring.md).
- For other model backends, keep serialized entry and use static/dynamic batching
  only when the function and model support the merged input shape. Select the
  execution path through [inference metadata](inference-metadata/index.md).
