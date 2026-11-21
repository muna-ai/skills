# Build targets and hardware

Read when deciding where a predictor can run. **The inference metadata used to
compile each model is the source of truth for its execution requirements.**
Start with that metadata, rather than assuming a model needs a particular GPU
because of its task, size, Python library or the hardware available on a node.

## Establish the model's execution requirements

Identify the model objects used by the function and their entries in
`@compile(metadata=[...])`. The selected metadata type chooses an inference path;
its backend-specific settings determine how that path uses hardware. Read the
[inference metadata guide](inference-metadata/index.md) for the selected type.

For example, a PyTorch module exported to ONNX Runtime has provider settings,
while the same kind of input module lowered through NanoSGL has an explicit
compute architecture. A Core ML model has a different execution policy again.
These declarations are not interchangeable. Even two text-generation predictors
can have different hardware requirements because they use different metadata.

A plain Python/numerical function can have no inference metadata at all. Do not
add GPU requirements to such a function merely because it will be hosted on a
GPU node. Conversely, removing metadata from a model that needs a supported
backend does not turn it into a CPU implementation.

Some backends can select among CPU and accelerators at runtime. Distinguish
an allowed accelerator from a mandatory device requirement; follow that
metadata type's documented behavior. Do not infer a required GPU from build-time
tensor placement or generalize one backend's architecture rules to another.

## Select a platform separately

The decorator's `targets` setting selects operating-system/CPU-ABI combinations
to compile for. It does not itself select CPU versus GPU inference. With
`targets=None`, the compiler selects defaults subject to backend support.
Where a metadata type also exposes `targets`, that setting scopes the model's
configuration to supported platforms; it does not make the backend portable.

For hosted deployment, an explicit target list must include the exact triple
`x86_64-unknown-linux-gnu`. A minimal function can use:

```python
from muna import compile

@compile(targets=["x86_64-unknown-linux-gnu"])
def area(radius: float) -> float:
    """Calculate the area of a circle."""
    return 3.141592653589793 * radius ** 2
```

This selects Linux x64 and adds no GPU requirement. For a model-backed function,
choose compatible inference metadata as well. Changing `targets` alone cannot
make a CUDA implementation run on iOS or WebAssembly.

## Validate the resulting combination

Check the metadata, model/checkpoint, precision and platform together. A field
accepted by the SDK does not prove the compiler has a working path for that
combination; the metadata-specific guides describe current restrictions.
If the predictor uses several model objects, account for every object's
execution path rather than copying the hardware settings from just one.

Use the metadata's architecture and multi-device settings rather than physical
GPU IDs or `CUDA_VISIBLE_DEVICES` in a hosted model script.

Follow [validation](validation.md) to check compilation and actual execution.
Read [deployment readiness](deployment-lifecycle.md) for hosted deployment and
first-request behavior, or [memory and colocation](memory-and-colocation.md)
when sizing NanoSGL workloads that share a GPU.
