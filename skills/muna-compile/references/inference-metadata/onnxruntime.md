# ONNX Runtime metadata

Use this family to execute an exported PyTorch graph or an existing ONNX model
with ONNX Runtime. Select CPU or accelerator execution through `providers`.

## Export a PyTorch module

`TorchToOnnxRuntimeInferenceMetadata` uses the
[shared PyTorch export fields](pytorch-export.md), plus:

- `optimization`: `"none"` (default), `"basic"` or `"extended"` graph optimization
  during compilation.
- `providers`: optional list drawn from `"cpu"`, `"coreml"` and `"cuda"`.

For example, with an initialized module and representative input tensor:

```python
from muna.beta import TorchToOnnxRuntimeInferenceMetadata

metadata = TorchToOnnxRuntimeInferenceMetadata(
    model=model,
    model_args=[sample_input],
    providers=["cpu"]
)
```

Pass this entry in the decorator's `metadata` list. The predictor should call
the same `model` instance. This fragment configures the backend; it does not
replace the function's typed input/output contract.

## Existing ONNX sessions

`OnnxRuntimeInferenceSessionMetadata` takes:

| Field | Meaning |
| --- | --- |
| `session` | Required actual `onnxruntime.InferenceSession` used by the function. |
| `model_path` | Required ONNX graph path, present inside the compiler sandbox. |
| `external_data_path` | Optional external weights file. Keep the graph's sidecar reference consistent with that file. |
| `providers` | Optional list with the same provider names as above. |
| `targets` | Optional platform scope. |

The session's Python construction alone is not the compiled runtime's provider
configuration; set the metadata deliberately. The model and external weights
are packaged as resources. See [dependencies and resources](../resources-and-dependencies.md)
when arranging files in the sandbox.

## Hardware and platforms

`providers` filters the providers available for the platform and runtime
acceleration setting. With `None`, the runtime uses its available defaults:
CPU is available, with Core ML on Apple platforms and CUDA on Linux x64 when
the corresponding acceleration is enabled. The list is treated as a set,
not an ordered priority list. ONNX Runtime can still execute unsupported
subgraphs on CPU; listing CUDA is not a guarantee of GPU-only execution.

Use `providers=["cpu"]` for an explicitly CPU-oriented session. Do not add
`compute_architecture`, `cuda_arch` or tensor-parallel settings: these are not
fields on this family. Provider names refer to this API, so use `"cuda"`, not
`"CUDAExecutionProvider"`.

Supported target platforms include Android, iOS, Linux x64/arm64, macOS, wasm32
and Windows, with different providers available on each. A CUDA provider on
Linux says nothing about GPU execution on mobile or wasm. Validate the actual
graph and selected target rather than treating the platform list as operator
parity across devices.
