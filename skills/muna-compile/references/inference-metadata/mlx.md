# MLX metadata

Use this family for MLX inference on macOS with an Apple GPU. The current
integration requires GPU execution; it does not offer a CPU-only option.

## Input types

- `TorchToMLXInferenceMetadata`: takes the [shared PyTorch export inputs](pytorch-export.md).
  Provide real tensor samples in `model_args` for conversion.
- `OnnxRuntimeToMLXInferenceMetadata`: takes the actual `session`, `model_path`,
  optional `external_data_path` and `targets`, using the
  [existing ONNX session interface](onnxruntime.md). The existing graph is
  converted to an MLX function rather than executed by ONNX Runtime.

There are no additional backend-specific constructor fields on these types.
In particular, neither accepts `compute_architecture` or `cuda_arch`.

## Limits

The PyTorch converter rejects any non-`None` `input_shapes`; do not claim dynamic
shape support from the shared field alone. Use representative fixed shapes and
validate the resulting function. The ONNX converter also needs a graph that the
MLX exporter can represent.

This integration targets macOS using Metal. Do not infer iOS,
Linux or WebAssembly support from MLX's presence in the metadata API. Ensure
external ONNX weights are available at the graph's referenced locations during
conversion.
