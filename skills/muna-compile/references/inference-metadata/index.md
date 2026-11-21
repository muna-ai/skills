# Choose inference metadata

Read when deciding how a model object becomes compiled inference. Pass the
selected metadata in `@compile(metadata=[...])`, attached to the actual object
used by the predictor. Metadata selects that object's execution path and is the
source of truth for its hardware configuration. The function's modality and
client-facing annotations do not choose a backend.

Start with the object you have and follow only the relevant family guide.
The tables cover the current public inference metadata types, using their
non-deprecated names. They describe available configuration, not a promise that
every graph, field or target combination works; read the family's limitations.

## PyTorch module

For graph export, read [shared PyTorch export inputs](pytorch-export.md), then
the backend guide. Native NanoSGL integration has its own interface.

| Execution path | Metadata type | Reference |
| --- | --- | --- |
| ONNX Runtime graph | `TorchToOnnxRuntimeInferenceMetadata` | [ONNX Runtime](onnxruntime.md) |
| Apple Core ML | `TorchToCoreMLInferenceMetadata` | [Core ML](coreml.md) |
| Apple MLX | `TorchToMLXInferenceMetadata` | [MLX](mlx.md) |
| LiteRT graph | `TorchToLiteRTInferenceMetadata` | [LiteRT](litert.md) |
| Intel OpenVINO | `TorchToOpenVINOInferenceMetadata` | [OpenVINO](openvino.md) |
| Qualcomm QNN | `TorchToQnnInferenceMetadata` | [QNN](qnn.md) |
| NVIDIA TensorRT | `TorchToTensorRTInferenceMetadata` | [TensorRT; current limitations](tensorrt.md) |
| NVIDIA TensorRT RTX | `TorchToTensorRTRTXInferenceMetadata` | [TensorRT RTX; current limitations](tensorrt.md) |
| ExecuTorch | `ExecuTorchInferenceMetadata` | [Not yet supported](executorch.md) |
| Native model integration | `TorchToSGLangInferenceMetadata` | [NanoSGL](sglang.md) |

## Existing model or pipeline

| Object used by the function | Metadata type | Reference |
| --- | --- | --- |
| ONNX Runtime session, kept on ONNX Runtime | `OnnxRuntimeInferenceSessionMetadata` | [ONNX Runtime](onnxruntime.md) |
| ONNX Runtime session, converted to Core ML | `OnnxRuntimeToCoreMLInferenceMetadata` | [Core ML](coreml.md) |
| ONNX Runtime session, converted to MLX | `OnnxRuntimeToMLXInferenceMetadata` | [MLX](mlx.md) |
| ONNX Runtime session, converted to TensorRT | `OnnxRuntimeToTensorRTInferenceMetadata` | [Not yet supported](tensorrt.md) |
| TensorFlow Lite / LiteRT interpreter | `TFLiteInterpreterMetadata` | [LiteRT](litert.md) |
| `llama_cpp.Llama` model | `LlamaCppInferenceMetadata` | [llama.cpp](llama-cpp.md) |
| Native Diffusers pipeline | `DiffusersToSGLangInferenceMetadata` | [NanoSGL](sglang.md) |

An ONNX-origin model is not required to execute on ONNX Runtime: the metadata
selects whether to keep that runtime or convert its graph for another backend.
Likewise, using Transformers does not automatically select NanoSGL.

## Keep configuration scoped to its purpose

The decorator's `targets` chooses build platforms; some metadata also exposes a
`targets` scope. Neither replaces the selected backend's hardware configuration.
See [targets and hardware](../targets-and-hardware.md).

Plain Python/numerical functions may need no inference metadata. Other entries
in `metadata` can serve a different purpose: for example,
[KV routing metadata](../llm/kv-routing.md) describes prompt routing, not the
model's inference backend or GPU requirements.
