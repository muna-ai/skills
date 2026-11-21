# LiteRT and TensorFlow Lite metadata

This family executes a `.tflite` graph with platform-specific interpreter and
delegate support. The default CPU path uses XNNPACK. GPU/NPU execution depends
on the selected options and target, not the model's modality.

## Input types

`TorchToLiteRTInferenceMetadata` takes the
[shared PyTorch export inputs](pytorch-export.md), fixes `exporter=None`, and adds
optional `options`. Supply positional tensor samples in `model_args`;
`input_shapes` alone does not establish dynamic-shape support.

`TFLiteInterpreterMetadata` takes:

| Field | Meaning |
| --- | --- |
| `interpreter` | Required `tensorflow.lite.Interpreter` or `ai_edge_litert.interpreter.Interpreter` used by the function. |
| `model_path` | Required `.tflite` file path, present in the compiler sandbox. |
| `options` | Optional list of interpreter/delegate options below. |
| `targets` | Optional platform scope. |

Both types configure the compiled interpreter through metadata. Do not assume
a Python interpreter's delegate objects are automatically retained.

## Hardware options

| Execution path | Option names |
| --- | --- |
| CPU / XNNPACK | `xnnpack_force_fp16`, `xnnpack_slow_consistent_arithmetic`, `xnnpack_subgraph_reshaping` |
| Core ML delegate on iOS/macOS | `coreml`, `coreml_neural_engine_only` |
| GPU delegate on Android | `gpu`, `gpu_min_latency`, `gpu_sustained_speed`, `gpu_disallow_precision_loss` |
| NNAPI delegate on Android | `nnapi`, `nnapi_disallow_fp16` |

Choose the delegate and any tuning options appropriate to that platform. These
are not CUDA compute-architecture strings. A delegate may handle only part of a
graph; validate execution and precision on the target device.

The PyTorch conversion path targets Android, iOS, Linux x64/arm64, macOS and
Windows. Existing interpreter models can also target wasm32. Do not infer WebAssembly support for the PyTorch conversion merely because
an existing `.tflite` model can use the interpreter path.
