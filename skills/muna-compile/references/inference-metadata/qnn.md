# Qualcomm QNN metadata

`TorchToQnnInferenceMetadata` converts a PyTorch module for QNN inference. Its current targets are Android arm64 and Windows arm64.
Choose it for a compatible Qualcomm execution path, not as a generic GPU backend.

It takes the [shared PyTorch export inputs](pytorch-export.md), plus:

| Field | Values and meaning |
| --- | --- |
| `backend` | `"cpu"` (default), `"gpu"` or `"htp"`. This selects the QNN execution backend. |
| `quantization` | Optional `"w8a8"`, `"w8a16"`, `"w4a8"` or `"w4a16"`; intended for HTP only. |
| `quantization_samples` | Optional list of representative input tuples for calibration; see the limitation below. |

Use real positional tensor samples in `model_args`. The current path uses the
legacy ONNX exporter; changing the shared `exporter` field does not select a
Dynamo conversion. `compute_architecture` and `cuda_arch` are not fields here.

## Quantization limitation

`quantization_samples` currently has no effect. Do not rely on it for model
calibration; confirm the required HTP quantization path with Muna. Preserve the
requested precision and compare outputs against the reference model.

A QNN GPU/HTP declaration requires that backend on the intended device. It is
not a request for an NVIDIA GPU or an Apple Neural Engine.
