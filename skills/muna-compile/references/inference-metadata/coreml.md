# Core ML metadata

Use this family for Apple-platform inference. Core ML can use CPU, GPU and
Apple Neural Engine resources; this is distinct from an NVIDIA architecture
requirement.

## Choose the input representation

- `TorchToCoreMLInferenceMetadata`: takes the [shared PyTorch export inputs](pytorch-export.md).
  The default export strategy is Dynamo.
- `OnnxRuntimeToCoreMLInferenceMetadata`: takes `session`, `model_path`, optional
  `external_data_path` and `targets`, as described for
  [existing ONNX sessions](onnxruntime.md). It converts the model to Core ML;
  the session does not remain an ONNX Runtime session at inference.

Both types declare these optional fields:

| Field | Values |
| --- | --- |
| `compute_units` | List drawn from `"cpu"`, `"gpu"`, `"ane"`. |
| `compute_precision` | `"fp16"` or `"fp32"`. |
| `ios_deployment_target` | Integer from 13 through 26. |
| `macos_deployment_target` | Integer from 11 through 26. |

The two OS-version overrides are mutually exclusive. They specify a minimum
OS version, not the decorator's platform target.

## Current execution and conversion limits

The current integration does not guarantee GPU-only/ANE-only execution through
`compute_units`, or a specific conversion precision through `compute_precision`.
Device selection also depends on the requested Muna acceleration. Validate on
the intended device when these requirements matter.

The OS-version overrides apply to ONNX conversion, but are not currently honored
for PyTorch conversion. Confirm the required minimum OS with Muna before
choosing the PyTorch path for older devices.

Integration targets iOS, macOS and visionOS, subject to conversion and OS support.
Selecting a Linux build target does not make Core ML usable on a hosted Linux
node. Validate the model on its intended Apple device.
