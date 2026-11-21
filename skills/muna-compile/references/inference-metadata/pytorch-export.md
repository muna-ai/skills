# Shared PyTorch graph-export inputs

Read for the PyTorch graph-export families linked from the
[metadata chooser](index.md). These fields are not the interface for native
`TorchToSGLangInferenceMetadata` or `DiffusersToSGLangInferenceMetadata`.

## Attach metadata to the actual module

`model` is the `torch.nn.Module` instance called by the predictor. Export
metadata is matched to that object, not a class name or checkpoint string. Keep
the module at module scope. Exporters generally need its real weights and
representative tensor inputs; do not copy a native model recipe's meta-device
construction unless that exporter supports it.

The shared fields are:

| Field | Meaning |
| --- | --- |
| `model` | Required module instance. |
| `model_args` | A list of positional `torch.Tensor` samples, in the model's input order. The schema permits `None`, but tracing/export normally requires samples. |
| `input_shapes` | Optional shape tuples describing static/dynamic axes where the selected exporter supports them. |
| `exporter` | `None`, `"dynamo"`, `"torchscript"` or `"none"`; backend behavior differs. LiteRT and ExecuTorch fix this field to `None`. |
| `targets` | Optional platform scope; it cannot expand the backend's supported platforms. |

`input_shapes` does not replace `model_args`. Use samples with the correct dtype,
rank and calling convention. A dynamic-shape declaration does not promise an
arbitrary graph or every shape is exportable.

For export paths using the shared dynamic-shape conversion, integer dimensions
are static, `...` marks a dynamic axis, a string names an axis, and a slice such
as `slice(1, 9)` supplies bounds. Some paths also accept `torch.export.Dim`.
Follow the backend's restrictions: MLX rejects `input_shapes`, for example,
and ONNX export through TorchScript does not retain bounded-axis constraints.

## Export strategy is backend-specific

ONNX Runtime and Core ML normally select Dynamo when `exporter=None`.
`"torchscript"` uses a scripted/traced representation. `"none"` is a
backend-specific conversion path; it does not mean the compiled function embeds
an eager Python model. OpenVINO uses an exported program, LiteRT uses its own
converter, and QNN currently exports through ONNX with the legacy exporter.
Do not assume every shared field is honored by every conversion path.

After selecting a family, follow its guide for hardware policy and target
restrictions, then [validate](../validation.md) compilation and reference outputs.
