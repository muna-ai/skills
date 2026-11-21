# ExecuTorch metadata

`ExecuTorchInferenceMetadata` is exposed by the SDK, but ExecuTorch compilation
is not currently supported. Contact Muna with the model and target device if
this backend is required.

The declared interface takes the [shared PyTorch export inputs](pytorch-export.md),
fixes `exporter=None`, and offers `backend="xnnpack"` for CPU or
`backend="vulkan"` for GPU. These options do not yet provide a working Android
compilation recipe; choose another supported backend if it meets the model's
requirements.
