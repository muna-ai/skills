# TensorRT and TensorRT RTX metadata

These metadata types select NVIDIA inference backends. The current TensorRT
integrations have compilation limitations and should not be treated as working
recipes. Contact Muna with the model, target and SDK/compiler versions before
choosing one for a new deployment.

## TensorRT

`TorchToTensorRTInferenceMetadata` takes the
[shared PyTorch export inputs](pytorch-export.md) and requires `cuda_arch`.
`OnnxRuntimeToTensorRTInferenceMetadata` takes `session`, `model_path`, optional
`external_data_path` and `targets`, and also requires `cuda_arch`.

The SDK accepts `sm_80`, `sm_89`, `sm_90`, `sm_100`, `sm_103`, `sm_107` and
`sm_120`. An accepted value does not establish that the selected model can be
converted for that architecture.

The PyTorch path is intended for Linux x64 and Windows x64, but currently has a
compiler compatibility issue. ONNX-session conversion to TensorRT is not
currently supported. Use a supported alternative only if it preserves the
required model behavior and precision.

## TensorRT RTX

`TorchToTensorRTRTXInferenceMetadata` takes the shared PyTorch export fields.
It has no `cuda_arch` or `compute_architecture` option. Its intended targets are
Linux x64 and Windows x64.

This path also currently has a compiler compatibility issue. Confirm support
with Muna before relying on it. Hardware support must be checked for the
specific TensorRT RTX path and model.
