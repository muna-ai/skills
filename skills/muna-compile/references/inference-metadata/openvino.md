# OpenVINO metadata

`TorchToOpenVINOInferenceMetadata` converts a PyTorch module to an OpenVINO model
for Linux x64 or Windows x64. It takes the
[shared PyTorch export fields](pytorch-export.md), with no additional
hardware-selection fields.

Supply representative tensor samples in `model_args`. This path uses PyTorch
export; selecting a different `exporter` does not change that conversion.

## Hardware policy

This type does not declare an NVIDIA compute architecture or mandatory GPU.
The runtime starts with CPU execution, or uses OpenVINO AUTO selection with
CPU fallback when Muna's acceleration setting requests an available Intel GPU
or NPU. Do not add `compute_architecture`, `cuda_arch`, `device` or a NanoSGL
tensor-parallel setting to the metadata.

Conversion currently saves weights with FP16 compression enabled. Validate
numerical behavior if preserving the original precision is a requirement;
there is no precision override field on this type. Selecting an Intel backend
does not prove that every device plugin supports the exported graph. Check the
actual target device and compare against reference outputs.
