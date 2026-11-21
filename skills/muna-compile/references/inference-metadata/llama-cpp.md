# llama.cpp metadata

`LlamaCppInferenceMetadata` applies to an actual `llama_cpp.Llama` instance used
by the predictor. It packages that instance's GGUF model and optional LoRA file,
rather than exporting a PyTorch module or constructing a NanoSGL engine.

## Configuration

- `model`: required `llama_cpp.Llama` instance.
- `backends`: optional list containing `"cuda"` and/or `"vulkan"` to enable those
  accelerated backends in the compiled library.

The model object's context, batch, GPU-layer offload, embedding and LoRA settings
are also captured when compiling. The metadata declaration and captured model
configuration together describe this execution path. Do not add NanoSGL fields
such as `compute_architecture`, `tensor_parallelism` or `max_total_tokens`.

## Hardware and platform limits

`backends=None` adds no explicit CUDA/Vulkan enablement; it is not a promise of
CPU-only execution on every platform. The build can use its platform defaults,
including Metal on macOS arm64. GPU offload also depends on the model's
`n_gpu_layers` and the available backend.

The current build enables CUDA only for Linux x64 and Metal only for macOS arm64.
Other supported target platforms include Android, iOS, Linux, WebAssembly and
Windows, with hardware acceleration depending on the platform. Verify the
chosen model's execution on the intended device.

Preserve the GGUF model/quantization and validate the Python call surface used by
the predictor. For chat contracts and output chunks, see [LLM authoring](../llm/authoring.md);
its native Transformers request-manager recipe is a different execution path.
