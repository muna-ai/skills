# Native NanoSGL metadata

Read when selecting NanoSGL through `TorchToSGLangInferenceMetadata` or
`DiffusersToSGLangInferenceMetadata`. Use the configuration for the specific
model; PyTorch, Transformers or Diffusers usage alone does not select it.

## Supported architectures and configurations

NanoSGL compatibility is determined by the model's architecture, supported
configuration and weight representation. A repository name identifies the
source of the weights; it is not an architectural requirement. A fine-tune or
merged checkpoint that changes only weight values can, in principle, use the
same engine.

Compatibility includes more than an architecture class name:

| Aspect | What must be compatible |
| --- | --- |
| Computation | Attention, recurrence, MoE, normalization, positional encoding and other model behavior. |
| Dimensions | Supported layer counts, hidden sizes, head geometry, vocabulary and component dimensions. Support for one model size does not imply all sizes in the family. |
| Weight representation | Tensor names, shapes, dtypes and packing understood by the model integration. |
| Quantization | Exporter format, group sizes, scales, activation quantization and per-module precision choices. |
| Execution | GPU architecture, tensor parallelism and requested inference features. |

Preserve the selected checkpoint's tokenizer, chat template and preprocessing.
A fine-tune with a resized vocabulary or changed model configuration needs
support for those changes; matching the base model's name is insufficient.
An adapter-only LoRA checkpoint also needs explicit adapter support and is
different from a merged full-weight checkpoint.

### Current compiler limitation

The compiler currently selects many native model recipes using explicit
checkpoint allowlists. A structurally compatible checkpoint under another
repository name can therefore still be rejected. Some task-specific recipes
also require pinned model/source revisions and component configurations.

The checkpoint names below are reference examples for the listed architectures
and formats, and identify the repositories recognized by those current recipes.
They are not a claim that NanoSGL fundamentally requires those exact weight
values. If a compatible derivative is rejected, provide its checkpoint,
configuration and quantization format to Muna support rather than substituting
the base model's weights.

For repository-backed recipes, do not assume in-memory weight changes are
packaged automatically. Use a supported path that preserves the actual modified
weights and revision; a model's original repository identity alone does not
describe those changes.

### Weight formats

An architecture can support several weight formats, but a label such as
`NVFP4` is not a complete compatibility description. For example, the Qwen3.8
27B integration supports BF16 and specific ModelOpt and compressed-tensors
exports that mix NVFP4 and FP8 across modules. Their packing, scale conventions
and module assignments differ.

Choose a checkpoint whose full representation is supported. Do not assume
another 4-bit export works because it has the same architecture or nominal
precision. Likewise, architecture-compatible target weights do not establish
that a particular speculative drafter is a supported or effective pairing.

### Language models

These models use `TorchToSGLangInferenceMetadata` with the native Transformers
generation recipe:

| Architecture / configuration | Reference checkpoints | Important limits |
| --- | --- | --- |
| Phi-4 Mini | `microsoft/Phi-4-mini-instruct` | BF16; use the Phi-4 Mini construction and tokenizer. |
| Qwen3 4B | `Qwen/Qwen3-4B-Instruct-2507`; `Qwen/Qwen3-4B` | These are distinct checkpoints; preserve their respective chat templates and thinking behavior. See the [Qwen3 recipe](../../examples/llm/qwen3.py). |
| Qwen3.6 27B | `Qwen/Qwen3.6-27B`; `Qwen/Qwen3.6-27B-FP8` | `sm_90`, TP=1; no disaggregation. Use the supported pinned revisions. |
| Qwen3.8 27B | `Qwen/Qwen3.8-27B`; `RadixArk/Qwen3.8-27B-NVFP4`; `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` | The supported 27B geometry; BF16 and specific ModelOpt / compressed-tensors NVFP4/FP8 representations. |
| Gemma 4 26B-A4B | `google/gemma-4-26B-A4B-it`; `nvidia/Gemma-4-26B-A4B-NVFP4` | Supported BF16 and NVFP4 configurations; preserve the MoE topology and weight representation. |
| Gemma 4 31B | `google/gemma-4-31B-it` | Dense BF16, text-only generation in this integration. |
| GPT-OSS 120B | `openai/gpt-oss-120b` | MXFP4 checkpoint; support does not extend to GPT-OSS 20B. |
| DeepSeek V4.1 Flash | `deepseek-ai/DeepSeek-V4.1-Flash`; `deepseek-ai/DeepSeek-V4.1-Flash-1017`; `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` | `sm_100`, TP=4; no speculative decoding or disaggregation. |
| GLM 5.3 Flash | `nvidia/GLM-5.3-Flash-NVFP4` | `sm_100`, TP in {2, 4, 8}; no speculative decoding or disaggregation. |
| MiniMax M3.1 Flash | `MiniMaxAI/MiniMax-M3.1-Flash` | `sm_100`, TP in {2, 4}; no speculative decoding or disaggregation. |

TP means `tensor_parallelism`. Where a row does not specify a GPU architecture
or TP degree, choose a supported configuration for the model and weight format;
the row does not imply every GPU architecture or TP degree works. Model support
also does not imply every optional feature is available. Follow
[LLM authoring](../llm/authoring.md), and use the supported target/drafter pairing
when adding [speculative decoding](../llm/speculative-decoding.md).

### Embeddings, transcription and depth

These also use `TorchToSGLangInferenceMetadata`, with their own model construction
and input/output contracts:

| Architecture / task | Reference checkpoint | Important limits |
| --- | --- | --- |
| Nomic Embed Text v1.5 | `nomic-ai/nomic-embed-text-v1.5` | Single GPU; SentenceTransformer with mean pooling. |
| Qwen3 Embedding 8B | `Qwen/Qwen3-Embedding-8B` | Single GPU; SentenceTransformer with last-token pooling. |
| Whisper Large v3 | `openai/whisper-large-v3` | Use the supported Whisper model/processor and FP16 weights; no disaggregation. |
| Depth Anything 3 Nested Giant/Large 1.1 | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | `sm_100`, single-GPU FP32; supported nested metric-depth wrapper; height/width must be multiples of 14 in [14, 1022]. |

The embedding paths reject decoder-specific metadata such as
`max_total_tokens`, `max_running_requests`, speculative decoding and
disaggregation. Follow [embedding authoring](../embeddings/authoring.md),
[transcription](../audio/transcription.md) or [depth outputs](../images/depth.md)
for the selected task.

### Diffusion and image-to-image models

Use `DiffusersToSGLangInferenceMetadata` for these pipelines:

| Architecture / task | Reference checkpoint(s) | Important limits |
| --- | --- | --- |
| FLUX.2 Klein 9B | `black-forest-labs/FLUX.2-klein-9B` | Single GPU; this does not cover other FLUX variants. |
| Lotus Depth G v2.1 disparity | `jingheya/lotus-depth-g-v2-1-disparity` | `sm_100`, single-GPU FP16; use the supported `LotusDepthGPipeline` and pinned model/source revisions. |
| Marigold v2 depth and albedo | `huawei-bayerlab/marigold-v2-0` with the pinned `Qwen/Qwen-Image-Edit-2509` base | Use `MarigoldV2DepthPipeline` with the `depth/Log-stage2` adapter, or `MarigoldV2AlbedoPipeline` with the `albedo` adapter. Preserve each task's prompt conditioning. |

The Lotus and Marigold entries require their task-specific pipeline recipes;
passing an arbitrary Diffusers pipeline with the same base model is insufficient.
See [image authoring](../images/authoring.md) for the public function interfaces.

### Idol image-to-3D recipe

NanoSGL also supports `speridlabs/idol` through the original
`idol.IdolPipeline` recipe. It does not take either SGLang metadata class.
Use that pipeline's supported construction instead of wrapping it in
`TorchToSGLangInferenceMetadata` or `DiffusersToSGLangInferenceMetadata`.

This recipe requires one `sm_100` GPU, the DINOv3 ViT-H+ encoder
(`facebook/dinov3-vith16plus-pretrain-lvd1689m`) with
`dinov3_dtype="bf16"`, and the `black-forest-labs/FLUX.2-dev` VAE encoder.
Preserve the supported SDK/component versions. It accepts one reference image
and produces a Gaussian-splat asset; follow [3D outputs](../images/model3d.md)
for the annotated image-list input and binary PLY output.

### Models not currently available through this path

Kimi K2.5, Kimi K2.6, Kimi K3, GLM 5.2 and DeepSeek V4 Flash have unfinished
inference support and should not be offered as working generation recipes.
DeepSeek V4 Flash is distinct from the supported V4.1 Flash checkpoints above.

For an unlisted architecture or configuration, ask Muna support about a
compatible recipe or another [inference backend](index.md). For an unlisted
checkpoint of a listed architecture, establish configuration/format compatibility
and account for the current compiler allowlist. Preserve the requested model
and precision when explaining any limitation.

## PyTorch model integration

`TorchToSGLangInferenceMetadata` takes:

| Field | Meaning |
| --- | --- |
| `model` | Required actual PyTorch module used by the predictor. |
| `compute_architecture` | Required execution architecture. See the restrictions below. |
| `max_running_requests` | Optional positive scheduler concurrency limit for integrations that support it. |
| `max_total_tokens` | Optional positive total KV-cache capacity for supported decoder engines. |
| `tensor_parallelism` | Optional positive number of cooperating GPUs. |
| `speculative_decoding` | Optional speculative-generation configuration. |
| `disaggregation` | Optional prefill/decode configuration. |
| `experimental` | Optional integration-specific settings; use only documented options. |

This metadata can select native integrations for language generation, supported
SentenceTransformer models, Whisper or depth models. Supported settings vary by
model; decoder controls do not apply to embedding, audio or image models. Follow the [LLM](../llm/authoring.md), [embedding](../embeddings/authoring.md),
[audio](../audio/authoring.md) or [image](../images/authoring.md) recipe for the
specific model. Preserve its checkpoint, precision and supported construction.

## Diffusers pipeline integration

`DiffusersToSGLangInferenceMetadata` requires `pipeline` (the actual
`diffusers.DiffusionPipeline` instance) and `compute_architecture`, with optional
positive `max_batch_size`. It does not take `model`, decoder-cache settings or
speculative-generation controls. Follow [image authoring](../images/authoring.md)
for supported pipeline construction and output semantics.

Neither native type inherits the graph-export fields `exporter`, `model_args`,
`input_shapes` or metadata-level `targets`. The decorator can still select build
platforms; that does not change the model's hardware requirements.

## GPU architecture and device requirements

An architecture identifies the instruction set, not a GPU product name. For
example, `sm_90` denotes Hopper and `sm_100` datacenter Blackwell; `h100` and
`b200` are not valid architecture strings.

The CUDA architecture options are `sm_80`, `sm_89`, `sm_90`, `sm_100`,
`sm_103` and `sm_120`, subject to each model's supported configurations.
The SDK also accepts `sm_107`, `gfx950`, `metal_3` and `metal_4`, but those names
alone do not establish a working native compilation path. Do not offer a model
on HIP/Metal or another architecture solely because its metadata validates.

Compile for the architecture of the intended GPU. An `sm_90` build cannot be
used on a B200 merely because Blackwell is newer; use a supported `sm_100`
configuration for that GPU.

Where supported, `tensor_parallelism` specifies the number of cooperating GPUs
needed to run the model. For advanced LLM configuration, read [parallelism](../llm/parallelism.md),
[cache capacity](../llm/batching-and-kv-cache.md),
[speculative decoding](../llm/speculative-decoding.md) or
[disaggregation](../llm/disaggregated-serving.md) only when requested.

Read [memory and colocation](../memory-and-colocation.md) when selecting context,
concurrency and hardware for several models sharing a GPU.
