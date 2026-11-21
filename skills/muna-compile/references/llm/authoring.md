# Author a language-model predictor

Read for text generation and conversational models. A typical chat predictor
prepares a prompt, submits generation to a shared model instance, and turns the
result into text chunks. Its public chat contract is separate from the execution
backend.

## Choose the execution path

Read [inference metadata](../inference-metadata/index.md) for the model object
and intended hardware. Native Transformers integration through NanoSGL and an
existing [llama.cpp model](../inference-metadata/llama-cpp.md) are different paths;
an exported graph needs a compatible exporter and generation implementation.
Using a Transformers class does not itself select NanoSGL or require a GPU.

Preserve the tokenizer, chat template, role formatting and stop tokens. Keep
model/tokenizer setup at module scope. Define a helper for prompt preparation
and have the prediction function use it consistently.

## Expose a chat interface

This signature enables OpenAI chat completions and can also support Anthropic
messages. The example shows the interface only: replace `...` with generation
logic and add the build configuration for the chosen model before compiling.

```python
from muna import compile
from muna.beta import Annotations
from muna.beta.openai import ChatCompletionChunk, Message
from typing import Annotated, Iterator

@compile()
def chat(
    messages: Annotated[list[Message], Annotations.ChatMessages(
        description="Conversation messages.",
        context_length=4096
    )]
) -> Iterator[ChatCompletionChunk]:
    """Stream chat completion chunks."""
    ...
```

For Muna to recognize this chat interface:

1. Declare one required `list[Message]` input using the SDK's `Message` type.
2. Annotate it with `Annotations.ChatMessages` and a positive `context_length`
   matching the model's supported limit. The parameter name alone is insufficient.
3. Return a synchronous stream of SDK `ChatCompletionChunk` objects, annotated
   as `Iterator[ChatCompletionChunk]`.
4. Keep the conversation input and chunk output unambiguous.
5. Give additional inputs a default unless the chat interface supplies them
   through a supported control annotation.

Other supported signatures remain callable through the Muna prediction API;
these requirements are for the standard chat interface.

Read [generation and streaming](generation-and-streaming.md) when implementing
chunk delivery, controls, usage or cancellation. Parameter defaults follow the
shared [signature rules](../signatures-and-annotations.md).

## Native Transformers / NanoSGL recipe

For a supported native model, attach
[TorchToSGLangInferenceMetadata](../inference-metadata/sglang.md) to the same
model object the function uses. Its architecture and supported device settings
supply this execution path's hardware requirements.

Where the model recipe supports it, construct the model on the meta device:
this records topology and tensor shapes without loading all weights during
script import. Load the actual tokenizer and preserve checkpoint identity.
This construction is specific to the native recipe; graph exporters generally
need real weights and representative tensors.

The native continuous-batching path uses a shared request manager. It admits
independent requests and lets the engine batch their work; keep it at module
scope rather than recreating it per call. This manager is not a requirement for
other backends or for the chat adapter contract.

See [the Qwen3 recipe](../../examples/llm/qwen3.py) for a complete text-only native
example. It does not implement tools, reasoning or a speculative drafter. The
smaller [chat fixture](../../examples/llm/chat.py) only exercises the interface.

Additional native settings are relevant only to the requested feature:

| Need | Setting and guide |
| --- | --- |
| Context/concurrency limits | `max_total_tokens`, `max_running_requests`: [batching and KV cache](batching-and-kv-cache.md) |
| Several cooperating GPUs | `tensor_parallelism`: [parallelism](parallelism.md) |
| Reuse prompt prefixes across replicas | Separate routing metadata: [KV routing](kv-routing.md) |
| Draft/verify generation | `speculative_decoding`: [speculative decoding](speculative-decoding.md) |
| Separate prefill/decode execution | `disaggregation`: [disaggregated serving](disaggregated-serving.md) |
| Image/audio attachments in chat | [Multimodal inputs](multimodal-inputs.md) |

`experimental` settings are not a general escape hatch for unsupported features.
Use only a documented option for the requested integration. Build and validate
the ordinary model path before adding optional mechanisms.
