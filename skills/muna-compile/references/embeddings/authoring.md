# Text embeddings

Read for embedding functions and SentenceTransformer models.

## Choose the execution path

Use [inference metadata](../inference-metadata/index.md) to choose the model's
backend and hardware. An embedding API contract does not require NanoSGL or a
GPU. A compatible module/graph can use an export backend; the native
SentenceTransformer recipe below is a separate integration with specific
checkpoint and configuration restrictions.

## Public contract

This example shows the OpenAI embeddings interface. Replace `...` with the
model call and add its build configuration before compiling.

```python
from muna import compile, Parameter
from numpy import ndarray
from typing import Annotated

@compile()
def embeddings(texts: list[str]) -> Annotated[
    ndarray,
    Parameter.Embedding(description="Float32 text embedding matrix.")
]:
    """Embed a list of texts."""
    ...
```

For Muna to recognize this embeddings interface:

1. Declare one required `list[str]` input.
2. Return a float32 array shaped `[texts, dimensions]`, with one row per input
   text. Bare `ndarray` declares float32; return data with that dtype.
3. Annotate the matrix with `Parameter.Embedding`.
4. Keep the text-list input and embedding output unambiguous.
5. Give additional inputs a default unless the embeddings interface supplies
   them through a supported annotation.

Add optional `Annotations.EmbeddingDims` only when truncation is implemented. Return
`EmbeddingCreateResponse.Usage` as a separate tuple output when exact tokenizer
usage is available. Token counts include model-required task prefixes and special
tokens. Use the SDK's `EmbeddingCreateResponse.Usage` type for that output;
an arbitrary dictionary is not a substitute.

The [protocol example](../../examples/embeddings/embeddings.py) demonstrates this
contract without a neural network; [Nomic](../../examples/embeddings/nomic.py)
is the native NanoSGL HF model recipe.

## Native SentenceTransformer path

The native path supports `nomic-ai/nomic-embed-text-v1.5` (mean pooling) and
`Qwen/Qwen3-Embedding-8B` (last-token pooling). They require a Transformer/Pooling
module pipeline, optionally followed by Normalize. They reject incompatible
pooling, excluded prompt tokens (`include_prompt=False`) and lowercase processing
(`do_lower_case=True`). Follow the exact profile rather than substituting another
embedding checkpoint with the same hidden size.

Use the supported meta-device model construction and attach
[TorchToSGLangInferenceMetadata](../inference-metadata/sglang.md) to the same
SentenceTransformer object. This path is single-GPU and rejects non-default
`tensor_parallelism`, speculative decoding, disaggregation, `max_running_requests`
and `max_total_tokens` decoder knobs.

## Concurrent requests

NanoSGL engines can batch compatible queued calls internally. Keep one invocation
per independent request. When configuring entry behavior, follow the shared
[batching guide](../batching.md); do not add a second Python batching layer solely
because the function accepts a list of texts. Multiple texts within a call and
multiple concurrent calls are different axes.

## Correctness checks

Preserve task instructions (`search_query`, `search_document`, etc.), prompt
inclusion in pooling, truncation, normalization and output dtype. Validate input
length explicitly if the product promises rejection rather than silent tokenizer
truncation. Test variable-length texts, requested dimensions and an empty-input
policy. Compare embeddings/similarity against the pinned Python reference.
