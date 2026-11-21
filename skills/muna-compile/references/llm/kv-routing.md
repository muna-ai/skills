# KV-cache-aware routing

Read when enabling prompt-prefix reuse across serving replicas. A key/value (KV)
cache retains attention state from earlier tokens. KV-aware routing can prefer a
replica that has already processed part of the incoming prompt.

## Declare the prompt-preparation function

```python
def _tokenize(messages: list[Message]) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    return tokenizer.encode(text, add_special_tokens=False)

# In @compile(metadata=[...]):
KVRoutingMetadata(tokenize=_tokenize)
```

The predictor must call **that same function** to obtain its prompt. Include
tools, images, audio, roles and any controls that affect prompt construction.
Routing based on a different prompt cannot reliably reuse the model's cache.

Use a plain module-level function, not a bound method, lambda or nested function.
Parameters must be named positional-or-keyword or keyword-only; no `*args`,
`**kwargs` or positional-only parameters. Its parameter names must be a subset
of the predictor's names.

Return prompt token IDs, or a supported processor output whose `input_ids`
contains the exact prompt. For multimodal models, preserve the processor output
when it carries image identity; different images can have identical placeholder
tokens. See [multimodal inputs](multimodal-inputs.md).

## Validate reuse for the intended workload

Routing is a performance optimization, not a guarantee of a cache hit or zero
prefill work. Use a model/configuration that supports KV reuse, and compare
repeated-prefix and unrelated-prompt requests at the expected concurrency.
Changing a prompt template, tool list or multimodal preprocessing can change
which prefixes are reusable.
