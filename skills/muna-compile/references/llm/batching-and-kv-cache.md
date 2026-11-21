# Batching, context and KV cache

Read when selecting LLM context or concurrency limits. During generation, the
engine keeps attention key/value (KV) state so later tokens can reuse earlier
work. This retained state affects how many requests and tokens fit together.

Keep these quantities separate:

| Quantity | Meaning |
| --- | --- |
| Annotated `context_length` | Signature-level advertised input context; keep consistent with what the model actually accepts. |
| Per-request maximum output | Sampling/request constraint; account for input length and model context. |
| `max_running_requests` | Maximum active requests; this does not guarantee they all fit at maximum context simultaneously. |
| `max_total_tokens` | Engine token-capacity/context setting. It is not a byte budget or a guarantee that every concurrent request can occupy its maximum context. |
| KV-cache memory | GPU memory used for active contexts and reusable prompt prefixes; depends on model, dtype and TP. |
| Weight/workspace/graph memory | Separate from KV and often substantial. |

## Continuous batching

Annotate the request-bearing parameter with `BatchConfig(mode="continuous")`
when concurrent calls should feed the shared engine. This is compatible with one
conversation per function call. Request IDs and stream state are per invocation;
the manager/engine is shared. Do not put the entire streamed generation under a
global Python lock or construct a fresh engine for every request.

Keep context validation in the predictor, including model-specific speculative
slack. A 16k context and concurrency 64 does not mean 64 full 16k KV allocations
are guaranteed on the chosen GPU. Validate realistic active sequence lengths and
colocated weights before promising a service envelope.

## Defaults and shared-GPU behavior

Common NanoSGL defaults are 32 running requests, `max_total_tokens=65536`
and `tensor_parallelism=1`; individual models can have stricter limits. Choose explicit limits appropriate to the
model and intended workload rather than inheriting a large default blindly.

For request-entry annotations and the difference between merged requests and
engine-managed batching, read [batching](../batching.md). For memory limits when
several models share a GPU, read
[memory and colocation](../memory-and-colocation.md).
