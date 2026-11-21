# NanoSGL GPU memory and colocation

Read for [NanoSGL execution](inference-metadata/sglang.md) when several models
will share a GPU or a load/inference request runs out of memory.

## Size the workload, not just the weights

GPU memory must fit model weights, working buffers and active request state.
For LLMs, the attention key/value (KV) cache can be substantial: longer contexts
and more concurrent generations increase memory demand.

Keep advertised context and concurrency limits consistent with the tested
configuration. An annotation does not reserve memory, and a maximum context
length is not a guarantee that every concurrent request can use that length.
See [batching and KV cache](llm/batching-and-kv-cache.md) for the relevant settings.

NanoSGL can release reusable cache memory when another model needs space. This
does not make room for arbitrary combinations of model weights and active
requests. A configuration that fits alone may fail when colocated.

## Respond to an out-of-memory error

Check the selected precision, context limit, token capacity, concurrency and
number of GPUs. Reduce only the limits the application can accept, or select
hardware with sufficient memory. Changing precision or model identity requires
checking output quality again.

For colocation, test the actual set of models together, including loading a new
model while the others are serving requests. Compare latency and throughput at
the expected concurrency. A successful isolated run does not establish that
service envelope.

If the delay is in deployment preparation or first-request initialization, use
[deployment readiness and cold starts](deployment-lifecycle.md) to distinguish
those stages before changing model settings.
