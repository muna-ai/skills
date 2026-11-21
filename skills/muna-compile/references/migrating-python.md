# Existing Python and Modal migration

Read when the starting point is an application rather than a clean model script.

## Extract the inference contract

Identify model construction, one request's inputs, inference call, outputs and
streaming behavior. Preserve preprocessing, tokenizer/chat-template behavior,
dtype conversions, stop conditions and postprocessing. Identify persistent
caches and state explicitly rather than assuming every global can be serialized.

For Modal code:

| Existing concern | Muna translation |
| --- | --- |
| `modal.Image` package/system dependencies | Local authoring dependencies plus explicit `Sandbox` pip/apt/setup commands. Only packages actually needed during build belong there. |
| `@modal.enter()` model initialization | Module-scope model/tokenizer setup. Use the selected inference metadata and its documented construction pattern. |
| `@modal.method()` or FastAPI endpoint | One typed synchronous function or synchronous generator decorated with `@compile`. Extract request parsing from computation. |
| Mounted volume / downloaded model directory | Supported checkpoint staging, uploaded build inputs or explicit compile resources. Not an undeclared runtime path. |
| GPU selection / container count | Compile hardware requirements plus Muna deployment policy. Do not translate container count into tensor parallelism. |
| Async HTTP generator | Synchronous generator of supported typed chunks. Async is rejected at the predictor boundary. |
| HTTP error / `StreamingResponse` / SSE framing | Predictor exception or typed output/chunks. SDK adapters and servers own HTTP framing. |
| Secrets | Build-time environment references only where needed; no hard-coded keys or automatic migration of production secrets. |

## Check feasibility before rewriting

Run through the actual computation's APIs. A Modal script can execute arbitrary
CPython and extension packages; Muna needs native operator/backend coverage.
`trust_remote_code=True`, a familiar HF class name, or successful eager Python
inference is not proof of compilability.

If the compiler reports an unsupported model or missing operator, tell the
user to raise a Muna support ticket with the model configuration and error.
An alternative backend is a proposal when it changes behavior
or precision, not a silent substitution. Do not erase validation, tool parsing,
sampling controls or cancellation merely to shorten an example.

Start with an existing supported model recipe, replacing only the supported
parts needed for the requested behavior. Validate the compiled predictor against
the original function using representative inputs.

See [compilation model](compilation-model.md), [library support](library-support.md)
and [models and backends](models-and-backends.md).
