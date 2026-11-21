# Diagnose a failed compile or prediction

Read after an error. Keep the full error message and the smallest script/input
that reproduces it before changing model architecture or dependencies.

| Stage / symptom | First useful checks |
| --- | --- |
| Local CLI import fails | Install local packages and check module paths. Remember that module-level setup can download files or initialize CUDA before compilation starts. |
| Sandbox import fails | Check `Sandbox` package versions, uploaded modules/files, working directory and credentials for gated checkpoints. |
| No entrypoint / wrong entrypoint | Use one decorated function in the module; imported decorated functions can also be discovered. |
| Signature extraction fails | Check input/return annotations, tuple arity, supported value types, synchronous functions and defaults for optional parameters. |
| Unsupported syntax | Follow [Python support](python-support.md); setup-time Python and the compiled prediction body have different restrictions. |
| Unsupported model or constant | Check the metadata type, the referenced model instance, supported checkpoint/construction and required files. Report unsupported models to Muna support. |
| Missing operator / no matching operator overload | Raise a Muna support ticket with a minimal script, full error, target, compute architecture and versions. |
| Native compilation fails | Check the documented target, metadata and dependency versions. If they match, send the full build error and script to Muna support. |
| No eligible implementation | Check the exact target triple, metadata's compute architecture/GPU requirements and that the intended version was compiled and deployed. |
| Waiting / staging looks stuck | Check available compatible hardware and the CLI's progress/error message; see [deployment readiness](deployment-lifecycle.md). |
| Predictor signature incompatible | Check the deployed function's annotations, required parameters and output types against the relevant modality guide. Image inputs must use the required list shape. |
| Model loading / queue timeout | Distinguish preparation, cold-start initialization and the request deadline using [deployment readiness](deployment-lifecycle.md). |
| Out of GPU memory | Check precision, context/token capacity and concurrency; test colocated models together. See [memory and colocation](memory-and-colocation.md). |
| Incorrect token/audio usage | Report actual token IDs or audio seconds, not text length, chunk count or prediction latency. Follow the modality's usage output type. |
| Unexpected latency | Measure input upload, queueing, cold start, inference and output download separately; measure TTFT and token rate separately for streaming. |
| Invalid raw value | Let the SDK encode/decode values; see [prediction values](prediction-values.md). |

## Confirm the fix

Repeat the check that failed, then run a representative compiled prediction and
compare it with the reference model. Include streaming completion, cancellation,
batching or multimodal inputs when those features matter to the application.
See [validation](validation.md) for the full sequence.

Report which checks passed. An output-shape test or successful Python execution
does not establish correct compiled inference.
