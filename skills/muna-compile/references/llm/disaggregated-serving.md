# Disaggregated prefill/decode serving

Read only when the application needs separate prefill/decode roles. Prefill
processes the prompt; decode generates subsequent tokens. Disaggregation runs
these phases on separate execution resources.

**Availability:** configuration types exist, but these recipes do not establish
a working hosted disaggregation setup. Confirm support for the selected model
and topology with Muna before choosing it for a deployment.

## Configuration

Pass `SGLangDisaggregationConfig` through
`TorchToSGLangInferenceMetadata.disaggregation`:

| Field | Meaning |
| --- | --- |
| `topology` | `"intra_node"` for separate roles on one node, or `"inter_node"` for roles on different nodes. |
| `prefill_tensor_parallelism` | Positive GPU count for prefill, or `None` to inherit `tensor_parallelism`. |
| `decode_tensor_parallelism` | Positive GPU count for decode, or `None` to inherit `tensor_parallelism`. |

Support is model-specific. Some native models reject disaggregation outright;
see [parallelism and model-specific limits](parallelism.md). A supported
ordinary generation configuration does not imply a supported disaggregated one.

Use ordinary combined prefill/decode serving unless disaggregation is required
and supported. Before relying on a disaggregated setup, validate real requests,
output correctness, cancellation, failure behavior and the intended latency /
throughput improvement.
