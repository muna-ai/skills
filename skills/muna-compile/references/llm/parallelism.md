# Model, replica and KV parallelism

Read before mapping a workload to several GPUs. The options below describe
NanoSGL execution; other backends have their own configuration.

## Distinguish the mechanisms

- **Tensor parallelism:** several GPUs cooperate on one model invocation. Set a
  supported `tensor_parallelism` in inference metadata. The deployment needs an
  eligible group of that many GPUs.
- **Replica parallelism:** independent copies of the model serve separate
  requests. Increasing tensor parallelism does not create more replicas.
- **KV partitioning:** distributing attention-cache state is model-specific;
  there is no general `kv_parallelism` field in the public metadata. Use only the
  documented parallelism options for the selected model.
- **KV-aware routing:** send a request to a replica that can reuse a prompt
  prefix. See [KV routing](kv-routing.md).
- **Disaggregation:** run prefill and decode separately. See
  [disaggregated serving](disaggregated-serving.md).

## Model-specific limits

TP must be positive. Check the [NanoSGL architectures and configurations](../inference-metadata/sglang.md#supported-architectures-and-configurations)
for each model's GPU architecture, permitted TP degrees and feature restrictions.

Other models may have additional constraints. Compile the selected configuration
and verify inference on suitable multi-GPU hardware rather than assuming every
positive TP degree is supported. Independently rented GPUs are not necessarily
a suitable group for tensor parallelism; placement and interconnect matter.
