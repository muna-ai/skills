# Speculative decoding

Read only when adding speculative decoding. A draft model proposes tokens that
the target model verifies; a supported pairing can reduce generation work while
preserving the target model's decoding contract.

The public configuration is `SpeculativeDecodingConfig(draft_model=...,
num_draft_tokens=..., eagle3_num_steps=..., eagle3_topk=...)`. The native strategy
depends on the selected target model and its supported draft checkpoint/config.

```python
TorchToSGLangInferenceMetadata(
    model=model,
    compute_architecture="sm_90",
    speculative_decoding=SpeculativeDecodingConfig(
        draft_model=draft_model,
        num_draft_tokens=8
    )
)
```

This is a configuration fragment, not a universally valid target/drafter pairing.

## Choose a supported drafter

Use the target model's documented drafter pairing. Hidden dimensions, vocabulary
mapping, layer taps, block size and checkpoint format must agree. Some draft
configs are not ordinary `AutoConfig` model types; the supported Phi DSpark recipe
uses `PretrainedConfig` and a `Module` carrying it. Follow the construction for
that pairing rather than substituting an arbitrary configuration.

Examples of real constraints: the Phi DSpark integration pins its
block/num-draft-tokens rules; Qwen3.8 DFlash2 derives its block size from the draft
config and rejects a different explicit `num_draft_tokens`. Use the exact
supported draft checkpoint even when another has a similar name.

## Preserve correctness

Verification produces several IDs at once and may need extra context/cache
slack beyond committed tokens. Follow the selected model's context and batching
limits rather than adding `num_draft_tokens` to every model's context limit.
Include drafter weights/workspace in practical memory planning and compare end-to-end TTFT and token rate at the intended concurrency.

Do not enable speculation by default merely because it is available. Establish
the base model works first, then validate the requested pairing with real
weights and representative prompts.
