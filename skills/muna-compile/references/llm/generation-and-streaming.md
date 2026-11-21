# Generation and streaming

Read when implementing chat output, tool calls, reasoning or cancellation.

This guide assumes the chat signature introduced in [LLM authoring](authoring.md):
annotated conversation messages and a stream of typed completion chunks.

Use the model's tokenizer and chat template, with the correct generation prompt.
Normalize roles only as required by that template. Keep system messages, tools,
assistant tool calls and multimodal content in a format the model was trained on.
Do not substitute a generic `"role: content"` concatenation.

`ChatCompletionChunk`, `DeltaMessage`, `StreamChoice` and token usage schemas are
available from `muna.beta.openai`. Use a stable
completion ID, creation time and model name throughout one stream. Announce the
assistant role, emit content/tool/reasoning deltas, then a finish reason and
accurate usage. Yield typed chunks; Muna formats the HTTP stream.

In the native Transformers request-manager recipe, outputs can contain
cumulative `generated_tokens`. Track how many IDs were already consumed and
decode only the new suffix. Use incremental decoding so a
multi-byte token sequence is not damaged by decoding individual IDs independently.
Speculative generation can deliver several token IDs in one output: one output
chunk is not necessarily one token or one engine step.

Trim the model's terminal token when appropriate; don't drop tool delimiters
before parsing. Distinguish EOS, output-limit and tool-call finish reasons. Count
generated IDs consistently with the model's usage contract, including terminal
tokens where the engine reports them.

## Expose generation controls

Use `muna.beta.Annotations` to identify controls the client adapter can supply.
The function must actually apply each control to prompt preparation or generation;
the annotation only connects the client request to that parameter.

| Control | Annotation |
| --- | --- |
| Output token limit | `MaxOutputTokens` |
| Sampling temperature / nucleus probability | `Temperature`, `TopP` |
| Reproducible sampling seed | `Seed` |
| Token repetition controls | `FrequencyPenalty`, `PresencePenalty` |
| Stop strings | `StopSequences` |
| Tool definitions | `ChatTools` |
| Structured response request | `ResponseFormat` |
| Model-supported reasoning level | `ReasoningEffort` |

Give optional controls suitable defaults and validate their bounds in the model
function. Unsupported required parameters prevent an adapter match. `TopK` is
available to the Anthropic messages adapter; keep it optional in a signature also
served through OpenAI chat, which does not bind that control.

Use the SDK's token-usage schema in chunk output. Counts must reflect actual
token IDs, not text length or the number of delivered chunks. A final usage
record and an empty content delta can still be meaningful stream output.

## Tools and reasoning

Expose only implemented controls. `Annotations.ChatTools` identifies a parameter;
it does not itself render tools into a model prompt or parse generated calls.
Use a covered model-specific parser and include incomplete JSON/call fragments
and parser finalization. A non-thinking model should not invent reasoning deltas.

An OpenAI chunk-producing predictor can also match `AnthropicMessagesAdapter`.
The adapter maps input messages and output events; authors need not produce two
different native generation protocols. Verify both adapters if both are promised.

## Cleanup

For the native request-manager recipe, put manager cleanup in `finally` around
stream consumption. Other backends should use their own cancellation mechanism:

```python
try:
    for output in manager.request_id_iter(request_id=request_id):
        # Decode newly produced tokens and yield typed chunks here.
        ...
finally:
    manager.cancel_request(request_id=request_id)
```

This is an explanatory fragment; see the complete recipe for executable code.
Cancellation of a completed request is a no-op. Propagate a failed request's
error instead of returning an apparently successful truncated completion.
