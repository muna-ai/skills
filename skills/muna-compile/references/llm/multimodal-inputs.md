# Multimodal LLM inputs

Read when chat messages contain images or audio. Do not apply image-generation
prompt conventions to a vision-language chat model.

The chat adapters bind conversational messages and can extract image/audio
attachments into separately annotated predictor parameters. Use:

- `Annotations.ChatMessages` for the conversation, with its positive `context_length`.
- `Annotations.ChatImages` on `list[PIL.Image.Image]` for images.
- `Annotations.ChatAudios(sample_rate=...)` on a supported `list[ndarray]` input
  for decoded PCM audio; the positive sample rate is required for matching.

Preserve ordering between message placeholders and attachment arrays. Use the
specific model processor, resolution/cropping policy, special tokens and chat
template. A tokenizer-only path is insufficient when the native model needs
vision features or expanded image-token spans.

Processor compatibility is model-specific. The supported preparation paths
include Gemma4, Qwen3-VL, GLM5-next, DeepSeek V4.1 and Whisper processors; this
does not imply arbitrary `AutoProcessor` support or every task for each model.
Use the corresponding model recipe, and report unsupported preparation to Muna
support if compilation fails.

For KV routing, reuse the exact processor preparation function. Return its
supported processor output when needed so image-content identity participates in
routing decisions. Counting identical image placeholder IDs for different pictures
does not establish reusable visual KV state.

Use a real multimodal reference input during validation. A successful text-only
request does not prove image/audio preprocessing, attachment ordering or vision
weight staging is correct.
