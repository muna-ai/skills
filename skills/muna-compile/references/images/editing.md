# Image editing

Read when exposing a function through `openai.images.edit`. The function receives
reference images and a text instruction, then returns edited images.

## Declare the interface

Use `list[Image.Image]` for the image input and annotate it with
`Annotations.ChatImages`, which marks the list as the request's reference images.
Use a string annotated with `Annotations.ImagePrompt` for the instruction. The
text may default to `""` when the model has no text conditioning.

Both annotations are required for matching. There is no fallback to a parameter
named `images` or `prompt`. A single `Image.Image` input does not match: use a
list even when a request contains only one image. Return `list[Image.Image]`.
See the complete [identity-edit fixture](../../examples/images/edit.py).

## Optional controls

Optional width, height and count use the same rules as [image generation](generation.md).
Additional required controls prevent matching unless the adapter binds them.
Other optional inputs can use predictor defaults, but that does not make them
available as endpoint request options.

The server handles supported image references and multipart uploads. Keep
HTTP parsing and file/URL decoding out of the model function unless they are
explicitly part of the raw-function task.
