# Image generation

Read for `openai.images.generate` compatibility.

```python
def generate(
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Image prompt.")
    ]
) -> list[Image.Image]:
    ...
```

This fragment shows the signature only. See the executable
[protocol fixture](../../examples/images/generation.py).

The prompt annotation is mandatory. A required `str` or a parameter named
`prompt` without `ImagePrompt` does not opt in. Return an image **list**, even
for one generated image. Matching needs one unambiguous image-list output.
Optional width, height and count can use `ImageWidth`,
`ImageHeight`, `ImageCount`; required versions of those controls reject adapter
matching. Reject unsupported dimensions/count in the predictor rather than
silently changing them.

The native Diffusers path supports the FLUX.2 Klein 9B pipeline on a single GPU;
that does not extend to every FLUX variant or text-to-image model or imply
tensor-parallel support. Other supported diffusion models have their own
checkpoint and construction requirements.

Preserve text encoder/tokenizer limits, prompt padding, component precision,
latents, RNG behavior, denoising schedule, VAE scaling and image conversion.
A prompt length or image-size cap should reflect the selected backend, not
the most permissive Python reference pipeline setting.

For an official OpenAI SDK consumer, image data may be returned as base64. Use
`b64decode` and `Path.write_bytes` in a deployment example; no image display
dependency is necessary. Do not claim OpenAI's general Responses API support
from the images adapter alone.
