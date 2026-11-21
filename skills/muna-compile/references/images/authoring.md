# Image workloads

Read for image generation, editing, depth or 3D reconstruction. Start with the
input/output task, then follow only its guide.

| Workload | Guide |
| --- | --- |
| Text to image | [Generation](generation.md) |
| Reference images and an edit instruction to images | [Editing](editing.md) |
| Image to depth or disparity arrays | [Depth](depth.md) |
| Image to a 3D asset | [3D outputs](model3d.md) |

For images inside a language conversation, use [multimodal LLM inputs](../llm/multimodal-inputs.md).
The task-specific guides introduce the annotations and response shapes needed by
the corresponding interface.

## Signature examples

Choose the signature for the intended image interface. These examples omit
model bodies and build configuration; replace `...` before compiling. A different
supported signature can still be called through the Muna prediction API.

### Image generation

```python
from muna import compile
from muna.beta import Annotations
from PIL import Image
from typing import Annotated

@compile()
def generate(
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Image prompt.")
    ]
) -> list[Image.Image]:
    """Generate images from a prompt."""
    ...
```

For Muna to recognize this image-generation interface:

1. Declare one required string prompt annotated with `Annotations.ImagePrompt`.
   An unannotated string or a parameter named `prompt` is insufficient.
2. Return one unambiguous `list[Image.Image]` output, including for a single image.
3. Give additional inputs defaults. Optional width, height and count controls
   need the annotations described in [generation](generation.md).

### Image editing

```python
from muna import compile
from muna.beta import Annotations
from PIL import Image
from typing import Annotated

@compile()
def edit(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Reference images.")
    ],
    *,
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Edit instruction.")
    ]=""
) -> list[Image.Image]:
    """Edit reference images."""
    ...
```

For Muna to recognize this image-editing interface:

1. Declare a `list[Image.Image]` input annotated with `Annotations.ChatImages`.
   A single-image parameter does not match, even for one reference image.
2. Declare a string annotated with `Annotations.ImagePrompt`. It can default
   to `""` when unused; a text-conditioned model should require the prompt it needs.
3. Return one unambiguous `list[Image.Image]` output.
4. Keep annotated inputs unambiguous and give additional inputs defaults.
   See [editing](editing.md) for optional controls.

### Depth or disparity

```python
from muna import compile, Parameter
from muna.beta import Annotations
from numpy import ndarray
from PIL import Image
from typing import Annotated

@compile()
def depth(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Reference images.")
    ],
    *,
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Depth instruction.")
    ]=""
) -> Annotated[
    list[ndarray],
    Parameter.DepthMap(description="Relative depth maps.")
]:
    """Predict relative depth maps."""
    ...
```

For Muna to recognize this depth interface:

1. Use the same image-list and annotated string-prompt inputs as image editing,
   with no additional required inputs.
2. Return one unambiguous list of arrays annotated with `Parameter.DepthMap`.
3. Produce float32 or float64 maps and describe whether values represent depth
   or disparity, including units and their relationship to the input images.
   See [depth outputs](depth.md) for numeric response formats.

### 3D reconstruction

```python
from muna import compile, Parameter
from muna.beta import Annotations
from PIL import Image
from typing import Annotated

@compile()
def reconstruct(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Reference images.")
    ],
    *,
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="3D instruction.")
    ]=""
) -> Annotated[
    bytes,
    Parameter.Model3D(description="PLY file containing 3D Gaussian splats.")
]:
    """Reconstruct a 3D asset from reference images."""
    ...
```

For Muna to recognize this 3D interface:

1. Use the same image-list and annotated string-prompt inputs as image editing,
   with no additional required inputs.
2. Return one unambiguous binary output annotated with `Parameter.Model3D`.
3. Describe the actual file representation in the output description. For the
   PLY example, return valid PLY bytes. The annotation has no file-format field.
   See [3D outputs](model3d.md) for response-format and validation details.

## Choose the model execution path

Choose [inference metadata](../inference-metadata/index.md) for the model or
pipeline. An exported graph can use a backend appropriate to its target; a native
NanoSGL integration has its own supported model/pipeline construction. Image
annotations and output formats do not select either path or imply a GPU.
Preserve checkpoint identity, topology, preprocessing, precision, denoising
behavior and output semantics.

For native Diffusers execution, follow
[DiffusersToSGLangInferenceMetadata](../inference-metadata/sglang.md). Its
architecture and batch settings configure that pipeline; LLM-specific cache and
speculative-generation settings do not apply. Use the supported construction
before replacing real weights with configuration-only setup. A custom Python
pipeline with familiar components is not automatically interchangeable with a
supported native integration.

For NanoSGL shared-GPU memory pressure or a slow cold load, read
[memory and colocation](../memory-and-colocation.md) or [deployment lifecycle](../deployment-lifecycle.md).
Neither is a prerequisite for writing a basic image function.
