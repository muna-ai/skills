# 3D asset outputs

Read for a reconstruction model that returns a mesh, point cloud or Gaussian
splat asset. Muna exposes this through an image-edit extension.

## Declare inputs and outputs

Use reference images as `list[Image.Image]` annotated with
`Annotations.ChatImages`, plus a string annotated with `Annotations.ImagePrompt`.
The prompt can default to `""` if unused. Both annotations identify the inputs;
parameter names alone do not match. Use a list even for a single reference image.
See [image editing](editing.md) for these shared input rules.

Return bytes annotated with `Parameter.Model3D`, for example:

```python
# Return annotation on the prediction function; the body must produce real PLY bytes.
Annotated[
    bytes,
    Parameter.Model3D(description="PLY file containing 3D Gaussian splats.")
]
```

`Model3D` identifies a 3D asset and deliberately does not specify its file format.
Describe the actual representation in the output description. Do not add a format
field to the annotation or assume every 3D result is PLY.

## Format and validate the result

`output_format="ply"` requests a representation from the serving adapter. The
predictor must actually return valid PLY for that use; an annotation or filename
does not convert arbitrary bytes into a splat asset. Use the SDK extension
mechanism appropriate to the client when passing this Muna-specific option.

Test the file header, element/property layout and an actual 3D consumer for a
real reconstruction model. The [PLY fixture](../../examples/images/model3d.py)
returns a valid empty asset to test the interface only.
