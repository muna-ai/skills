# Depth and disparity outputs

Read for a model that produces numeric spatial maps from images. Muna exposes
this through its image-edit extensions, with a different output contract from
an ordinary edited image.

## Declare inputs and outputs

Use a list of PIL images annotated with `Annotations.ChatImages` for the
reference images, plus a string annotated with `Annotations.ImagePrompt` for the
instruction. The prompt can default to `""` if unused. Names alone do not match
the interface, and a single image must still be passed through a list parameter.
See [image editing](editing.md) for the shared input rules.

Return a `list[ndarray]` annotated with `Parameter.DepthMap`. The
`DepthMap` annotation identifies the output's meaning. Describe whether values
are relative depth, metric depth or disparity, their units/range, and how array
dimensions relate to the input image. The [depth fixture](../../examples/images/depth.py)
shows this interface without claiming to estimate real depth.

## Preserve model-specific behavior

Lotus, DA3 and Marigold differ in checkpoints, topology, precision, normalization
and output semantics. The DA3 nested Giant/Large integration requires `sm_100`,
single-GPU FP32, its pinned constructor topology and checkpoint/source identities.
Dimensions must be multiples of 14 in [14,1022]. This does not establish support
for a different DA3 configuration or a Hopper target.

## Encode numeric results

Request a supported array format such as `npy` through Muna's output option.
`output_format` is a top-level argument in the Muna SDK. With the official OpenAI
SDK, use its extension mechanism, for example
`extra_body={"output_format": "npy"}`.

Decode the returned bytes with `numpy.load` and preserve the raw array. An
automatically colorized image would lose the numeric result. Avoid converting
disparity to metric depth unless the model's output contract supplies that meaning.
