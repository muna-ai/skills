# Values in Muna prediction calls

Read when calling the raw Muna prediction API with arrays, images, audio or
encoded values. Inside the predictor, use the ordinary typed inputs and outputs
from [Python value support](python-support.md#supported-input-and-output-types).
That reference covers numeric widths, tensor dtypes/byte order, structured
records, images and binary data, with examples for each value category.

## Let the SDK encode values

Pass Python strings, arrays, images and other supported values to
`muna.predictions.create`. The SDK handles their encoding and decodes returned
values. See [serving and deployment](serving-and-deployment.md) for a complete
remote-call example using `acceleration="remote_auto"`.

Use this serialization rather than hand-assembling a data URL. Raw float bytes
in base64 are not interchangeable with the SDK's tensor archive: dtype, shape
and payload format must agree. This applies to raw tensor and audio requests
even when the surrounding JSON is valid.

## Reuse encoded values when chaining predictions

A `RemoteValue` represents an encoded payload or resource reference. Remote
prediction calls can accept an existing remote value without first decoding
and re-encoding it. Local prediction calls decode it before invoking the
compiled function.

Prediction results are normally decoded. Use `passthrough=True` only when you
need to retain encoded results, for example to forward a large output to another
remote prediction. Those results are `RemoteValue` objects; decode them before
performing ordinary array or image operations on their contents.
