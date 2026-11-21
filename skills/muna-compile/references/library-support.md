# Python libraries in compiled predictors

Read when choosing preprocessing, inference and postprocessing code. Installing
a package in `Sandbox` makes it available during compilation; it does not embed
that package or CPython into the compiled predictor.

## Use libraries in the context of the workload

- **NumPy and PyTorch:** use arrays/tensors for numerical preprocessing and
  postprocessing around the selected model backend. Keep dtypes, shapes and
  layouts explicit when they affect results. Support for a library does not mean
  every function, argument combination or dtype is available on every target.
- **PIL, torchvision and OpenCV:** use image processing appropriate to the model.
  Preserve color space, channel ordering, interpolation, normalization and image
  size. A different resize or conversion can change model behavior even when it
  compiles successfully.
- **Transformers and tokenizers:** follow the [LLM guide](llm/authoring.md) for
  model construction, chat templates, streaming decoding and continuous batching.
  Native model execution and tokenizer/processor support are distinct; a familiar
  `AutoModel` or `AutoProcessor` class is not a guarantee of compatibility.
- **SentenceTransformers:** follow [embedding authoring](embeddings/authoring.md)
  for supported model pipelines, pooling, normalization and task prefixes. Do not
  apply LLM decoder configuration to an embedding engine.
- **Diffusers:** follow [image authoring](images/authoring.md) for supported
  pipelines and their construction patterns. A custom pipeline needs compatible
  components, checkpoint identity and inference behavior; arbitrary Diffusers
  subclasses are not interchangeable.
- **SGLang parsers:** supported model scripts can use reasoning/tool parsers
  alongside native inference. Installing the SGLang package does not make an
  arbitrary SGLang server application compilable.
- **Standard library:** ordinary data processing can use supported strings,
  containers, math and serialization. The predictor remains a compiled function,
  not a general Python application server. See [Python support](python-support.md)
  for language restrictions.

Use the supplied examples as complete patterns for their particular workloads.
Keep model-specific preprocessing and output semantics intact rather than
replacing them with approximate code merely to get through compilation.

## If compilation reports a missing operator

Tell the user to **raise a Muna support ticket**. Provide the smallest script
that reproduces the error, the complete compiler message, the target/compute
architecture and SDK/compiler version when available. Keep the requested model
and behavior intact. Include the declared dependencies and inference metadata
so support can reproduce the failure.

Compilation is the check for a particular call that is not covered explicitly in
these guides. Distinguish a missing-operator failure from a missing package,
incorrect annotation or unsupported Python construct; those have different
remedies described in [troubleshooting](troubleshooting.md).
