---
name: muna-compile
description: Author or adapt Python functions and Hugging Face model scripts for Muna compilation and inference. Use for compiler compatibility, model initialization, signatures, annotations, resources, hardware targets, streaming and deployment.
---

# Author a Muna Predictor

A predictor is a typed Python function compiled by Muna for native inference.
Help the developer author, compile, deploy and call their model script while
preserving its intended behavior. Focus on code, configuration, supported
features and observable results. Include platform details only when they change
what the developer can build or needs to do.

Use the references below as needed; do not load every modality or advanced guide.

## Start with the user's input

- **Hugging Face model:** identify the architecture, checkpoint, task and intended
  hardware. Read [model and backend selection](references/models-and-backends.md),
  then choose a modality guide below.
- **Python function:** read the [compilation model](references/compilation-model.md)
  to separate setup from per-request work. Use [Python support](references/python-support.md)
  for language features and restrictions.
- **Existing application or inference service:** use [migration](references/migrating-python.md)
  to extract the model setup and prediction function from application infrastructure.

## Choose the relevant modality

| Workload | Guide |
| --- | --- |
| Language generation or conversational models | [LLM authoring](references/llm/authoring.md) |
| Text embeddings | [Embedding authoring](references/embeddings/authoring.md) |
| Image generation, editing, depth or 3D | [Image authoring](references/images/authoring.md) |
| Transcription or speech synthesis | [Audio authoring](references/audio/authoring.md) |
| Other Python or numerical functions | [Python support](references/python-support.md) and [library support](references/library-support.md) |

Each modality guide introduces its own interface and links to optional features.
Follow only the branches relevant to the requested model.

## Author and validate

1. **Prepare the script.** Follow the [compilation model](references/compilation-model.md)
   for the function entrypoint and initialization. Read [dependencies and resources](references/resources-and-dependencies.md)
   when adding packages, checkpoints or local files.
2. **Define inputs and outputs.** Use [signatures and annotations](references/signatures-and-annotations.md)
   for types, defaults and descriptions. The modality guide supplies any interface-specific requirements.
3. **Choose execution and platform.** For model-backed functions, select
   [inference metadata](references/inference-metadata/index.md) for the model and
   intended hardware. Use [targets and hardware](references/targets-and-hardware.md)
   for platform selection. Preserve the requested model precision and behavior.
4. **Check the result.** Follow [validation](references/validation.md). Distinguish
   successful compilation, interface compatibility and correct model inference
   when reporting results. Use [troubleshooting](references/troubleshooting.md) for failures.
5. **Deploy when requested.** Follow [serving and deployment](references/serving-and-deployment.md)
   after the script is ready. Report the checks actually performed and any remaining limitations.

## Missing support

If compilation reports a missing operator, tell the user to raise a Muna support
ticket with the failing example, compiler error and requested configuration.
Preserve the intended model and behavior when explaining the limitation.

These references are self-contained. Explain uncertainty instead of inventing
configuration or silently changing the requested model to bypass a limitation.
