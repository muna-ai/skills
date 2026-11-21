# Model and backend selection

Read before authoring a wrapper around a Hugging Face repository or changing a
model's execution path. A model architecture and its inference backend are
separate choices. NanoSGL is one supported integration, not an implicit default
for every model, PyTorch object or modality.

## Choose execution metadata for the model

1. Identify the actual architecture/configuration, checkpoint precision, processor
   and requested task. A repository name or common model family is insufficient.
2. Identify the model object the function will use: a PyTorch module, Diffusers
   pipeline, ONNX Runtime session, LiteRT interpreter or llama.cpp model.
3. Use [inference metadata](inference-metadata/index.md) to select a documented
   execution path for that object and the intended hardware. Preserve precision,
   preprocessing and output semantics when choosing between native integration
   and graph export.
4. Follow the relevant [LLM](llm/authoring.md), [embedding](embeddings/authoring.md),
   [image](images/authoring.md) or [audio](audio/authoring.md) guide for the public
   function contract and any model-specific recipe. Those contracts do not
   themselves select the execution backend.
5. Compile the actual script to establish compatibility. For a missing-operator
   failure, tell the user to raise a Muna support ticket. Do not silently change
   model, precision, context limit or semantics to make a script compile.

The metadata reference has separate guides for graph exporters, existing model
files and native integrations, including their fields and current limitations.
Use the selected guide rather than copying configuration from another backend
with a similar name. Read [targets and hardware](targets-and-hardware.md) to
separate platform selection from the model's execution requirements.

## Model identity and validation

A supported architecture does not guarantee that every fine-tune, quantized
checkpoint or custom processor is compatible. Preserve checkpoint identity and
model construction from the documented recipe. Validate changes with an actual
compilation and reference inference rather than assuming the same hidden size
makes two checkpoints interchangeable.

Pin model/tokenizer revisions where supported and record the resolved checkpoint
used for validation. Some native recipes select a fixed checkpoint revision or
resolve the current HF head during staging; setting `revision` on a configuration
loader alone does not guarantee all staged files use it. If a required revision
cannot be selected through the documented path, raise that requirement with Muna
support instead of claiming reproducibility that has not been established.
