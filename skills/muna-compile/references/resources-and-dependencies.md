# Dependencies and build inputs

Read when the model script needs Python packages, local modules, checkpoint files
or credentials during compilation. Muna uses a build environment called a
`Sandbox`; it is separate from the developer's local Python environment.

## Prepare both Python environments

The CLI imports the script locally before submitting it for compilation. Install
its authoring dependencies locally, and declare the remote dependencies in the
script's `sandbox` setting:

```python
from muna import Sandbox

sandbox = Sandbox().pip_install("numpy")
# Pass sandbox=sandbox to the @compile(...) decorator.
```

PEP 723 dependency comments can help a script runner install local dependencies;
they do not replace the `Sandbox` declaration. Installing a package for the build
does not make every API in that package compilable. See [library support](library-support.md)
if a particular call fails.

Choose versions matching the model recipe. Do not copy an LLM's Transformers
pins onto a diffusion or audio script with different API requirements.

## Include local modules and files

`Sandbox` provides `workdir`, `upload_file`, `upload_directory`, `apt_install`
and `run_commands` for build setup. Upload local modules or data files the script
imports; uploading the entrypoint does not package an arbitrary project directory.
Paths used during the remote build must exist inside that build environment.

For helpers in additional Python modules, `@compile(trace_modules=[...])` controls
which modules' source can be compiled. It does not provide missing C extensions
or runtime library support. An uploaded file and a compilable operation solve
different requirements.

## When downloading a Hugging Face checkpoint

Follow the modality recipe's construction pattern. Some native integrations can
prepare model configuration and tensor shapes without loading the weights during
import; other exporters need a real model or sample tensors. Use the pattern
supported by the chosen backend rather than replacing the model with a dummy
object. [Model selection](models-and-backends.md) explains that distinction.

For gated checkpoints, make HF access available both locally and in the sandbox:

```python
from muna import Sandbox
from os import environ

sandbox = Sandbox().env({"HF_TOKEN": environ.get("HF_TOKEN")})
```

Reference environment variables instead of embedding credentials in source or
support logs. Keep checkpoint identity and revision information reproducible.

## When an integration needs explicit packaged resources

Most model recipes stage their required files automatically. If an existing
integration asks for resource descriptors rather than an ordinary file, read
[compile resources](compile-resources.md). For slow deployment preparation or
first requests, read [deployment readiness](deployment-lifecycle.md).
