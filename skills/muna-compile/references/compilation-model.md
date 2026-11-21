# Compilation model

Read when creating a model script or deciding which parts of an application
belong in its prediction function.

## Start with one function

The entrypoint is the function that Muna compiles and clients call. Mark one
function in a module with the `compile` decorator:

```python
from muna import compile

@compile()
def area(radius: float) -> float:
    """Calculate the area of a circle."""
    return 3.141592653589793 * radius ** 2
```

The CLI discovers a decorated function when importing the module. Use one
entrypoint per module; imported decorated functions can also be discovered.
Without an explicit tag, the CLI derives `@<signed-in-user>/<function-name>`
with underscores replaced by hyphens. Supply a short docstring or decorator
`description`; an explicit description is limited to 100 characters.

See [signatures and annotations](signatures-and-annotations.md) when defining
more involved inputs, outputs or defaults.

## What executes where

1. The CLI imports the module on the developer's machine to discover the function.
2. A remote build environment installs the declared dependencies and imports it again.
3. The compiler reads the function and supported helpers, converts their operations
   into native code, and packages the model files and libraries required to run it.
4. At inference, the compiled predictor executes without an embedded Python runtime.

The compiler examines the function's source, including its branches; it does not
compile only the path exercised by one representative request. Installing a
Python package during the build does not imply all of its APIs can run in the
compiled function. See [library support](library-support.md) for that boundary.

## Initialization versus per-request work

Place persistent model/tokenizer setup at module scope so it can be reused
across requests rather than recreated for every prediction. Keep request-dependent
preprocessing, inference and postprocessing inside the function or its helpers.
Arbitrary mutable Python objects are not automatically supported persistent state.

Module-level code runs during both imports. The decorator does not defer a
`from_pretrained` call or a weight download. Follow the modality recipe for model
construction; some paths support configuration-only models without loading
weights, while other exporters need real tensors or a model file.

Keep local demos under `if __name__ == "__main__":`. For packages, uploaded files
and remote setup, read [dependencies and resources](resources-and-dependencies.md).
For an existing application, use the [migration guide](migrating-python.md).

## Additional configuration

Use [inference metadata](inference-metadata/index.md) to select how each model
object is compiled and which hardware its execution path uses. This is separate
from `targets`, which selects [build platforms](targets-and-hardware.md). Plain
functions without a model backend may need no inference metadata.

The [model/backend guide](models-and-backends.md) helps choose an execution path;
modality guides describe the function's input/output contract. Follow
[validation](validation.md) before claiming the resulting combination works.
