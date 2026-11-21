# Define inputs and outputs

Read when writing a prediction function's public parameters and return value.
A signature describes those inputs and outputs so Muna and its clients know how
to call the compiled function.

## Types and defaults

Annotate every parameter and the return value. A default makes an input optional;
a union with `None` alone does not. For example, `label: str | None = None` can be
omitted, while `label: str | None` still requires an argument.

A fully typed tuple return such as `tuple[float, str]` declares separate
outputs. A synchronous generator uses `Iterator[str]` when each yielded result is a string. See [Python support](python-support.md) for supported value types,
container forms and generator restrictions.

## Add descriptive or semantic annotations

Python's `Annotated` attaches metadata to an ordinary type. Muna's `Parameter`
helpers describe the value without replacing its Python type:

```python
from muna import compile, Parameter
from typing import Annotated

@compile()
def area(
    radius: Annotated[float, Parameter.Numeric(
        description="Circle radius.",
        min=0.0
    )]
) -> float:
    """Calculate the area of a circle."""
    if radius < 0.0:
        raise ValueError("Radius must be nonnegative")
    return 3.141592653589793 * radius ** 2
```

For description-only annotations, put the type and annotation on separate lines
inside `Annotated[...]`. When adding other fields, keep
`Annotated[T, Annotation(` together and put each field on its own line.

`Parameter.Generic` adds a description to other supported values. Some helpers
also identify a value's semantic meaning, recorded as its denotation. The
modality guides introduce those meanings where needed. An annotation records
metadata; implement shape, range and semantic validation in the predictor rather
than assuming the annotation enforces it.

## When exposing an OpenAI or Anthropic interface

An adapter translates a standard API request into prediction inputs and turns
prediction outputs into that API's response. Its rules are additional to basic
Python typing. Use the relevant modality guide for the exact annotations and
response shapes: [chat](llm/authoring.md), [embeddings](embeddings/authoring.md),
[images](images/authoring.md) or [audio](audio/authoring.md).

Muna provides these semantic annotations through `muna.beta.Annotations` and
`muna.Parameter`. A parameter's name alone does not opt into an interface.
Matches must be unambiguous: unsupported required inputs or duplicate names can
prevent a match; extra optional inputs can use the predictor's defaults. If no
adapter matches, use the raw Muna prediction interface and do not present an
OpenAI example as compatible.

For concurrency settings, read [batching](batching.md) when the workload needs
it. For SDK calls with arrays, images or encoded values, read
[prediction values](prediction-values.md).
