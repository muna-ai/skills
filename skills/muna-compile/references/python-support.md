# Python language and value support

Read before translating an existing function. Language syntax, value types and
library calls must all be compatible with compilation. The rules below describe
what to write in the model script.

## Entry signatures

- Annotate every input and return value. Prefer named, fixed parameters.
- Regular functions and synchronous generator functions are supported.
  Use `Iterator[str]` for a stream of strings, or the appropriate yielded type. `async def` and async generators
  are explicitly rejected by both the decorator and signature extraction.
- A parameter is optional when it has a default. `value: str | None` without a
  default is still required. Set `= None` when callers may omit it.
- A tuple return annotation declares separate outputs. It must be fully typed,
  such as `tuple[float, str]`; a variadic `tuple[T, ...]` is rejected there.

## Supported input and output types

Use these annotations for the function's public parameters and return values,
following Muna's [supported types](https://docs.muna.ai/compile/requirements/types.md).
Types used inside the function must also support the operations performed on them.

| Value | Annotation | Important behavior |
| --- | --- | --- |
| Floating-point scalar | `float`, `np.float16`, `np.float32`, `np.float64` | Python `float` compiles to 32 bits. Use a NumPy scalar type for an explicit width. |
| Integer scalar | `int`, `np.int8`, `np.int16`, `np.int32`, `np.int64` | Python `int` compiles to a signed 32-bit integer, without Python's arbitrary precision. |
| Boolean | `bool` | A boolean scalar. |
| String | `str` | A text value. |
| Tensor / array | `numpy.typing.NDArray[T]` or `np.ndarray` | Specify the element dtype with `T`; bare `np.ndarray` assumes float32. |
| List | `list[T]` | Prefer an explicit supported element type. Lists of Pydantic models describe structured items. |
| Dictionary / record | `dict[str, T]` or a Pydantic `BaseModel` subclass | Dictionary keys must be strings. Prefer `BaseModel` for named, typed fields and a complete schema. |
| Image | `PIL.Image.Image` | A Pillow image; use `list[Image.Image]` for a list of images. |
| Binary data | `bytes`, `bytearray`, `io.BytesIO` | Encoded files or other binary payloads. |

Here `np` means `import numpy as np`. Supported enums and `Literal[...]` can
describe constrained values. Use concrete types for structured data; arbitrary
Python object instances are not supported input/output values.

### Floating-point values

Annotate ordinary floating-point inputs and outputs with `float`. Python normally
uses 64-bit floats, but a compiled Muna `float` is **32-bit**. Choose
`np.float16`, `np.float32` or `np.float64` explicitly when precision matters:

```python
from muna import compile, Sandbox
import numpy as np

@compile(sandbox=Sandbox().pip_install("numpy"))
def square(number: np.float64) -> np.float64:
    """Square a double-precision number."""
    return number * number
```

The input and output annotations each specify their own width. Changing a
parameter to `np.float64` does not make a return annotated as `float` 64-bit.

### Integer values

Use `int` for ordinary integer parameters and results. A compiled Muna `int`
is a **signed 32-bit** value, unlike Python's arbitrary-precision integers.
Choose `np.int8`, `np.int16`, `np.int32` or `np.int64` for an explicit width;
select a range suitable for both the inputs and computed results.

```python
from muna import compile, Sandbox
import numpy as np

@compile(sandbox=Sandbox().pip_install("numpy"))
def increment(number: np.int64) -> np.int64:
    """Increment a 64-bit integer."""
    return number + 1
```

### Boolean values

Use the built-in `bool` annotation for boolean inputs and outputs:

```python
from muna import compile

@compile()
def invert(enabled: bool) -> bool:
    """Invert a boolean value."""
    return not enabled
```

### Tensor values

Use `numpy.typing.NDArray[T]`, where `T` is the element dtype. A bare
`np.ndarray` is also accepted, but always declares float32 elements, regardless
of NumPy's usual defaults. Specify `T` when the interface needs another dtype.

Supported tensor element types and their Muna dtype names are:

| NumPy / Python element type | Muna dtype |
| --- | --- |
| `np.float16` | `float16` |
| `np.float32` | `float32` |
| `np.float64` | `float64` |
| `np.int8` | `int8` |
| `np.int16` | `int16` |
| `np.int32` | `int32` |
| `np.int64` | `int64` |
| `np.uint8` | `uint8` |
| `np.uint16` | `uint16` |
| `np.uint32` | `uint32` |
| `np.uint64` | `uint64` |
| `bool` | `bool` |

**Complex numbers and complex-valued tensors are unsupported.** Multi-byte
tensor elements must use **little-endian** ordering; Muna assumes that ordering
when interpreting the data.

An annotation describes the value's dtype; it does not cast the data. Return
arrays with the declared dtype, using an explicit conversion when necessary:

```python
from muna import compile, Sandbox
import numpy as np
from numpy.typing import NDArray

@compile(sandbox=Sandbox().pip_install("numpy"))
def to_float32(values: NDArray[np.float64]) -> NDArray[np.float32]:
    """Convert a float64 array to float32."""
    return values.astype(np.float32)
```

### String values

Use the built-in `str` annotation for text inputs and outputs:

```python
from muna import compile

@compile()
def uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()
```

### List values

Use `list[T]`, where `T` describes the elements. The element annotation is
optional, but strongly recommended: it provides the schema for the list's
contents. For example:

```python
from muna import compile

@compile()
def first_three(items: list[str]) -> list[str]:
    """Return the first three strings."""
    return items[:3]
```

For structured list items, use `list[YourModel]` with a Pydantic `BaseModel`
subclass. Muna generates a JSON schema including that model's fields, rather
than leaving callers to guess each item's structure.

### Dictionary and record values

Dictionary-like values can use `dict[str, T]` or a Pydantic `BaseModel` subclass.
**Dictionary keys must be strings.** Values can use supported value types;
giving the value type makes the intended contents explicit:

```python
from muna import compile

@compile()
def text_stats(text: str) -> dict[str, int]:
    """Count characters in a string."""
    return {"characters": len(text)}
```

Prefer Pydantic `BaseModel` for records with named fields: it gives Muna a full
JSON schema. Define the class at module scope, annotate its fields and use it
directly as an input or output type. Fields can also use `Literal[...]` to
describe a fixed set of choices.

```python
from muna import compile, Sandbox
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

@compile(sandbox=Sandbox().pip_install("pydantic"))
def birthday(person: Person) -> Person:
    """Add one year to a person's age."""
    return Person(name=person.name, age=person.age + 1)
```

### Image values

Use Pillow's `PIL.Image.Image` type for images:

```python
from muna import compile, Sandbox
from PIL import Image

@compile(sandbox=Sandbox().pip_install("pillow"))
def thumbnail(image: Image.Image) -> Image.Image:
    """Resize an image to 256 by 256 pixels."""
    return image.resize((256, 256))
```

This is a valid raw prediction signature. Standard image interfaces have
additional annotation and image-list requirements; see [image authoring](images/authoring.md).

### Binary values

Binary inputs and outputs can use any of these annotations:

1. `bytes` for byte sequences.
2. `bytearray` for mutable byte buffers.
3. `io.BytesIO` for in-memory binary streams.

For example, a function can accept raw pixels, process them with Pillow and
return raw pixel bytes:

```python
from muna import compile, Sandbox
from PIL import Image

@compile(sandbox=Sandbox().pip_install("pillow"))
def resize_pixels(pixels: bytes) -> bytes:
    """Resize raw grayscale pixels from 4 by 4 to 8 by 8."""
    image = Image.frombytes("L", (4, 4), pixels)
    return image.resize((8, 8)).tobytes()
```

Here the function supplies the pixel format and dimensions. Those bytes are
not an encoded PNG or JPEG file. Match the binary representation to what the
function expects; use the SDK to serialize values when making a
[prediction call](prediction-values.md).

For descriptions, ranges and modality-specific meanings, use
[signatures and annotations](signatures-and-annotations.md).

## Supported patterns and meaningful limits

| Construct | Contract / qualification |
| --- | --- |
| Assignment, annotated assignment, arithmetic, comparison, boolean expressions | Supported for compatible operand types. |
| `if`/`elif`/`else`, ternary, `for`, `while`, `break`, `continue` | Supported with compilable values and iterables. |
| Lists, tuples, dicts, comprehensions, generator expressions | Supported forms include list/set/dict comprehensions; contained values and the operations performed on them must also compile. |
| Helpers, lambdas, nested `def` | Supported when captured values have compilable types and all operations in the helper are supported. |
| `yield`, early generator return | Supported patterns; use a synchronous generator. |
| `match` | Value, singleton, OR, capture/wildcard and guarded forms are handled; sequence, mapping and class patterns are rejected. |
| `raise`, `try`/`except`, re-raise, `finally` | Supported native exception paths; not arbitrary Python exception machinery. |
| `try ... else`, `except*` | Explicitly rejected. |
| `finally` | Cannot yield or transfer control out with return/break/continue; ordinary cleanup is supported. |
| Assignment unpacking | Simple unpacking is handled; nested/starred assignment unpacking is rejected. Validate loop/comprehension unpacking separately when it uses more complex patterns. |
| Function-local imports | Resolved through the sandbox; not dynamic Python imports at inference. Prefer module-level imports. |
| Context managers (`with`), async/await, runtime class definitions | Not generally supported in the compiled prediction body; move setup out or replace with a supported pattern. A setup-time `with init_empty_weights()` is a different case. |

These constructs do not imply support for every Python metaprogramming pattern
or every library call inside them. Compile unusual forms as a small example.
When an error reports a missing operator, tell the user to raise a Muna support
ticket with that example and the compiler message.

## Avoid accidental overclaims

`list[Message]` being serializable does not make an entrypoint chat-compatible.
An annotated dtype/range does not replace model validation. An empty or
heterogeneous collection may need type propagation that a particular overload
does not support. Narrow the concrete failing expression rather than rewriting
all preprocessing unnecessarily.
