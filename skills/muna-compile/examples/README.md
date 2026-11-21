# Authoring examples

Each file contains one decorated entrypoint. Protocol fixtures are ordinary
executable Python functions with deliberately simple behavior, not neural model
implementations. They let you inspect a contract without downloading weights.

| File | Behavior / purpose |
| --- | --- |
| [function.py](function.py) | CPU circle area; no adapter matches. |
| [llm/chat.py](llm/chat.py) | One-chunk echo completion; OpenAI and Anthropic contracts. |
| [embeddings/embeddings.py](embeddings/embeddings.py) | Synthetic vectors; embedding contract, no fabricated tokenizer usage. |
| [images/generation.py](images/generation.py) | Blank image generation; annotated prompt and image-list output. |
| [images/edit.py](images/edit.py) | Identity image edit; list inputs, annotated optional prompt. |
| [images/depth.py](images/depth.py) | Small synthetic depth arrays; no depth-estimation claim. |
| [images/model3d.py](images/model3d.py) | Valid empty ASCII PLY fixture; no reconstruction claim. |
| [audio/transcription.py](audio/transcription.py) | Returns sample count and actual input duration; no speech recognition. |
| [audio/speech.py](audio/speech.py) | Silence; speech SDK adapter only, no HTTP route claim. |
| [llm/qwen3.py](llm/qwen3.py) | Actual HF Qwen3-4B text-only recipe with shared continuous batching. |
| [embeddings/nomic.py](embeddings/nomic.py) | Actual HF Nomic embedding recipe using the native embedding path. |

The fixtures need `muna`, NumPy, Pillow and Pydantic in the local environment.
Their `Sandbox` declarations cover their build dependencies. HF recipes declare
additional local/sandbox dependencies; module import can access HF. Read them
before importing. Compile and test an HF recipe on the intended hardware before
relying on its outputs or performance; the local contract tests do not run the
neural models.

Use a fixture as an I/O pattern, then replace its body with a supported model
operation. Keep annotation names, list shapes and output schemas intact. Use
`muna compile <file>` to submit a compilation. The simple protocol examples can
run locally without submitting a build or provisioning a GPU.
