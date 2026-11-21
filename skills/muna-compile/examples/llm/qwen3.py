"""HF recipe: text-only Qwen3-4B, without tools, reasoning or a drafter.

This recipe constructs the model on the meta device rather than loading weights
for eager Python inference. Importing it can access Hugging Face. Compile for
Hopper (sm_90), then validate the compiled model with representative prompts.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate", "muna", "tokenizers==0.22.2", "torch==2.13.0",
#   "transformers==5.15.0"
# ]
# ///

from accelerate import init_empty_weights
from muna import compile, BatchConfig, Sandbox
from muna.beta import Annotations, KVRoutingMetadata, TorchToSGLangInferenceMetadata
from muna.beta.openai import (
    ChatCompletion, ChatCompletionChunk, DeltaMessage, Message, StreamChoice
)
from os import environ
from time import time
from tokenizers.decoders import DecodeStream
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM
from transformers.generation import ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching import RequestStatus
from typing import Annotated, Iterator
from uuid import uuid4


CHECKPOINT = "Qwen/Qwen3-4B-Instruct-2507"
MAX_TOTAL_TOKENS = 8192
config = AutoConfig.from_pretrained(CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
with init_empty_weights():
    model = Qwen3ForCausalLM._from_config(config, attn_implementation="eager")

STOP_TOKENS = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
    tokenizer.convert_tokens_to_ids("<|endoftext|>")
]
manager = model.init_continuous_batching(
    generation_config=GenerationConfig(
        max_new_tokens=512,
        eos_token_id=STOP_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.8
    ),
    continuous_batching_config=ContinuousBatchingConfig(
        per_request_processors=True,
        use_cuda_graph=True
    )
)


def _tokenize(messages: list[Message]) -> list[int]:
    # Use the same prompt preparation for generation and KV routing.
    prompt = [
        {"role": "system", "content": message["content"]}
        if message["role"] == "developer" else message
        for message in messages
    ]
    text = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False
    )
    return tokenizer.encode(text, add_special_tokens=False)


@compile(
    targets=["x86_64-unknown-linux-gnu"],
    sandbox=(
        Sandbox()
        .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("accelerate", "transformers==5.15.0", "tokenizers==0.22.2")
        .env({"HF_TOKEN": environ.get("HF_TOKEN")})
    ),
    metadata=[
        TorchToSGLangInferenceMetadata(
            model=model,
            compute_architecture="sm_90",
            tensor_parallelism=1,
            max_running_requests=4,
            max_total_tokens=MAX_TOTAL_TOKENS
        ),
        KVRoutingMetadata(tokenize=_tokenize)
    ]
)
def qwen3(
    messages: Annotated[list[Message], Annotations.ChatMessages(
        description="Text-only conversation messages.",
        context_length=MAX_TOTAL_TOKENS,
        batch=BatchConfig(mode="continuous")
    )],
    *,
    max_output_tokens: Annotated[int, Annotations.MaxOutputTokens(
        description="Maximum output tokens.", min=1, max=MAX_TOTAL_TOKENS
    )] = 512,
    temperature: Annotated[float, Annotations.Temperature(
        description="Sampling temperature.", min=0.0, max=2.0
    )] = 0.7,
    top_p: Annotated[float, Annotations.TopP(
        description="Nucleus probability.", min=0.0, max=1.0
    )] = 0.8
) -> Iterator[ChatCompletionChunk]:
    """Generate text-only chat using Qwen3-4B on Hopper."""
    input_ids = _tokenize(messages)
    if max_output_tokens < 1 or len(input_ids) + max_output_tokens > MAX_TOTAL_TOKENS:
        raise ValueError("Input plus output limit exceeds the configured context")
    if temperature < 0.0 or temperature > 2.0 or top_p <= 0.0 or top_p > 1.0:
        raise ValueError("Invalid sampling parameters")
    request_id = f"chatcmpl-{uuid4()}"
    created = int(time())
    manager.add_request(
        input_ids=input_ids,
        request_id=request_id,
        streaming=True,
        max_new_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p
    )
    try:
        decoder = DecodeStream(skip_special_tokens=False)
        consumed = 0
        role_sent = False
        for output in manager.request_id_iter(request_id=request_id):
            if output.status == RequestStatus.FAILED:
                raise RuntimeError(output.error or "Generation failed")
            if not role_sent:
                yield _chunk(request_id, created, DeltaMessage(role="assistant", content=""))
                role_sent = True
            finished = output.status == RequestStatus.FINISHED
            new_tokens = output.generated_tokens[consumed:]
            if finished and new_tokens and new_tokens[-1] in STOP_TOKENS:
                new_tokens = new_tokens[:-1]
            text = ""
            for token in new_tokens:
                piece = decoder.step(tokenizer.backend_tokenizer, token)
                if piece is not None:
                    text += piece
            consumed = len(output.generated_tokens)
            if text:
                yield _chunk(request_id, created, DeltaMessage(content=text))
            if finished:
                reason = "length" if consumed >= max_output_tokens else "stop"
                yield _chunk(
                    request_id, created, DeltaMessage(content=""), reason,
                    ChatCompletion.Usage(
                        prompt_tokens=len(input_ids),
                        completion_tokens=consumed,
                        total_tokens=len(input_ids) + consumed
                    )
                )
    finally:
        manager.cancel_request(request_id=request_id)


def _chunk(
    request_id: str,
    created: int,
    delta: DeltaMessage,
    finish_reason: str | None = None,
    usage: ChatCompletion.Usage | None = None
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=request_id,
        created=created,
        model=CHECKPOINT,
        choices=[StreamChoice(index=0, delta=delta, finish_reason=finish_reason)],
        usage=usage
    )
