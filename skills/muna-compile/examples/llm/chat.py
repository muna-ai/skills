"""Protocol fixture: emit an echo chunk, without loading an LLM."""

from muna import compile
from muna.beta import Annotations
from muna.beta.openai import ChatCompletionChunk, DeltaMessage, Message, StreamChoice
from time import time
from typing import Annotated, Iterator
from uuid import uuid4


@compile(targets=["x86_64-unknown-linux-gnu"])
def chat(
    messages: Annotated[
        list[Message],
        Annotations.ChatMessages(description="Conversation messages.", context_length=4096)
    ]
) -> Iterator[ChatCompletionChunk]:
    """Echo a message to exercise the chat streaming contract."""
    text = messages[-1]["content"] if messages else ""
    yield ChatCompletionChunk(
        id=f"chatcmpl-{uuid4()}",
        created=int(time()),
        model="protocol-echo",
        choices=[StreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant", content=str(text)),
            finish_reason="stop"
        )]
    )
