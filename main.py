import asyncio
from typing import Awaitable

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler

from llms.knowledge_base import get_answer
from llms.zhipu.zhipu_knowledge_base import get_zhipu_answer
from llms.zhipu.zhipu_sdk import sdk_sse_invoke

app = FastAPI()


def event_stream(prompt):
    prompts = [{"role": "user", "content": f"{prompt}"}]
    response = sdk_sse_invoke(prompts=prompts)  # Invoke SSE and get the response object
    for event in response.events():
        yield event.data
        # time.sleep(1)  # Add a delay between events (for demonstration purposes)


async def answer(question):
    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

    callback = AsyncIteratorCallbackHandler()

    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(
        get_answer(question, callback), callback.done),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        yield f"data: {token}\n\n"

    await task


@app.get("/events/{prompt}")
async def events(prompt: str):
    headers = {"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"}
    return StreamingResponse(event_stream(prompt), headers=headers)


@app.get("/zhipu/{prompt}")
async def events(prompt: str):
    headers = {"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"}
    return StreamingResponse(get_zhipu_answer(prompt), headers=headers)


@app.get("/llm/{prompt}")
async def llm(prompt: str):
    headers = {"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"}
    return StreamingResponse(answer(prompt), headers=headers)


if __name__ == "__main__":
    # Start the FastAPI app
    uvicorn.run(app, host="127.0.0.1", port=8000)
