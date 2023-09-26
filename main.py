import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from llms.zhipu.zhipu_sdk import sdk_sse_invoke

app = FastAPI()


def event_stream(prompt):
    prompts = [{"role": "user", "content": f"{prompt}"}]
    response = sdk_sse_invoke(prompts=prompts)  # Invoke SSE and get the response object
    for event in response.events():
        yield event.data
        # time.sleep(1)  # Add a delay between events (for demonstration purposes)


@app.get("/events/{prompt}")
async def events(prompt: str):
    headers = {"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"}
    return StreamingResponse(event_stream(prompt), headers=headers)


if __name__ == "__main__":
    # Start the FastAPI app
    uvicorn.run(app, host="127.0.0.1", port=8000)
