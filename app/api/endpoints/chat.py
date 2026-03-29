import json

from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.core.logger import get_logger
from app.services.chat.agent import chat_agent

logger = get_logger("chat_endpoint")

chat_router = APIRouter()

@chat_router.post("/agentic")
def rag_query(query: str):
    logger.info("Received agentic query", extra={"query": query})
    def event_stream():
        try:
            for event in chat_agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            ):
                message = event["messages"][-1]
                data_to_send = None

                if message.type == "ai":
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        data_to_send = {
                            "type": "tool_calls",
                            "tools": [{"name": t["name"], "args": t.get("args", {})} for t in message.tool_calls]
                        }
                        logger.debug("Yielding tool calls", extra={"tools": data_to_send["tools"]})
                    elif message.content:
                        data_to_send = {
                            "type": "answer",
                            "content": message.content
                        }
                        logger.info("Yielding final answer", extra={"content_length": len(message.content)})
                elif message.type == "tool":
                    data_to_send = {
                        "type": "tool_result",
                        "name": message.name,
                        # We avoid sending huge raw content to frontend if it's too large, but for now we stream it
                        "content": message.content
                    }
                    logger.debug("Yielding tool result", extra={"tool_name": message.name})
                
                if data_to_send:
                    yield f"data: {json.dumps(data_to_send)}\n\n"
        except Exception as e:
            logger.error("Error generating agent stream", exc_info=True, extra={"query": query})
            yield f"data: {json.dumps({'type': 'error', 'content': 'Internal Server Error'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")