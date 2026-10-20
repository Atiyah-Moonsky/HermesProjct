
import os
import json
import time
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Local AI Agent API")

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "agent-api",
    }


LLAMA_URL = os.getenv("LLAMA_URL", "http://llama-server:8080/v1")
WORKSPACE = Path(os.getenv("AGENT_WORKSPACE", "/workspace")).resolve()
MODEL = os.getenv("MODEL_NAME", "Qwen-local")
def safe_path(relative_path: str) -> Path:
    """Resolve a path and prevent access outside the workspace."""
    candidate = (WORKSPACE / relative_path).resolve()

    try:
        candidate.relative_to(WORKSPACE)
    except ValueError:
        raise ValueError("Access outside workspace is not allowed")

    return candidate
MAX_STEPS = 2


SYSTEM_PROMPT = """
You are a helpful local AI agent that can use workspace tools.

TOOL CALL RULES:
- When you need a tool, return exactly ONE valid JSON object.
- Do not use Markdown fences around tool-call JSON.
- Do not put explanations before or after a tool-call JSON object.
- Use double quotes for all JSON keys and string values.
- Never return multiple tool calls in one response.
- Wait for the actual tool result before deciding what to do next.
- Never claim a file was created, read, edited, or tested unless the tool result confirms it.
- If a tool returns an error, explain the error honestly.
- If you do not need a tool, answer normally in natural language.
- After receiving tool results, use them to decide the next step.
- If the requested work is complete, return a finish tool call with a concise answer.

AVAILABLE TOOLS:

1. List files:
{"tool":"list_files","path":"."}

2. Read a file:
{"tool":"read_file","path":"example.txt"}

3. Search text:
{"tool":"search_text","query":"keyword","path":"."}

4. Create a new file:
{"tool":"write_file","path":"draft.py","content":"print('Hello')"}

5. Edit an existing file:
{"tool":"edit_file","path":"draft.py","old_text":"Hello","new_text":"Agent"}

6. Finish and report:
{"tool":"finish","answer":"Describe the verified result here"}

WORKSPACE SAFETY:
- All paths must remain inside the workspace.
- Never access or modify files outside the workspace.
- write_file creates a new file only. Never overwrite an existing file.
- edit_file requires old_text to match exactly once and creates a backup before changing the file.
- Use write_file or edit_file only when the user requests file changes.
- Do not invent tools that are not listed above.
- Be concise, accurate, and honest.
"""



def run_tool(call: dict) -> str:
    tool = call.get("tool", "")

    if tool == "list_files":
        folder = safe_path(call.get("path", "."))

        if not folder.is_dir():
            return "Error: folder not found"

        items = []

        for item in sorted(folder.iterdir()):
            kind = "DIR " if item.is_dir() else "FILE"
            items.append(f"{kind} {item.name}")

        return "\n".join(items) or "(empty folder)"

    if tool == "read_file":
        path = safe_path(call.get("path", ""))

        if not path.is_file():
            return "Error: file not found"

        if path.stat().st_size > 200_000:
            return "Error: file too large to read"

        return path.read_text(encoding="utf-8", errors="replace")

    if tool == "search_text":
        query = call.get("query", "")
        folder = safe_path(call.get("path", "."))

        if not query or not folder.is_dir():
            return "Error: query or folder is invalid"

        matches = []

        for path in folder.rglob("*"):
            if len(matches) >= 50:
                break

            if not path.is_file():
                continue

            try:
                if path.stat().st_size > 200_000:
                    continue

                lines = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()

                for number, line in enumerate(lines, 1):
                    if query.lower() in line.lower():
                        matches.append(
                            f"{path.relative_to(WORKSPACE)}:"
                            f"{number}: {line[:300]}"
                        )

                        if len(matches) >= 50:
                            break

            except OSError:
                pass

        return "\n".join(matches) or "No matches found"

    if tool == "write_file":
        relative_path = call.get("path", "")
        content = call.get("content", "")

        if not isinstance(content, str):
            return "Error: content must be text"

        if len(content.encode("utf-8")) > 200_000:
            return "Error: content too large"

        path = safe_path(relative_path)

        if path.exists():
            return "Error: file already exists; refusing to overwrite"

        if not path.parent.is_dir():
            return "Error: parent folder does not exist"

        try:
            with path.open("x", encoding="utf-8", newline="") as file:
                file.write(content)
        except FileExistsError:
            return "Error: file already exists"

        return f"Created file: {path.relative_to(WORKSPACE)}"

    if tool == "edit_file":
        relative_path = call.get("path", "")
        old_text = call.get("old_text")
        new_text = call.get("new_text")

        if not isinstance(old_text, str) or not old_text:
            return "Error: old_text must be non-empty text"

        if not isinstance(new_text, str):
            return "Error: new_text must be text"

        path = safe_path(relative_path)

        if not path.is_file():
            return "Error: file not found"

        if path.stat().st_size > 200_000:
            return "Error: file too large to edit"

        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return f"Error: cannot read file: {exc}"

        if original.count(old_text) != 1:
            return (
                "Error: old_text must match exactly once; "
                "file was not changed"
            )

        backup = path.with_name(path.name + ".bak")

        if backup.exists():
            return f"Error: backup already exists: {backup.name}"

        updated = original.replace(old_text, new_text, 1)

        try:
            with backup.open("x", encoding="utf-8", newline="") as file:
                file.write(original)

            with path.open("w", encoding="utf-8", newline="") as file:
                file.write(updated)

        except OSError as exc:
            return (
                f"Error while editing file: {exc}. "
                f"Check backup: {backup.name}"
            )

        return (
            f"Updated file: {path.relative_to(WORKSPACE)}; "
            f"backup: {backup.name}"
        )

    if tool == "finish":
        return call.get("answer", "")

    return (
        "Unknown tool. Available tools: "
        "list_files, read_file, search_text, "
        "write_file, edit_file, finish"
    )


def call_model(messages, max_tokens=256, temperature=0.2):
    started = time.perf_counter()

    # Bound the requested output length to keep test requests small.
    try:
        token_limit = int(max_tokens)
    except (TypeError, ValueError):
        token_limit = 128

    token_limit = max(1, min(token_limit, 512))

    try:
        temp = float(temperature)
    except (TypeError, ValueError):
        temp = 0.2

    temp = max(0.0, min(temp, 2.0))

    print(
        f"[QWEN] Request started | messages={len(messages)} "
        f"| max_tokens={token_limit}",
        flush=True,
    )

    try:
        response = requests.post(
            f"{LLAMA_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": temp,
                "max_tokens": token_limit,
                "stream": False,
            },
            timeout=(10, 120),
        )

        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content") or ""

        print(
            f"[QWEN] finish_reason={data['choices'][0].get('finish_reason')}",
            flush=True,
      )

        print(
            f"[QWEN] Request completed | "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

        return content

    except requests.RequestException as exc:
        print(
            f"[QWEN] Request failed | "
            f"elapsed={time.perf_counter() - started:.1f}s "
            f"| error={exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=f"Qwen request failed: {exc}",
        )

    except (KeyError, IndexError, ValueError) as exc:
        print(f"[QWEN] Invalid response: {exc}", flush=True)

        raise HTTPException(
            status_code=502,
            detail=f"Invalid Qwen response: {exc}",
        )



def agent_reply(messages, max_tokens=256, temperature=0.2):
    history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    history.extend(messages[-6:])

    for step in range(MAX_STEPS):
        answer = call_model(
            history,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if not answer.strip():
            return "เน€เธยเน€เธเธเน€เธโฌเน€เธโ€เน€เธเธ…เน€เธยเน€เธเธเน€เธยเน€เธยเน€เธโ€เน€เธยเน€เธเธเน€เธยเน€เธยเน€เธยเน€เธยเน€เธเธเน€เธยเน€เธเธเน€เธเธ’เน€เธเธเน€เธยเน€เธเธ…เน€เธเธ‘เน€เธยเน€เธเธเน€เธเธ’ เน€เธยเน€เธเธเน€เธเธเน€เธโ€เน€เธเธ’เน€เธเธ…เน€เธเธเน€เธยเน€เธยเน€เธเธเน€เธเธเน€เธยเน€เธเธเน€เธเธ•เน€เธยเน€เธยเน€เธเธเน€เธเธ‘เน€เธยเน€เธย"

        try:
            call = json.loads(answer.strip())
        except json.JSONDecodeError:
            return answer

        if not isinstance(call, dict) or "tool" not in call:
            return answer

        try:
            result = run_tool(call)
        except (ValueError, OSError) as exc:
            result = f"Tool error: {exc}"

        tool_name = call.get("tool", "")

        print(
            f"[AGENT] Tool step {step + 1}: {tool_name}",
            flush=True,
        )

        if tool_name == "finish":
            return result

        history.append({
            "role": "assistant",
            "content": answer,
        })
        history.append({
            "role": "user",
            "content": (
                "Tool result:\n"
                + result[:12000]
                + "\n\nUse this result to answer the user's request. "
                "If another tool is necessary, return one JSON tool call. "
                "Otherwise, answer the user normally in natural language."
            ),
        })

    # Ask for a final natural-language answer after tool execution.
    history.append({
        "role": "user",
        "content": (
            "Now give the user the final answer using the tool results "
            "above. Do not call any more tools. Reply in natural language, "
            "not JSON."
        ),
    })

    final_answer = call_model(
        history,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Avoid showing a tool-call JSON object as the final response.
    try:
        final_data = json.loads(final_answer.strip())
        if isinstance(final_data, dict) and "tool" in final_data:
            tool_name = final_data.get("tool", "")
            if tool_name == "finish":
                return str(final_data.get("answer", ""))
            return (
                "Agent เน€เธโฌเน€เธเธเน€เธเธ•เน€เธเธเน€เธยเน€เธโฌเน€เธยเน€เธเธเน€เธเธ—เน€เธยเน€เธเธเน€เธยเน€เธเธเน€เธเธ—เน€เธเธเน€เธยเน€เธเธเน€เธยเน€เธยเน€เธเธ“เน€เธยเน€เธเธเน€เธยเน€เธเธเน€เธเธเน€เธยเน€เธโ€”เน€เธเธ•เน€เธยเน€เธยเน€เธเธ“เน€เธเธเน€เธยเน€เธโ€เน€เธยเน€เธเธ…เน€เธยเน€เธเธ "
                "เน€เธยเน€เธโ€ขเน€เธยเน€เธเธเน€เธเธ‘เน€เธยเน€เธยเน€เธเธเน€เธยเน€เธเธเน€เธเธ’เน€เธเธเน€เธเธ’เน€เธเธเน€เธโ€“เน€เธเธเน€เธเธเน€เธยเน€เธเธ’เน€เธยเน€เธยเน€เธเธ“เน€เธโ€ขเน€เธเธเน€เธยเน€เธเธเน€เธเธเน€เธโ€เน€เธโ€”เน€เธยเน€เธเธ’เน€เธเธเน€เธยเน€เธโ€เน€เธย "
                "เน€เธยเน€เธเธเน€เธเธเน€เธโ€เน€เธเธ’เน€เธเธ…เน€เธเธเน€เธยเน€เธยเน€เธเธ“เน€เธเธเน€เธเธ‘เน€เธยเน€เธยเน€เธโ€”เน€เธเธ•เน€เธยเน€เธเธเน€เธเธ‘เน€เธยเน€เธยเน€เธเธ…เน€เธย"
            )
    except json.JSONDecodeError:
        pass

    return final_answer


class ChatRequest(BaseModel):
    model: str = "Qwen-local"
    messages: list[dict]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = 128


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "agent-api",
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "object": "model",
                "owned_by": "local-agent",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    user_messages = [
        {
            "role": item.get("role", "user"),
            "content": item.get("content", ""),
        }
        for item in request.messages
        if item.get("role") in ("user", "assistant")
        and isinstance(item.get("content", ""), str)
    ]

    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="messages must not be empty",
        )

    max_tokens = (
        request.max_tokens
        if request.max_tokens is not None
        else 128
    )

    temperature = (
        request.temperature
        if request.temperature is not None
        else 0.2
    )

    answer = agent_reply(
        user_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
    }

    if request.stream:
        def events():
            chunk = {
                "id": result["id"],
                "object": "chat.completion.chunk",
                "created": result["created"],
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": answer,
                        },
                        "finish_reason": None,
                    }
                ],
            }

            yield (
                "data: "
                + json.dumps(chunk, ensure_ascii=False)
                + "\n\n"
            )

            chunk["choices"][0]["delta"] = {}
            chunk["choices"][0]["finish_reason"] = "stop"

            yield (
                "data: "
                + json.dumps(chunk, ensure_ascii=False)
                + "\n\n"
            )

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
        )

    return JSONResponse(result)
