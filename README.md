# Local AI Agent Starter (Qwen + llama-server)

A small, local-first agent controller that calls an OpenAI-compatible llama-server and can:
- list files in a dedicated workspace
- read text files
- create or replace text files (with confirmation in the terminal)
- search text in workspace files
- create a simple project report

Safety defaults:
- File tools are restricted to `agent_workspace/`.
- No arbitrary shell execution.
- Writes require confirmation.
- The model is asked to return one JSON action at a time; output is validated by Python.
- Do not expose llama-server to untrusted networks. Bind to `127.0.0.1` when possible.

## 1. Requirements
- Windows 10/11
- Python 3.10+
- Your existing Qwen GGUF and llama-server
- `requests` package

## 2. Install
Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Start Qwen
In a separate PowerShell window, start the model using your existing working command. For example:

```powershell
& "D:\AI-Agent\llama-new\llama-server.exe" `
  -m "D:\AI-Agent\models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf" `
  --host 127.0.0.1 `
  --port 8080 `
  -c 8192 `
  -t 2 `
  -tb 2 `
  -np 1
```

Use `--host 127.0.0.1` for local-only access. If you need Docker/Open WebUI to connect, note that binding only to 127.0.0.1 may prevent container access; do not change the current working setup until you've confirmed how you want to connect. For this CLI agent, local loopback is sufficient.

## 4. Run the agent
In another PowerShell window, from this folder:

```powershell
python agent.py
```

The agent creates `agent_workspace` beside `agent.py`. Try:
- `ช่วยสร้างไฟล์ index.html หน้าเว็บแนะนำตัวอย่างง่าย`
- `ดูไฟล์ใน workspace แล้วสรุปว่ามีอะไรบ้าง`
- `ค้นคำว่า TODO ในไฟล์ทั้งหมด`

For a first test, choose a harmless task and approve the write when asked.

## 5. What this prototype is / isn't
This is an initial tool-using agent prototype, not a fully autonomous OS assistant. It does not execute arbitrary commands, access folders outside its workspace, or delete files. It uses the local model through `/v1/chat/completions`.
