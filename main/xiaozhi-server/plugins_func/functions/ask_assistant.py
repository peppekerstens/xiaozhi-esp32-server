"""ask_assistant: give a slow task to the alpha2-assistant (opencode on legion-t5).

Issue peppekerstens/alpha2-brain#17. The fast voice brain calls this tool for research,
the internet, the weather or a multi-step task. The tool answers at once with a short
"I will look that up" sentence. A worker thread asks the assistant. When the answer is
ready and the robot is not speaking, the robot speaks the answer (at most 3 sentences).

Settings (environment of the Xiaozhi container):
  ALPHA2_ASSISTANT_URL       for example http://192.168.2.133:4096 (empty: the tool is off)
  ALPHA2_ASSISTANT_PASSWORD  the basic-auth password of the opencode API (user "opencode")
  ALPHA2_ASSISTANT_TIMEOUT   seconds, default 180
"""

import base64
import json
import os
import re
import threading
import time
import urllib.request

from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action

TAG = __name__
logger = setup_logging()

ACK = {"en": "I will look that up.", "nl": "Ik zoek het op."}
FAIL = {"en": "Sorry, I could not find the answer.", "nl": "Sorry, ik kon het antwoord niet vinden."}

ASK_ASSISTANT_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "ask_assistant",
        "description": (
            "Give a slow task to the research assistant: questions that need the internet, "
            "the weather, news, facts that you do not know, or several steps. "
            "The assistant answers later, and the robot speaks the answer. "
            "Do not use it for robot motion, lights or small talk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The full task in one sentence, with the place and the day if they matter.",
                },
                "language": {
                    "type": "string",
                    "description": "The language of the person: en or nl.",
                },
            },
            "required": ["task"],
        },
    },
}


def _call(url, auth, method, path, body=None, timeout=30.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method=method,
                                 headers={"Authorization": auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def ask(task, url, password, timeout=180.0):
    """Ask the opencode API. Return the text of the last assistant message ("" if none)."""
    auth = "Basic " + base64.b64encode(f"opencode:{password}".encode()).decode()
    session = _call(url, auth, "POST", "/api/session", {"agent": "assistant", "title": "robot " + task[:40]})
    sid = (session.get("data") or session)["id"]
    _call(url, auth, "POST", f"/api/session/{sid}/prompt", {"text": task})
    _call(url, auth, "POST", f"/api/experimental/session/{sid}/wait", None, timeout=timeout)
    messages = _call(url, auth, "GET", f"/api/session/{sid}/message")["data"]
    for m in messages:  # newest first
        if m.get("type") != "assistant":
            continue
        texts = [c["text"] for c in m.get("content", []) if c.get("type") == "text"]
        if texts:
            return texts[-1]
    return ""


def speech_text(answer, max_sentences=3):
    """Remove the Source line, markdown and URLs. Keep at most max_sentences sentences."""
    lines = [l for l in answer.splitlines() if not l.strip().lower().startswith("source:")]
    text = " ".join(l.strip() for l in lines if l.strip())
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`>|]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences]).strip()


def _speak_when_idle(conn, text, max_wait=60.0):
    """Wait until the robot does not speak, then speak text."""
    from core.handle.intentHandler import speak_txt

    end = time.monotonic() + max_wait
    while getattr(conn, "client_is_speaking", False) and time.monotonic() < end:
        time.sleep(0.5)
    speak_txt(conn, text)


def _worker(conn, task, lang, url, password, timeout):
    t0 = time.monotonic()
    try:
        text = speech_text(ask(task, url, password, timeout))
    except Exception as e:  # network, timeout, HTTP error
        logger.bind(tag=TAG).error(f"ask_assistant failed: {type(e).__name__}: {e}")
        text = ""
    logger.bind(tag=TAG).info(f"ask_assistant answer after {time.monotonic() - t0:.1f} s: {text!r}")
    _speak_when_idle(conn, text or FAIL.get(lang, FAIL["en"]))


@register_function("ask_assistant", ASK_ASSISTANT_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def ask_assistant(conn, task=None, language="en"):
    lang = language if language in ACK else "en"
    url = os.environ.get("ALPHA2_ASSISTANT_URL", "").rstrip("/")
    password = os.environ.get("ALPHA2_ASSISTANT_PASSWORD", "")
    if not task:
        return ActionResponse(Action.REQLLM, "The task is empty. Ask the person what to look up.", None)
    if not url or not password:
        return ActionResponse(Action.RESPONSE, None, FAIL[lang])
    timeout = float(os.environ.get("ALPHA2_ASSISTANT_TIMEOUT", "180"))
    # The date makes "tomorrow" exact. It also keeps the 1-hour response cache of the
    # LiteLLM router from giving an answer of an earlier day.
    task = f"Today is {time.strftime('%A %Y-%m-%d')}. {task}"
    threading.Thread(target=_worker, args=(conn, task, lang, url, password, timeout),
                     name="ask_assistant", daemon=True).start()
    logger.bind(tag=TAG).info(f"ask_assistant started: {task!r}")
    return ActionResponse(Action.RESPONSE, None, ACK[lang])
