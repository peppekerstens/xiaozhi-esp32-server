"""Turn gate for open-mic (auto mode) devices.

An open microphone hears background talk, noise and the device's own speech.
Without a gate each ASR fragment aborts the current answer and starts a new
LLM call. The gate does four things (config section ``turn_gate``):

1. Noise: drop short fragments (fewer than ``min_words`` words) unless all
   their content words are on ``allow_words``. Drop filler-only fragments.
2. Barge-in: while the device speaks, drop each fragment, except a stop
   phrase. A stop phrase aborts the answer.
3. Echo: drop a fragment that repeats the words of the last reply
   (``echo_overlap``), or that arrives within ``echo_guard_ms`` after the
   reply ended.
4. Single flight: one LLM call per connection at a time. A newer question
   replaces the pending one. With ``supersede`` the running call stops at its
   next stream chunk and the newest question runs next.

The gate is off when ``turn_gate.enabled`` is false (the upstream behaviour).
"""

import re
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__

WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

DEFAULTS = {
    "enabled": False,
    "min_words": 3,
    "allow_words": [
        "stop", "hello", "hi", "hey", "thanks", "thank", "help", "dance", "wave",
        "sit", "stand", "goodbye", "bye", "hallo", "dank", "dankjewel", "doei", "dag",
    ],
    # Short answers. They count only as a reply to a question of the device
    # (the last reply ends with "?") within ``answer_window_s``.
    "answer_words": ["yes", "no", "okay", "ok", "sure", "yeah", "ja", "nee", "prima", "goed"],
    "answer_window_s": 20,
    "filler_words": [
        "uh", "um", "umm", "uhm", "hm", "hmm", "mm", "mhm", "ah", "eh", "er", "oh",
        "the", "a", "an", "robot", "alpha", "please", "you", "it", "now", "that",
        "nou", "eh", "ehm", "he", "de", "het", "een", "maar", "dus",
    ],
    "stop_words": ["stop", "quiet", "silence", "shut", "enough", "stil", "ophouden", "genoeg"],
    "barge_in": "stop_words",  # stop_words | any | none
    "echo_guard_ms": 1500,
    "echo_overlap": 0.6,
    "echo_window_s": 15,
    "supersede": True,
}


def _cfg(conn: "ConnectionHandler") -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(conn.config.get("turn_gate") or {})
    return cfg


def enabled(conn: "ConnectionHandler") -> bool:
    return bool(_cfg(conn).get("enabled"))


def words(text: str) -> list:
    return WORD_RE.findall((text or "").lower())


def is_noise(text: str, cfg: dict, asked: bool = False) -> bool:
    """True when ``text`` is probably noise and not a request of the user.

    ``asked``: the device asked a question a short time ago. Then a short
    answer such as "yes" or "okay" is not noise.
    """
    ws = words(text)
    filler = set(cfg["filler_words"])
    content = [w for w in ws if w not in filler]
    if not content:
        return True
    if len(ws) < int(cfg["min_words"]):
        allow = set(cfg["allow_words"])
        if asked:
            allow |= set(cfg["answer_words"])
        return not all(w in allow for w in content)
    return False


def is_stop(text: str, cfg: dict) -> bool:
    """True for a short stop phrase such as "stop", "stop it" or "be quiet"."""
    ws = words(text)
    return 0 < len(ws) <= 4 and any(w in set(cfg["stop_words"]) for w in ws)


def is_echo(text: str, cfg: dict, spoken: str) -> bool:
    """True when most content words of ``text`` are in the reply ``spoken``."""
    if not spoken:
        return False
    filler = set(cfg["filler_words"])
    ws = [w for w in words(text) if w not in filler]
    if not ws:
        return False
    said = set(words(spoken))
    return sum(w in said for w in ws) / len(ws) >= float(cfg["echo_overlap"])


def _state(conn: "ConnectionHandler") -> dict:
    st = getattr(conn, "_turn_gate", None)
    if st is None:
        st = {
            "lock": threading.Lock(),
            "gen": 0,  # number of the newest accepted question
            "running_gen": 0,  # number of the question in the LLM now
            "inflight": False,
            "pending": None,  # (gen, text) of the next question
            "spoken": "",  # reply text of the current turn
            "last_spoken": "",  # reply text of the last finished turn
            "speak_end": 0.0,  # monotonic time of the last tts stop
            "speaking": False,  # audio of the current turn has started
        }
        conn._turn_gate = st
    return st


# ---- hooks from the TTS path ----------------------------------------------

def on_sentence(conn: "ConnectionHandler", text: Optional[str]) -> None:
    """Called at each tts sentence_start: the device speaks ``text`` now."""
    st = _state(conn)
    st["speaking"] = True
    if text:
        st["spoken"] = (st["spoken"] + " " + text).strip()


def on_speech_end(conn: "ConnectionHandler") -> None:
    """Called at tts stop."""
    st = _state(conn)
    if st["spoken"]:
        st["last_spoken"] = st["spoken"]
    st["spoken"] = ""
    st["speaking"] = False
    st["speak_end"] = time.monotonic()
    if enabled(conn):
        # Push-to-talk scripts wait for this line to close the microphone.
        conn.logger.bind(tag=TAG).info("turn_gate: speech end")


# ---- decision for one ASR fragment ------------------------------------------

def check(conn: "ConnectionHandler", text: str) -> str:
    """Return the action for a voice fragment: ``chat``, ``drop`` or ``stop``."""
    cfg = _cfg(conn)
    st = _state(conn)
    log = conn.logger.bind(tag=TAG)
    speaking = st["speaking"]
    busy = conn.client_is_speaking or st["inflight"]
    stop = is_stop(text, cfg)

    if stop:
        if busy:
            log.info(f"turn_gate: stop phrase, abort the answer: {text!r}")
            return "stop"
        log.info(f"turn_gate: stop phrase while idle, drop: {text!r}")
        return "drop"
    since_speech = time.monotonic() - st["speak_end"]
    asked = st["last_spoken"].rstrip().endswith("?") and since_speech < float(cfg["answer_window_s"])
    if is_noise(text, cfg, asked):
        log.info(f"turn_gate: noise, drop: {text!r}")
        return "drop"
    if speaking and cfg["barge_in"] != "any":
        log.info(f"turn_gate: heard while speaking, drop: {text!r}")
        return "drop"
    recent = since_speech < float(cfg["echo_window_s"])
    if recent and is_echo(text, cfg, st["last_spoken"]):
        log.info(f"turn_gate: echo of the last reply, drop: {text!r}")
        return "drop"
    if since_speech * 1000 < float(cfg["echo_guard_ms"]):
        log.info(f"turn_gate: inside the echo guard after speech, drop: {text!r}")
        return "drop"
    log.info(f"turn_gate: accept: {text!r}")
    return "chat"


# ---- single flight -----------------------------------------------------------

def submit(conn: "ConnectionHandler", text: str) -> None:
    """Run ``conn.chat(text)``, one call at a time. The newest question wins."""
    cfg = _cfg(conn)
    st = _state(conn)
    log = conn.logger.bind(tag=TAG)
    with st["lock"]:
        st["gen"] += 1
        gen = st["gen"]
        if st["inflight"]:
            if st["pending"] is not None:
                log.info(f"turn_gate: drop the stale pending question: {st['pending'][1]!r}")
            st["pending"] = (gen, text)
            if cfg["supersede"]:
                log.info(f"turn_gate: newer question supersedes the running one: {text!r}")
            return
        st["inflight"] = True
    conn.executor.submit(_run, conn, gen, text)


def _run(conn: "ConnectionHandler", gen: int, text: str) -> None:
    st = _state(conn)
    while True:
        st["running_gen"] = gen
        st["spoken"] = ""
        try:
            conn.chat(text)
        except Exception as e:  # noqa: BLE001
            conn.logger.bind(tag=TAG).error(f"turn_gate: chat failed: {e}")
        with st["lock"]:
            if st["pending"] is None:
                st["inflight"] = False
                return
            gen, text = st["pending"]
            st["pending"] = None
        conn.client_abort = False


def superseded(conn: "ConnectionHandler") -> bool:
    """True when a newer question waits and ``supersede`` is on."""
    st = getattr(conn, "_turn_gate", None)
    if st is None or not st["inflight"] or st["pending"] is None:
        return False
    return bool(_cfg(conn)["supersede"]) and not st["speaking"]


def clear_pending(conn: "ConnectionHandler") -> None:
    st = _state(conn)
    with st["lock"]:
        st["pending"] = None
