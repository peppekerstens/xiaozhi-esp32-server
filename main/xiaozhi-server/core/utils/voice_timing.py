"""Time log of one voice turn (alpha2-brain, queue item 6).

One log line per turn with the text ``VoiceTiming`` and a stable key=value
format::

    VoiceTiming v=1 session=<id> turn=3 asr_start=0 asr_end=420 llm_req=431
    llm_first=690 llm_end=1510 llm_calls=1 tts_req=702 tts_first=905
    audio_first=911 tts_stop=4380

All values are milliseconds from ``asr_start`` on a monotonic clock
(``time.monotonic``). A value that did not occur is ``-``.

- asr_start: the VAD found the end of speech and the ASR starts (t0).
- asr_end: the ASR result is ready.
- llm_req: the first LLM request of the turn.
- llm_first: the first stream chunk of the first LLM call (text or tool call).
- llm_end: the end of the last LLM stream of the turn.
- llm_calls: the number of LLM calls (a tool call adds one).
- tts_req: the first text segment goes to the TTS.
- tts_first: the first TTS audio chunk is ready to send.
- audio_first: the first audio frame of this turn went to the device. The
  turn binds to the sentence id of its answer (the sentence id of the
  top-level LLM call, or the TTS message at ``tts_req``). A frame of an older answer
  (barge-in, follow-up) does not set ``audio_first`` of a new turn.
- tts_stop: the tts stop message goes to the device (all audio is sent).

The line comes at tts stop, or at the start of the next turn when a turn did
not get to tts stop. A turn without ASR (typed text) gives no line. The module
only records times. It changes no behaviour.
"""

import threading
import time
from typing import Callable, Dict, Optional

TAG = __name__

KEYS = (
    "asr_start",
    "asr_end",
    "llm_req",
    "llm_first",
    "llm_end",
    "tts_req",
    "tts_first",
    "audio_first",
    "tts_stop",
)
# Keys that keep the last time. All other keys keep the first time.
LAST_KEYS = ("llm_end",)


class Turn:
    def __init__(self, turn_id: int, t0: float):
        self.turn_id = turn_id
        self.t0 = t0
        self.at: Dict[str, float] = {}
        self.llm_calls = 0
        # The sentence id of the answer of this turn. The first bind wins.
        self.sentence_id: Optional[str] = None

    def bind(self, sentence_id: Optional[str]) -> None:
        if self.sentence_id is None and sentence_id:
            self.sentence_id = sentence_id

    def mark(self, key: str, now: float) -> None:
        if key in LAST_KEYS or key not in self.at:
            self.at[key] = now

    def format(self, session: str) -> str:
        parts = ["VoiceTiming v=1", f"session={session or '-'}", f"turn={self.turn_id}"]
        for key in KEYS:
            if key in self.at:
                parts.append(f"{key}={int(round((self.at[key] - self.t0) * 1000))}")
            else:
                parts.append(f"{key}=-")
            if key == "llm_end":
                parts.append(f"llm_calls={self.llm_calls}")
        return " ".join(parts)


class Recorder:
    """Per connection. Thread safe: the ASR, LLM and TTS run on other threads."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._counter = 0
        self._turn: Optional[Turn] = None

    def start(self) -> Optional[Turn]:
        """A new turn at the ASR start. Returns the open turn that did not end."""
        with self._lock:
            old = self._turn
            self._counter += 1
            now = self._clock()
            self._turn = Turn(self._counter, now)
            self._turn.mark("asr_start", now)
            return old

    def mark(self, key: str, sentence_id: Optional[str] = None) -> None:
        """Mark a key. For tts_req, sentence_id binds the turn. For
        audio_first, sentence_id must match the bound id of the turn."""
        with self._lock:
            turn = self._turn
            if turn is None:
                return
            if key == "audio_first":
                if turn.sentence_id is None or sentence_id != turn.sentence_id:
                    return
            elif sentence_id is not None:
                turn.bind(sentence_id)
            turn.mark(key, self._clock())

    def llm_call(self, sentence_id: Optional[str] = None) -> None:
        with self._lock:
            if self._turn is not None:
                self._turn.llm_calls += 1
                self._turn.bind(sentence_id)
                self._turn.mark("llm_req", self._clock())

    def finish(self) -> Optional[Turn]:
        """tts stop: mark it and close the turn. Returns the closed turn."""
        with self._lock:
            turn = self._turn
            self._turn = None
            if turn is not None:
                turn.mark("tts_stop", self._clock())
            return turn


# Guards the first creation of the Recorder of a connection. The ASR thread,
# the TTS thread and the event loop can call _recorder() at the same time.
_CREATE_LOCK = threading.Lock()


def _recorder(conn) -> Recorder:
    rec = getattr(conn, "_voice_timing", None)
    if rec is not None:
        return rec
    with _CREATE_LOCK:
        rec = getattr(conn, "_voice_timing", None)
        if rec is None:
            rec = Recorder()
            try:
                conn._voice_timing = rec
            except Exception:
                pass
    return rec


def _log(conn, turn: Optional[Turn]) -> None:
    if turn is None:
        return
    try:
        conn.logger.bind(tag=TAG).info(turn.format(getattr(conn, "session_id", "")))
    except Exception:
        pass


def start(conn) -> None:
    """The ASR starts: a new turn. An open turn writes its line first."""
    try:
        _log(conn, _recorder(conn).start())
    except Exception:
        pass


def mark(conn, key: str, sentence_id: Optional[str] = None) -> None:
    try:
        _recorder(conn).mark(key, sentence_id)
    except Exception:
        pass


def llm_call(conn, sentence_id: Optional[str] = None) -> None:
    """An LLM call. Only the top-level chat call (depth 0) gives its
    sentence_id. A tool-call recursion (depth > 0) gives None and does not
    bind the turn: it can belong to an older answer."""
    try:
        _recorder(conn).llm_call(sentence_id)
    except Exception:
        pass


def finish(conn) -> None:
    """tts stop: the turn writes its line."""
    try:
        _log(conn, _recorder(conn).finish())
    except Exception:
        pass
