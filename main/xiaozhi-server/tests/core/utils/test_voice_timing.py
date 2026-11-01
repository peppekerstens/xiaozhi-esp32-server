"""Tests for the per-turn voice time log (core.utils.voice_timing)."""
from core.utils import voice_timing


class FakeClock:
    def __init__(self):
        self.t = 5000.0  # a large t0: values must be relative

    def __call__(self):
        return self.t

    def at(self, ms):
        self.t = 5000.0 + ms / 1000.0


class FakeLog:
    def __init__(self, lines):
        self.lines = lines

    def bind(self, **_):
        return self

    def info(self, text):
        self.lines.append(text)


class FakeConn:
    def __init__(self, clock):
        self.lines = []
        self.logger = FakeLog(self.lines)
        self.session_id = "s1"
        self._voice_timing = voice_timing.Recorder(clock)


def test_full_turn_gives_one_line():
    clock = FakeClock()
    conn = FakeConn(clock)
    voice_timing.start(conn)
    clock.at(420); voice_timing.mark(conn, "asr_end")
    clock.at(431); voice_timing.llm_call(conn)
    clock.at(690); voice_timing.mark(conn, "llm_first")
    clock.at(702); voice_timing.mark(conn, "tts_req")
    clock.at(800); voice_timing.mark(conn, "llm_end")
    clock.at(850); voice_timing.llm_call(conn)  # tool call: a second LLM call
    clock.at(900); voice_timing.mark(conn, "llm_first")  # first only
    clock.at(905); voice_timing.mark(conn, "tts_first")
    clock.at(911); voice_timing.mark(conn, "audio_first")
    clock.at(950); voice_timing.mark(conn, "audio_first")
    clock.at(1510); voice_timing.mark(conn, "llm_end")  # last wins
    assert conn.lines == []
    clock.at(4380); voice_timing.finish(conn)
    assert conn.lines == [
        "VoiceTiming v=1 session=s1 turn=1 asr_start=0 asr_end=420 llm_req=431"
        " llm_first=690 llm_end=1510 llm_calls=2 tts_req=702 tts_first=905"
        " audio_first=911 tts_stop=4380"
    ]


def test_open_turn_writes_at_next_start():
    clock = FakeClock()
    conn = FakeConn(clock)
    voice_timing.start(conn)
    clock.at(300); voice_timing.mark(conn, "asr_end")
    clock.at(2000); voice_timing.start(conn)
    assert conn.lines == [
        "VoiceTiming v=1 session=s1 turn=1 asr_start=0 asr_end=300 llm_req=-"
        " llm_first=- llm_end=- llm_calls=0 tts_req=- tts_first=- audio_first=- tts_stop=-"
    ]


def test_no_turn_no_line():
    clock = FakeClock()
    conn = FakeConn(clock)
    voice_timing.mark(conn, "audio_first")
    voice_timing.llm_call(conn)
    voice_timing.finish(conn)
    assert conn.lines == []


def test_conn_without_recorder_and_without_logger():
    class Bare:
        pass

    conn = Bare()
    voice_timing.start(conn)  # no logger: no exception
    voice_timing.finish(conn)
    assert isinstance(conn._voice_timing, voice_timing.Recorder)
