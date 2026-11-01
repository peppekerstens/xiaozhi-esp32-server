"""Tests for the per-turn voice time log (core.utils.voice_timing)."""
import threading

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
        self.sentence_id = "a1"
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
    clock.at(911); voice_timing.mark(conn, "audio_first", "a1")
    clock.at(950); voice_timing.mark(conn, "audio_first", "a1")
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


def test_old_answer_audio_does_not_set_new_turn():
    """Barge-in: frames of answer a1 still go out after turn 2 starts."""
    clock = FakeClock()
    conn = FakeConn(clock)
    voice_timing.start(conn)
    clock.at(100); voice_timing.llm_call(conn)  # binds turn 1 to a1
    clock.at(200); voice_timing.mark(conn, "audio_first", "a1")
    clock.at(1000); voice_timing.start(conn)  # turn 2, not bound yet
    clock.at(1100); voice_timing.mark(conn, "audio_first", "a1")  # old audio
    conn.sentence_id = "b2"
    clock.at(1400); voice_timing.llm_call(conn)  # binds turn 2 to b2
    clock.at(1500); voice_timing.mark(conn, "audio_first", "a1")  # old audio
    clock.at(1700); voice_timing.mark(conn, "audio_first", "b2")
    clock.at(3000); voice_timing.finish(conn)
    assert "turn=1 " in conn.lines[0] and "audio_first=200 " in conn.lines[0]
    assert "turn=2 " in conn.lines[1] and "audio_first=700 " in conn.lines[1]


def test_tts_req_binds_turn_without_llm_call():
    clock = FakeClock()
    conn = FakeConn(clock)
    conn.sentence_id = None
    voice_timing.start(conn)
    clock.at(50); voice_timing.mark(conn, "audio_first", "c3")  # not bound
    clock.at(100); voice_timing.mark(conn, "tts_req", "c3")
    clock.at(300); voice_timing.mark(conn, "audio_first", "c3")
    clock.at(900); voice_timing.finish(conn)
    assert "tts_req=100 " in conn.lines[0]
    assert "audio_first=300 " in conn.lines[0]


def test_recorder_creation_race_gives_one_recorder():
    class Bare:
        pass

    for _ in range(50):
        conn = Bare()
        barrier = threading.Barrier(8)
        seen = []

        def worker():
            barrier.wait()
            seen.append(voice_timing._recorder(conn))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len({id(r) for r in seen}) == 1
        assert seen[0] is conn._voice_timing
