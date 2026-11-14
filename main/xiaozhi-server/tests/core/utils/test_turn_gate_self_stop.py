"""The robot must not stop itself on its own word with "stop" in it (alpha2-brain#92)."""
import logging
import time
from types import SimpleNamespace

import pytest

from core.utils import turn_gate


class _Log:
    def bind(self, **_):
        return logging.getLogger("test")


def _conn(speaking=True):
    conn = SimpleNamespace(
        config={"turn_gate": {"enabled": True}},
        logger=_Log(),
        client_is_speaking=speaking,
    )
    return conn


def _says(conn, text):
    turn_gate.on_sentence(conn, text)


@pytest.mark.parametrize("robot", ["Unstoppable means that nothing can halt it.",
                                   "The car stopped.", "It is a non-stop show."])
def test_robot_word_with_stop_does_not_stop(robot):
    conn = _conn()
    _says(conn, robot)
    assert turn_gate.check(conn, "stop") == "drop"


def test_owner_stop_during_other_speech_stops():
    conn = _conn()
    _says(conn, "Once there was a long story about a dragon.")
    assert turn_gate.check(conn, "stop") == "stop"


def test_owner_stop_when_robot_says_whole_word_stop_stops():
    conn = _conn()
    _says(conn, "At the red light you must stop and look.")
    assert turn_gate.check(conn, "stop") == "stop"


def test_stop_with_no_robot_speech_stops_when_busy():
    conn = _conn()
    assert turn_gate.check(conn, "stop") == "stop"


def test_old_sentence_is_not_in_the_window():
    conn = _conn()
    _says(conn, "Unstoppable means no one can halt it.")
    _says(conn, "Here is the second sentence.")
    _says(conn, "And a third sentence follows.")
    assert turn_gate.check(conn, "stop") == "stop"


def test_window_after_speech_end():
    conn = _conn(speaking=False)
    conn.client_is_speaking = True
    _says(conn, "Unstoppable means no one can halt it.")
    turn_gate.on_speech_end(conn)
    assert turn_gate.check(conn, "stop") == "drop"
    st = turn_gate._state(conn)
    st["recent"] = [(t, time.monotonic() - 10) for t, _ in st["recent"]]
    assert turn_gate.check(conn, "stop") == "stop"


def test_filter_can_be_switched_off():
    conn = _conn()
    conn.config["turn_gate"]["self_word_filter"] = False
    _says(conn, "Unstoppable means no one can halt it.")
    assert turn_gate.check(conn, "stop") == "stop"


def test_is_self_word_unit():
    cfg = dict(turn_gate.DEFAULTS)
    assert turn_gate.is_self_word("stop", cfg, "unstoppable") is True
    assert turn_gate.is_self_word("stop", cfg, "stop it") is False
    assert turn_gate.is_self_word("stop", cfg, "") is False
    assert turn_gate.is_self_word("hello", cfg, "unstoppable") is False
