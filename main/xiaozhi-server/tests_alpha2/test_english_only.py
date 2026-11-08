"""English only (alpha2-brain#88, D-62): Dutch voice text gives no turn.

Run: python3 -m pytest --noconftest -p no:cacheprovider tests_alpha2/
robot_guard uses only the standard library, so no server package is needed.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import robot_guard  # noqa: E402

DUTCH = [
    "wat vind je van het weer vandaag",
    "zet je ogen op groen alsjeblieft",
    "ik wil graag koffie en een koekje",
    "hoe laat komen ze morgen thuis",
    "sta op",
]
ENGLISH = [
    "what is the weather like today",
    "make your eyes green please",
    "how far is the wall",
    "tell me a joke",
    "hello robot",
]


def conn(languages="unset", turn_language="auto"):
    cfg = {"turn_language": turn_language}
    if languages != "unset":
        cfg["voice"] = {"languages": languages}
    return SimpleNamespace(config=cfg, dialogue=SimpleNamespace(), turn_language=None)


def test_dutch_gives_no_turn_when_english_only():
    c = conn(["en"])
    for text in DUTCH:
        ok, reason = robot_guard.language_check(c, text)
        assert not ok, text
        assert "language nl is off" in reason


def test_english_gives_the_normal_turn_when_english_only():
    c = conn(["en"])
    for text in ENGLISH:
        assert robot_guard.language_check(c, text) == (True, ""), text


def test_dutch_works_again_when_nl_is_in_the_list():
    c = conn(["en", "nl"])
    for text in DUTCH + ENGLISH:
        assert robot_guard.language_check(c, text)[0], text


def test_no_setting_means_no_restriction():
    c = conn()
    for text in DUTCH + ENGLISH:
        assert robot_guard.language_check(c, text)[0], text


def test_string_setting_is_accepted():
    assert not robot_guard.language_check(conn("en"), DUTCH[0])[0]


def test_answer_language_and_voice_stay_english():
    c = conn(["en"])
    robot_guard.update_turn_language(c, DUTCH[0])
    assert c.turn_language == "en"
    assert c.dialogue.answer_language == "English"
    assert c.dialogue.answer_hint is None
    c = conn(["en", "nl"])
    robot_guard.update_turn_language(c, DUTCH[0])
    assert c.turn_language == "nl"
    assert c.dialogue.answer_hint == "(Antwoord in het Nederlands.)"
