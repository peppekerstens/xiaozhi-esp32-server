"""Short command fragments are requests, not noise (alpha2-brain#134).

A short fragment (fewer than ``min_words`` words) is a request when at least
one of its content words is on ``allow_words``. Filler-only fragments stay
noise.
"""
import pytest

from core.utils import turn_gate


def _cfg():
    return dict(turn_gate.DEFAULTS)


@pytest.mark.parametrize("text", [
    "Stand up.", "Squat down.", "Squatted down.", "Volume down.", "Down.", "Stop.",
])
def test_short_command_is_not_noise(text):
    assert turn_gate.is_noise(text, _cfg()) is False


@pytest.mark.parametrize("text", ["Aha.", "Alpha.", "Alpha?", "Uh.", "Hmm."])
def test_filler_is_noise(text):
    assert turn_gate.is_noise(text, _cfg()) is True


def test_short_fragment_without_an_allowed_word_is_noise():
    assert turn_gate.is_noise("Big red.", _cfg()) is True
