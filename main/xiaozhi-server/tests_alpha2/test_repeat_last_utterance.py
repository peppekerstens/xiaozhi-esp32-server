"""Repeat the last user sentence (alpha2-brain#108).

Run: cd main/xiaozhi-server && uv run --isolated --with pytest python -m pytest -q tests_alpha2/
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# Stub config.logger so importing the plugin does not load the server config
# (which needs data/.config.yaml and would reach the network). The function
# under test does not use the logger in the paths this test covers.
class _Bound:
    def _noop(self, *a, **k):
        pass

    debug = info = warning = error = exception = _noop


class _FakeLogger:
    def bind(self, *a, **k):
        return _Bound()

    def _noop(self, *a, **k):
        pass

    debug = info = warning = error = exception = _noop


fake_logger = types.ModuleType("config.logger")
fake_logger.setup_logging = lambda config=None: _FakeLogger()
sys.modules.setdefault("config.logger", fake_logger)

from plugins_func.functions.repeat_last_utterance import repeat_last_utterance  # noqa: E402
from plugins_func.register import Action  # noqa: E402
from core.utils.dialogue import Dialogue, Message  # noqa: E402

FALLBACK = "I did not hear anything before this."


def _conn(roles_contents):
    d = Dialogue()
    for role, content in roles_contents:
        d.put(Message(role=role, content=content))
    return types.SimpleNamespace(dialogue=d)


def test_repeats_the_utterance_before_the_request():
    conn = _conn(
        [
            ("user", "the sky is green"),
            ("assistant", "ok"),
            ("user", "repeat what I said"),
        ]
    )
    resp = repeat_last_utterance(conn)
    assert resp.action == Action.RESPONSE
    assert resp.result == "the sky is green"
    assert resp.response == "the sky is green"


def test_skips_the_current_request():
    conn = _conn([("user", "repeat what I said")])
    resp = repeat_last_utterance(conn)
    assert resp.action == Action.RESPONSE
    assert resp.result == FALLBACK
    assert resp.response == FALLBACK
