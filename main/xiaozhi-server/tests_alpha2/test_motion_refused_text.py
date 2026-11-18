"""Motion-refused text (alpha2-brain#145): a refused motion gets a specific reply.

Run: cd main/xiaozhi-server && python3 -m pytest --noconftest -p no:cacheprovider tests_alpha2/
tool_error.py uses only the standard library, so the test loads it by file
path and reads the handler sources. It imports no server package.
"""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_ERROR_PATH = os.path.join(
    ROOT, "core", "providers", "tools", "device_mcp", "tool_error.py"
)
HANDLERS = [
    "core/providers/tools/device_mcp/mcp_handler.py",
    "core/providers/tools/mcp_endpoint/mcp_endpoint_handler.py",
]

MOTION_TEXT = "My motion is off. The owner must turn it on."
GENERIC_TEXT = "I could not do that."


def _load_tool_error():
    spec = importlib.util.spec_from_file_location(
        "tool_error_under_test", TOOL_ERROR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool_error = _load_tool_error()


def _error_result(text):
    return {"isError": True, "error": text}


def test_motion_refused_gives_the_motion_text():
    result = _error_result(
        "Motion is refused: play_action moves more than the head, and the "
        "setting motion_enabled is off."
    )
    assert tool_error.tool_error_message(result) == MOTION_TEXT


def test_other_error_gives_the_generic_text():
    assert tool_error.tool_error_message(_error_result("the tool failed.")) == GENERIC_TEXT


def test_motion_text_in_a_content_item_gives_the_motion_text():
    result = {
        "isError": True,
        "content": [{"type": "text", "text": "Motion is refused: the motion is off."}],
    }
    assert tool_error.tool_error_message(result) == MOTION_TEXT


def test_other_error_in_a_content_item_gives_the_generic_text():
    result = {
        "isError": True,
        "content": [{"type": "text", "text": "the tool failed for another reason."}],
    }
    assert tool_error.tool_error_message(result) == GENERIC_TEXT


def test_both_handlers_raise_the_shared_error_text():
    for path in HANDLERS:
        source = open(os.path.join(ROOT, path), encoding="utf8").read()
        assert "raise RuntimeError(tool_error_message(raw_result))" in source, path
