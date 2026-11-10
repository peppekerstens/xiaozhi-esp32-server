"""error_text must return the app text of a device MCP tool error result.

Run with: python -m pytest -q --noconftest tests/core/providers/tools/test_tool_error.py
(the test needs no project import and no config file)."""
import importlib.util
from pathlib import Path

# Load the module by file path, so the heavy package imports never run.
_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "core"
    / "providers"
    / "tools"
    / "device_mcp"
    / "tool_error.py"
)
_spec = importlib.util.spec_from_file_location("device_mcp_tool_error", _MODULE_PATH)
_tool_error = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tool_error)

error_text = _tool_error.error_text

NO_DETAILS = "The tool gave no error details."


def test_exact_issue_result_returns_the_app_text():
    raw_result = {
        "content": [
            {
                "type": "text",
                "text": "color and brightness are required unless preset=stop",
            }
        ],
        "isError": True,
    }
    assert error_text(raw_result) == (
        "color and brightness are required unless preset=stop"
    )


def test_error_field_is_used_when_it_is_a_non_empty_string():
    raw_result = {"isError": True, "error": "the app said no"}
    assert error_text(raw_result) == "the app said no"


def test_empty_content_gives_no_error_details():
    assert error_text({"isError": True, "content": []}) == NO_DETAILS


def test_result_that_is_not_a_dict_gives_no_error_details():
    assert error_text("not a dict") == NO_DETAILS
