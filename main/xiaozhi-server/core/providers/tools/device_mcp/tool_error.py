"""Read the error text of a device MCP tool result.

The handler in mcp_handler.py calls error_text() when a tool result has
isError set. The app puts its message in the error field or in the text
of a content item. This module uses only the standard library, so the
unit test can load it by file path without the heavy package imports.
"""

NO_ERROR_DETAILS = "The tool gave no error details."

# The error text of a tool that refuses motion. Both handlers use these texts,
# so they live here in one place.
MOTION_REFUSED_PREFIX = "Motion is refused"
MOTION_OFF_TEXT = "My motion is off. The owner must turn it on."
GENERIC_ERROR_TEXT = "I could not do that."


def error_text(raw_result) -> str:
    """Return the error text of a device MCP tool result.

    When raw_result is a dict with isError true, return the value of the
    error field if it is a non-empty string. Otherwise return the text of
    the first item of content that is a dict with a non-empty text string.
    Otherwise return 'The tool gave no error details.'.
    """
    if not isinstance(raw_result, dict):
        return NO_ERROR_DETAILS
    if raw_result.get("isError") is not True:
        return NO_ERROR_DETAILS

    error = raw_result.get("error")
    if isinstance(error, str) and error:
        return error

    content = raw_result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                return text

    return NO_ERROR_DETAILS


def tool_error_message(raw_result) -> str:
    """Return the text the robot speaks for a tool error result.

    A result whose error text starts with 'Motion is refused' gets the
    motion-off text. Every other error gets the generic text.
    """
    if error_text(raw_result).startswith(MOTION_REFUSED_PREFIX):
        return MOTION_OFF_TEXT
    return GENERIC_ERROR_TEXT
