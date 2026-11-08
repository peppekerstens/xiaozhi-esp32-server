from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

repeat_last_utterance_function_desc = {
    "type": "function",
    "function": {
        "name": "repeat_last_utterance",
        "description": (
            "Repeat the last sentence that the user said before this request. "
            "Call this when the user asks you to repeat what they said."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


@register_function(
    "repeat_last_utterance", repeat_last_utterance_function_desc, ToolType.SYSTEM_CTL
)
def repeat_last_utterance(conn: "ConnectionHandler"):
    # Speak the last user sentence from before this request.
    try:
        messages = conn.dialogue.dialogue
        # Skip temporary (few-shot) messages and the current request.
        user_messages = [
            m
            for m in messages
            if m.role == "user" and not getattr(m, "is_temporary", False)
        ]
        # The newest user message is the current request, so skip it.
        if len(user_messages) < 2:
            text = "I did not hear anything before this."
            return ActionResponse(
                action=Action.RESPONSE, result=text, response=text
            )
        text = user_messages[-2].content or ""
        logger.bind(tag=TAG).info(f"Repeat last utterance: {text}")
        return ActionResponse(action=Action.RESPONSE, result=text, response=text)
    except Exception as e:
        logger.bind(tag=TAG).error(f"Repeat last utterance error: {e}")
        return ActionResponse(action=Action.NONE, result="I could not repeat it.", response="")
