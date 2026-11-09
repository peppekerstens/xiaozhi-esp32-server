"""English texts (alpha2-brain#95, #94): the voice path speaks no Chinese text.

Run: python3 -m pytest --noconftest -p no:cacheprovider tests_alpha2/test_english_texts.py
The test reads the source with ast. It imports no server package.
"""
import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CJK = re.compile(r"[一-鿿]")

# Files that carry a text that the robot speaks, or a text that the LLM gets back from a tool.
VOICE_FILES = [
    "core/connection.py",
    "core/handle/helloHandle.py",
    "core/handle/intentHandler.py",
    "core/handle/receiveAudioHandle.py",
    "core/handle/textHandler/listenMessageHandler.py",
    "core/providers/tools/device_mcp/mcp_handler.py",
    "core/providers/tools/device_mcp/mcp_executor.py",
    "core/providers/tools/mcp_endpoint/mcp_endpoint_handler.py",
    "core/providers/tools/mcp_endpoint/mcp_endpoint_executor.py",
    "core/providers/tools/server_mcp/mcp_executor.py",
    "core/providers/tools/server_mcp/mcp_manager.py",
    "core/providers/tools/server_plugins/plugin_executor.py",
    "core/providers/tools/unified_tool_manager.py",
    "core/providers/tools/unified_tool_handler.py",
    "core/utils/current_time.py",
    "core/utils/dialogue.py",
    "core/utils/prompt_manager.py",
    "core/utils/tts.py",
    "plugins_func/functions/handle_exit_intent.py",
]
# Values that must stay in Chinese: comparison values, speaker names, calendar data, source names.
KEEP = {"你的", "未知说话人", "澎湃新闻;百度热搜;财联社", "%s年%s%s"}
LOG_METHODS = {"info", "debug", "warning", "error", "exception", "critical", "print"}


def _chinese_texts(path):
    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf8").read())
    skip = set()
    for node in ast.walk(tree):
        is_scope = isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
        if is_scope and node.body and isinstance(node.body[0], ast.Expr):
            skip.add(id(node.body[0].value))
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in LOG_METHODS:
                skip.update(id(n) for n in ast.walk(node))
    return [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and CJK.search(n.value)
        and id(n) not in skip
        and n.value not in KEEP
    ]


def test_voice_path_has_no_chinese_text():
    found = {p: _chinese_texts(p) for p in VOICE_FILES}
    assert {p: v for p, v in found.items() if v} == {}


def test_tool_error_is_a_short_english_sentence():
    for path in (
        "core/providers/tools/device_mcp/mcp_handler.py",
        "core/providers/tools/mcp_endpoint/mcp_endpoint_handler.py",
    ):
        source = open(os.path.join(ROOT, path), encoding="utf8").read()
        assert 'raise RuntimeError("I could not do that.")' in source
        assert "工具调用错误" not in source
