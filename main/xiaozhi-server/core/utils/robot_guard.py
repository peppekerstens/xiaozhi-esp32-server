"""Alpha 2 robot guard (fork branch alpha2/voice-tools, alpha2-brain issue #16).

Settings in ``Intent.function_call`` of the config:

- ``tool_required_patterns``: regular expressions. When the user text matches one,
  the first LLM call of the turn uses ``tool_choice: required``. A small LLM then
  cannot say "I set my eyes to green" without a tool call.
- ``single_tool_call``: true runs only the first tool call of each LLM step.
  Small models repeat the same call many times in one stream.
- ``action_allow``: a map spoken name -> action file (``Squat-myrobot.ubx``).
  The server gives the LLM only these names (an enum) and rejects all other names.
  An empty map removes the play_action tool.
- ``tools_deny``: tool names (fnmatch) that the server never runs.

Setting ``turn_language`` (top level): ``auto`` sets ``conn.turn_language`` to
``nl`` or ``en`` from the user text. The custom TTS uses it to select the voice.

Setting ``voice.languages`` (top level, alpha2-brain#88, owner decision D-62): the
languages that the robot understands and speaks, for example ``[en]``. Voice text
that is not in one of these languages gets no turn (see ``language_check``). The
answer language and the TTS voice stay inside the list. Add ``nl`` to the list to
switch Dutch on again. When the setting is missing, nothing is restricted.
"""

import copy
import fnmatch
import json
import re

PLAY_ACTION = "self_robot_play_action"
RETRY_HINT = "(Call the matching tool now.)"
FAIL_RESPONSES = {
    "en": "Sorry, that did not work. Please say it again.",
    "nl": "Sorry, dat lukte niet. Zeg het nog een keer.",
}


def fail_response(conn):
    """The fixed answer when a required tool call does not come (setting tool_fail_response)."""
    texts = _cfg(conn).get("tool_fail_response") or FAIL_RESPONSES
    lang = getattr(conn, "turn_language", None) or "en"
    return texts.get(lang) or texts.get("en") or FAIL_RESPONSES["en"]

_NL_WORDS = set(
    "de het een en ik je jij u wij we ze is zijn ben bent was wat waar wie hoe "
    "niet geen ook maar dat dit die er naar van voor met op aan uit om te "
    "maak zet doe ga kun kan kunt wil mag moet hebt heb heeft graag alsjeblieft "
    "eens even nu dan nog al mijn jouw jou hallo hoi dag ja nee goed ogen hoofd "
    "licht lampje rood groen blauw geel paars wit afstand muur knieën zitten sta "
    "vertel iets over kun maken jou vandaag waarom wanneer welke weet vind zeg "
    "laat zien toe uit hem haar ons onze jullie bij tot zo heel veel weinig "
    "wel zou zou je zal gaat gaan komt kom moeten willen hebben worden wordt "
    "mooi lekker gezellig koffie morgen vanavond straks nodig eigenlijk tegen "
    "omdat zodat terwijl toch misschien altijd nooit soms helemaal gewoon echt "
    "zeker alles niets iemand mensen kinderen huis buiten binnen werk dank bedankt "
    "weer wij zij mij dus ook nou".split()
)
_EN_WORDS = set(
    "the a an and i you he she we they it about tell like is are am was were what where who how "
    "not no also but that this there to of for with on at from in "
    "make set turn do go can could will would please let me my your hello hi "
    "yes good eyes head light red green blue yellow purple white wall stand sit".split()
)


def _cfg(conn):
    try:
        return conn.config["Intent"][conn.config["selected_module"]["Intent"]] or {}
    except (KeyError, TypeError, AttributeError):
        return {}


def needs_tool(conn, text):
    """True when the user text asks for a robot action (tool_choice required)."""
    if not text:
        return False
    for p in _cfg(conn).get("tool_required_patterns") or []:
        try:
            if re.search(p, text, re.I):
                return True
        except re.error:
            continue
    return False


def single_call(conn):
    return bool(_cfg(conn).get("single_tool_call", False))


def rewrite_functions(conn, functions):
    """Replace the play_action schema with the allowed names, or remove the tool."""
    allow = _cfg(conn).get("action_allow")
    if allow is None or not functions:
        return functions
    out = []
    for f in functions:
        name = f.get("function", {}).get("name", "")
        if name != PLAY_ACTION:
            out.append(f)
            continue
        if not allow:
            continue  # no allowed action: the LLM does not see the tool
        f = copy.deepcopy(f)
        f["function"]["description"] = (
            "Play one body movement of the robot. Use only a name from the list."
        )
        f["function"]["parameters"] = {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": list(allow)}},
            "required": ["name"],
        }
        out.append(f)
    return out


def check_call(conn, name, arguments):
    """Return (arguments, None) to run the call, or (None, reason) to reject it."""
    cfg = _cfg(conn)
    for p in cfg.get("tools_deny") or []:
        if fnmatch.fnmatch(name, p):
            return None, f"The tool {name} is not allowed on this robot."
    if name != PLAY_ACTION or cfg.get("action_allow") is None:
        return arguments, None
    allow = cfg.get("action_allow") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
    spoken = str((arguments or {}).get("name", "")).strip().lower()
    for key, file in allow.items():
        if spoken == key.lower() or spoken == str(file).lower():
            return {"name": file}, None
    return None, (
        f"The movement '{spoken}' is not allowed. Allowed: {', '.join(allow)}."
    )


def call_key(tool_call):
    return (tool_call.get("name"), (tool_call.get("arguments") or "").strip())


def detect_language(text, default="en"):
    """Return 'nl' or 'en' from a word count. Ties give the default."""
    words = re.findall(r"[a-zA-Zà-ÿ']+", (text or "").lower())
    nl = sum(w in _NL_WORDS for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if nl > en:
        return "nl"
    if en > nl:
        return "en"
    return default


def allowed_languages(conn):
    """The set of language codes from ``voice.languages``, or None when it is not set."""
    try:
        langs = (conn.config.get("voice") or {}).get("languages")
    except AttributeError:
        return None
    if isinstance(langs, str):
        langs = [x for x in re.split(r"[,\s]+", langs) if x]
    if not isinstance(langs, (list, tuple)) or not langs:
        return None
    return {str(x).strip().lower() for x in langs}


def language_check(conn, text):
    """Return ``(True, "")`` when the voice text may start a turn, else ``(False, reason)``.

    The ASR (Parakeet) finds the language itself and does not report it. Thus the
    language comes from a word count of the text (``detect_language``). A text with no
    word of either list counts as an allowed language: there is no sign of Dutch.
    """
    allowed = allowed_languages(conn)
    if allowed is None or not text:
        return True, ""
    default = "en" if "en" in allowed else sorted(allowed)[0]
    found = detect_language(text, default)
    if found in allowed:
        return True, ""
    return False, f"language {found} is off (voice.languages: {', '.join(sorted(allowed))})"


def update_turn_language(conn, text):
    mode = str(conn.config.get("turn_language", "")).lower()
    if mode != "auto" or not text:
        return
    allowed = allowed_languages(conn)
    found = detect_language(text, getattr(conn, "turn_language", None) or "en")
    if allowed and found not in allowed:
        found = "en" if "en" in allowed else sorted(allowed)[0]
    conn.turn_language = found
    names = {"nl": "Dutch (Nederlands)", "en": "English"}
    conn.dialogue.answer_language = names.get(conn.turn_language, "English")
    hints = conn.config.get("turn_language_hints") or {"nl": "(Antwoord in het Nederlands.)"}
    conn.dialogue.answer_hint = hints.get(conn.turn_language)
