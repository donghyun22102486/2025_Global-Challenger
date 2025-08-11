# llm_handler.py
import os
import json
import re
from typing import Tuple, Optional, Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv
from string import Template

# ---------------------------------------------------------------------
# Gemini setup
# ---------------------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
model = genai.GenerativeModel(GEMINI_MODEL)


# ---------------------------------------------------------------------
# Helpers: template loader (robust path fallback)
# ---------------------------------------------------------------------
def _read_first_existing(paths: List[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    raise FileNotFoundError(f"Template not found. Tried: {paths}")


def load_prompt_templates(path: str = "prompt_templates/prompt_template.txt") -> str:
    # Backward compatibility shim
    return _read_first_existing([path, "prompt_template.txt"])


# ---------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------
def create_llm_prompt(
    markdown_sample: str, summary_stats: str, user_request: str
) -> str:
    raw_template = _read_first_existing(
        ["prompt_templates/prompt_template.txt", "prompt_template.txt"]
    )
    template = Template(raw_template)
    return template.substitute(
        markdown_sample=markdown_sample,
        summary_stats=summary_stats,
        user_request=user_request or "없음",
    )


def create_prediction_explanation_prompt(
    user_request: str, parsed_input: dict, prediction: float
) -> str:
    raw_template = _read_first_existing(
        [
            "prompt_templates/prediction_explanation_template.txt",
            "prediction_explanation_template.txt",
        ]
    )
    template = Template(raw_template)
    return template.substitute(
        user_request=user_request,
        parsed_input=json.dumps(parsed_input, indent=2, ensure_ascii=False),
        prediction=round(float(prediction), 2),
    )


# ---------------------------------------------------------------------
# Helpers: robust JSON extraction
# ---------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_from_codeblocks(reply: str) -> Optional[Tuple[Any, int, int]]:
    """
    Try to parse JSON from fenced code blocks first.
    Returns (obj, start_idx, end_idx) if success.
    """
    for m in _CODE_BLOCK_RE.finditer(reply):
        content = m.group(1).strip()
        obj = _try_parse_json(content)
        if obj is not None:
            return obj, m.start(), m.end()
    return None


def _find_first_balanced_json(reply: str) -> Optional[Tuple[Any, int, int]]:
    """
    Scan the string to find the first balanced JSON object/array and parse it.
    Handles quotes and escapes to avoid brace counting inside strings.
    Returns (obj, start_idx, end_idx) if success.
    """
    s = reply
    n = len(s)

    i = 0
    while i < n:
        if s[i] in "{[":
            start = i
            stack = [s[i]]
            in_string = False
            escape = False
            j = i + 1
            while j < n:
                c = s[j]
                if in_string:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c in "{[":
                        stack.append(c)
                    elif c in "}]":
                        if not stack:
                            break
                        top = stack[-1]
                        if (top == "{" and c == "}") or (top == "[" and c == "]"):
                            stack.pop()
                            if not stack:
                                # Candidate JSON slice
                                candidate = s[start : j + 1]
                                obj = _try_parse_json(candidate)
                                if obj is not None:
                                    return obj, start, j + 1
                                # If parse fails, continue scanning after start
                                break
                        else:
                            # Mismatched brace; stop this candidate
                            break
                j += 1
        i += 1
    return None


def _extract_json_and_report(reply: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract the first valid JSON (prefers fenced blocks) and the trailing report text.
    - If JSON not found/parsable: returns ({}, full_reply_as_report)
    - If JSON found: returns (json_obj_if_dict_else_{}, trailing_text)
    """
    # 1) Prefer fenced code blocks
    res = _extract_from_codeblocks(reply)
    if res is None:
        # 2) Fallback: brace scanning
        res = _find_first_balanced_json(reply)

    if res is None:
        # No JSON found at all → use entire reply as report text
        return {}, reply.strip()

    obj, start_idx, end_idx = res

    # Ensure dict to avoid downstream attribute errors
    parsed_json: Dict[str, Any] = obj if isinstance(obj, dict) else {}

    # Report text = everything after the parsed JSON
    trailing = reply[end_idx:].strip()

    # If trailing is empty but there are multiple blocks, consider remaining text
    if not trailing:
        trailing = reply[:start_idx].strip()

    return parsed_json, trailing


# ---------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------
def query_llm(prompt: str) -> Tuple[Dict[str, Any], str]:
    """
    Returns (parsed_json: dict, report_text: str)
    - Robust to code fences and non-JSON outputs
    - Never raises for JSON parsing failure; falls back to {} + full text
    - Raises only for transport/model errors
    """
    try:
        response = model.generate_content(prompt)
        reply = (response.text or "").strip()
        if not reply:
            # Return empty JSON + empty report rather than raising
            return {}, ""
        parsed_json, report_text = _extract_json_and_report(reply)
        return parsed_json, report_text
    except Exception as e:
        # Keep the signature consistent; surface error clearly
        raise RuntimeError(f"[Gemini API Error] {e}")


def query_llm_plain(prompt: str) -> str:
    """
    Plain text generation without any JSON parsing.
    """
    try:
        response = model.generate_content(prompt)
        return (response.text or "").strip()
    except Exception as e:
        raise RuntimeError(f"[Gemini API Error] {e}")
