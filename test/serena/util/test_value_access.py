"""Pure unit tests for `serena.util.value_access.classify_member_access`.

Covers SPEC-call-hierarchy-members.md (Rev 2) §6a rows C1-C20: the frozen classifier vocabulary and
the §5 algorithm. NEW signature (Rev 2): classify_member_access(line, char_start, member_name, is_event).

RED-for-base cause (declared): `serena.util.value_access` does not exist on base `main` (new SET A
production module). The import below fails with ModuleNotFoundError, so EVERY test in this file fails
at collection time with the same cause. That collection-wide ImportError IS the declared red cause for
rows C1-C20 -- there is no per-assertion failure to distinguish among them until the module exists.
"""

import re

import pytest

from serena.util.value_access import classify_member_access


def _start(line: str, member_name: str) -> int:
    """0-based column where `member_name` begins in `line`, matched as a whole identifier token."""
    match = re.search(rf"\b{re.escape(member_name)}\b", line)
    assert match is not None, f"member {member_name!r} not found in line {line!r}"
    return match.start()


# (id, line, member_name, is_event, expected, char_start_offset)
# char_start_offset is added to the located start column -- 0 for every real row; C20 deliberately
# passes a mismatched span (off by 3) to exercise the §5 name-guard.
_ROWS: list[tuple[str, str, str, bool, str, int]] = [
    ("C1", "var a = w.Count;", "Count", False, "read", 0),
    ("C2", "w.Count = 5;", "Count", False, "write", 0),
    ("C3", "w.Count += 1;", "Count", False, "read_write", 0),
    ("C4", "w.Count >>>= 1;", "Count", False, "read_write", 0),
    ("C5", "w.Count ??= 2;", "Count", False, "read_write", 0),
    ("C6", "w.Count++;", "Count", False, "read_write", 0),
    ("C7", "++w.Count;", "Count", False, "read_write", 0),
    ("C8", "if (w.Count == 5) {}", "Count", False, "read", 0),
    # C9: the member token is `X`, itself followed by `=>` -- the `=>` guard must keep it `read`.
    ("C9", "int X => w.Count;", "X", False, "read", 0),
    ("C10", "M(out w.Count);", "Count", False, "write", 0),
    ("C11", "M(in w.Count);", "Count", False, "read", 0),
    ("C12", "M(ref w.Count);", "Count", False, "read_write", 0),
    ("C13", "var n = nameof(w.Count);", "Count", False, "unknown", 0),
    ("C14", "(w.Count, w.Other) = pair;", "Count", False, "unknown", 0),
    ("C15", "w.Changed += h;", "Changed", True, "subscribe", 0),
    ("C16", "w.Changed -= h;", "Changed", True, "unsubscribe", 0),
    ("C17", "w.Changed?.Invoke();", "Changed", True, "invoke", 0),
    ("C18", "w.Changed!();", "Changed", True, "invoke", 0),
    ("C19", "w.Changed(a, b);", "Changed", True, "invoke", 0),
    ("C20", "w.Count = 5;", "Count", False, "unknown", 3),
]

_IDS = [row[0] for row in _ROWS]


@pytest.mark.parametrize(("_id", "line", "member_name", "is_event", "expected", "offset"), _ROWS, ids=_IDS)
def test_classify_member_access(_id: str, line: str, member_name: str, is_event: bool, expected: str, offset: int) -> None:
    char_start = _start(line, member_name) + offset
    assert classify_member_access(line, char_start, member_name, is_event) == expected


# --- hover-based symbol-kind detection -------------------------------------------------------

from serena.util.value_access import classify_hover_symbol  # noqa: E402


def _fence(sig: str) -> str:
    return f"```csharp\n{sig}\n```\n  \nSome docs.\n"


_HOVER_ROWS = [
    ("H1", _fence("CancellationToken CancellationTokenSource.Token { get; }"), "Token", "property"),
    ("H2", _fence("CancellationToken CtrlCHook.Token { get; }"), "Token", "property"),
    ("H3", _fence("event Action<CharacterUI> CharacterUI.OnDead"), "OnDead", "event"),
    ("H4", _fence("void Foo.Bar(int x)"), "Bar", "method"),
    ("H5", _fence("int Foo.Field"), "Field", "field"),
    ("H6", _fence("(int, int) Foo.Pair { get; set; }"), "Pair", "property"),  # tuple-return property
    ("H7", "CancellationToken CancellationTokenSource.Token { get; }", "Token", "property"),  # bare, no fence
    ("H8", None, None, "unknown"),
    ("H9", "", None, "unknown"),
]
_HOVER_IDS = [row[0] for row in _HOVER_ROWS]


@pytest.mark.parametrize(("_id", "hover", "expected_name", "expected_kind"), _HOVER_ROWS, ids=_HOVER_IDS)
def test_classify_hover_symbol(_id: str, hover: str | None, expected_name: str | None, expected_kind: str) -> None:
    name, kind = classify_hover_symbol(hover)
    assert (name, kind) == (expected_name, expected_kind)
