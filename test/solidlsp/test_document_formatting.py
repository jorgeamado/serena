"""
Unit tests for the document-formatting additions to `solidlsp` (SPEC-format-file.md rev 2,
rows U1-U9):

  * `SolidLanguageServer.request_document_formatting` (U1-U3)
  * `SolidLanguageServer.get_server_capabilities` (U4, U8), captured centrally in
    `LanguageServerInterface.send_request` (src/solidlsp/ls_process.py:352)
  * the pure function `apply_text_edits_to_text` (U5, U6, U7, U9)

No language markers: these use local test doubles (mirroring
`test/solidlsp/test_rename_didopen.py` and `test/solidlsp/test_content_modified_retry.py`) and
run in catch-all.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import LanguageServerId
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.ls_process import LanguageServerInterface
from solidlsp.lsp_protocol_handler.server import LSPError


def _import_apply_text_edits_to_text() -> Callable[[str, list[dict[str, Any]]], str]:
    """
    `apply_text_edits_to_text` is frozen only by exported name (SPEC-format-file.md section B2);
    its module location is the implementer's choice between `solidlsp.ls_utils` and `solidlsp.ls`.
    Try both so the test fails with a plain ImportError (missing function) on the current tree,
    rather than an arbitrary AttributeError tied to one guessed location.
    """
    try:
        from solidlsp.ls_utils import apply_text_edits_to_text  # type: ignore[attr-defined]

        return apply_text_edits_to_text
    except ImportError:
        pass
    from solidlsp.ls import apply_text_edits_to_text  # type: ignore[attr-defined]

    return apply_text_edits_to_text


class DummyLanguageServer(SolidLanguageServer):
    """Minimal concrete `SolidLanguageServer` subclass for constructing bare instances via
    `object.__new__`, exactly as `test/solidlsp/test_rename_didopen.py` does."""

    def _start_server(self) -> None:
        raise AssertionError("Not used in this test")

    def _create_base_initialize_params(self) -> dict:
        return {}


def _make_ls(tmp_path, send: MagicMock | None = None) -> SolidLanguageServer:
    server = MagicMock()
    server.notify = MagicMock()
    server.send = send if send is not None else MagicMock()

    ls = object.__new__(DummyLanguageServer)
    ls.repository_root_path = str(tmp_path)
    ls.server_started = True
    ls.open_file_buffers = {}
    ls._encoding = "utf-8"
    ls.language_id = "python"
    ls.server = server
    return ls


# ---------------------------------------------------------------------------
# U1-U3: request_document_formatting
# ---------------------------------------------------------------------------


def test_u1_request_document_formatting_sends_correct_method_and_params_and_returns_edits_verbatim(tmp_path) -> None:
    (tmp_path / "sample.py").write_text("x=1\n", encoding="utf-8")
    canned_edits = [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}, "newText": "y"}]

    # event trace proving the file is OPEN in the LS while the request is made (rev 2.1 U1
    # extension), same technique as test/solidlsp/test_rename_didopen.py
    events: list[str] = []
    notify = MagicMock()
    notify.did_open_text_document.side_effect = lambda *_a, **_k: events.append("didOpen")
    notify.did_close_text_document.side_effect = lambda *_a, **_k: events.append("didClose")

    send = MagicMock()

    def _formatting(params):
        events.append("formatting")
        return canned_edits

    send.formatting.side_effect = _formatting

    ls = _make_ls(tmp_path, send=send)
    ls.server.notify = notify
    expected_uri = ls._resolve_file_uri("sample.py")

    result = ls.request_document_formatting("sample.py")

    send.formatting.assert_called_once_with(
        {
            "textDocument": {"uri": expected_uri},
            "options": {"tabSize": 4, "insertSpaces": True},
        }
    )
    assert result is canned_edits
    assert events == ["didOpen", "formatting", "didClose"], f"request must be made with the file open, got {events}"

    # rev 2.1: module-level constants exist (spec section B places them in src/solidlsp/ls.py)
    # and feed the params asserted above
    import solidlsp.ls as ls_module

    assert getattr(ls_module, "FORMATTING_TAB_SIZE") == 4
    assert getattr(ls_module, "FORMATTING_INSERT_SPACES") is True


def test_u2_lsp_error_method_not_found_is_mapped_to_unsupported_message(tmp_path) -> None:
    (tmp_path / "sample.py").write_text("x=1\n", encoding="utf-8")
    send = MagicMock()
    # per ls_process.py:379, the transport always wraps a raw LSPError in SolidLSPException(cause=...)
    send.formatting.side_effect = SolidLSPException(
        "Error processing request textDocument/formatting with params:\n...",
        cause=LSPError(-32601, "Method not found"),
    )

    ls = _make_ls(tmp_path, send=send)

    with pytest.raises(SolidLSPException, match="does not support textDocument/formatting"):
        ls.request_document_formatting("sample.py")


def test_u3_other_lsp_error_propagates_unchanged(tmp_path) -> None:
    (tmp_path / "sample.py").write_text("x=1\n", encoding="utf-8")
    original_cause = LSPError(-32603, "Internal error")
    original_exception = SolidLSPException(
        "Error processing request textDocument/formatting with params:\n...",
        cause=original_cause,
    )
    send = MagicMock()
    send.formatting.side_effect = original_exception

    ls = _make_ls(tmp_path, send=send)

    with pytest.raises(SolidLSPException) as exc_info:
        ls.request_document_formatting("sample.py")

    # "SAME exception propagates (instance or cause preserved, message unmodified)" -- assert the
    # strictest reading (identity) since a correct implementation only special-cases -32601 and
    # otherwise does not catch-and-rewrap at all.
    assert exc_info.value is original_exception
    assert exc_info.value.cause is original_cause
    assert "does not support textDocument/formatting" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# U4, U8: get_server_capabilities, captured in LanguageServerInterface.send_request
# ---------------------------------------------------------------------------


class _ScriptedHandler(LanguageServerInterface):
    """Test double answering `initialize` synchronously from a canned payload, without a real
    language server process -- mirrors `test/solidlsp/test_content_modified_retry.py`'s
    `_ScriptedServer`, since exercising the REAL `send_request` (not a stub) is required to test
    where SPEC-format-file.md rev 2 now places the capability-capture logic.
    """

    def __init__(self, initialize_payload: dict) -> None:
        super().__init__(LanguageServerId.PYTHON, lambda _line: logging.INFO)
        self._initialize_payload = initialize_payload

    def is_running(self) -> bool:
        return True

    def _start(self) -> None:
        pass

    def _stop(self, timeout: float) -> None:
        pass

    def _send_payload(self, payload: dict) -> None:
        request = self._pending_requests[payload["id"]]
        if payload.get("method") == "initialize":
            request.on_result(self._initialize_payload)
        else:
            request.on_result(None)


def test_u4_get_server_capabilities_returns_captured_dict_after_initialize_via_wrapper() -> None:
    handler = _ScriptedHandler({"capabilities": {"documentFormattingProvider": True, "hoverProvider": {"x": 1}}})
    handler.send.initialize({})  # type: ignore[arg-type]  # via the LanguageServerRequest wrapper

    ls = object.__new__(DummyLanguageServer)
    ls.server = handler

    assert ls.get_server_capabilities() == {"documentFormattingProvider": True, "hoverProvider": {"x": 1}}


def test_u4_get_server_capabilities_returns_none_when_nothing_captured() -> None:
    handler = _ScriptedHandler({"capabilities": {"documentFormattingProvider": True}})
    # note: initialize() is deliberately never called on this handler

    ls = object.__new__(DummyLanguageServer)
    ls.server = handler

    assert ls.get_server_capabilities() is None


def test_u8_get_server_capabilities_captured_via_direct_send_request_dart_al_idiom() -> None:
    """dart_language_server.py and al_language_server.py call `send_request("initialize", ...)`
    directly, bypassing the `LanguageServerRequest.initialize` wrapper -- capture must live in
    `LanguageServerInterface.send_request` itself (ls_process.py:352) so this idiom is covered too.
    """
    handler = _ScriptedHandler({"capabilities": {"documentFormattingProvider": {"foo": "bar"}}})
    handler.send_request("initialize", {})  # direct call, NOT via handler.send.initialize(...)

    ls = object.__new__(DummyLanguageServer)
    ls.server = handler

    assert ls.get_server_capabilities() == {"documentFormattingProvider": {"foo": "bar"}}


# ---------------------------------------------------------------------------
# U5, U6, U7, U9: apply_text_edits_to_text (pure function)
# ---------------------------------------------------------------------------


def test_u5_astral_char_edit_uses_utf16_offsets() -> None:
    apply_text_edits_to_text = _import_apply_text_edits_to_text()

    # "😀" (U+1F600) is one Python code point but two UTF-16 code units, so any edit positioned
    # after it on the line must account for the surrogate pair to land on the right character.
    text = 'x = "😀"; y = 1'
    # UTF-16 unit offsets: ... '"'=7, ';'=8, ' '=9, 'y'=10, ' '=11, '='=12, ' '=13, '1'=14
    edits = [{"range": {"start": {"line": 0, "character": 10}, "end": {"line": 0, "character": 11}}, "newText": "z"}]

    result = apply_text_edits_to_text(text, edits)

    assert result == 'x = "😀"; z = 1'


def test_u6_same_position_inserts_preserve_array_order() -> None:
    apply_text_edits_to_text = _import_apply_text_edits_to_text()

    text = "12"
    pos = {"line": 0, "character": 1}
    edits = [
        {"range": {"start": pos, "end": pos}, "newText": "A"},
        {"range": {"start": pos, "end": pos}, "newText": "B"},
    ]

    result = apply_text_edits_to_text(text, edits)

    assert result == "1AB2"


def test_u7_insert_and_replace_sharing_a_start_position() -> None:
    apply_text_edits_to_text = _import_apply_text_edits_to_text()

    text = "abcd"
    start = {"line": 0, "character": 1}
    insert_edit = {"range": {"start": start, "end": start}, "newText": "PRE"}
    replace_edit = {"range": {"start": start, "end": {"line": 0, "character": 3}}, "newText": "X"}
    edits = [insert_edit, replace_edit]

    result = apply_text_edits_to_text(text, edits)

    # insert text precedes replacement content (array order); the replaced range ("bc") is the
    # original one, not re-derived after the insertion.
    assert result == "aPREXd"


def test_u9_empty_edits_returns_text_unchanged() -> None:
    apply_text_edits_to_text = _import_apply_text_edits_to_text()

    text = "hello\nworld\n"

    assert apply_text_edits_to_text(text, []) == text
