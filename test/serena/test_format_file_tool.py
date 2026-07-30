"""
Tool-layer tests for `FormatFileTool` (SPEC-format-file.md rev 2 + rev 2.1 addendum,
rows T1-T6, T8-T12 incl. T3b/T3c/T3d/T4b/T5a-c).

Layer key (per the spec): T = tool, mocked or hybrid LS, tmp project. T1/T11/T12 use the hybrid
seam (real, live pyright on a tmp copy of the python test_repo, with `request_document_formatting`
and `get_server_capabilities` stubbed) per the spec's explicit notes -- a fully mocked editor
would make them vacuous. The remaining rows use a fully mocked language server (fake object
returned by a patched `create_language_server_symbol_retriever`), which is legitimate for those
rows since none of them reach the real edit-application/disk-write path.

Frozen strings (spec section D) are TEST-OWNED below: deliberately duplicated literals, compared
by EQUALITY after placeholder substitution. Importing the production constants into the oracle is
forbidden (rev 2.1) -- an implementation that changes any frozen string must fail here.

`FormatFileTool` does not exist on the base tree, so every test imports it lazily (inside the
test body, never at module level) so that `pytest --collect-only` succeeds and each test fails at
run time with the declared ImportError, not a collection error.
"""

from __future__ import annotations

import shutil
from unittest.mock import MagicMock, patch

import pytest

from serena.config.serena_config import LanguageBackend, SerenaConfig
from serena.project import Project
from solidlsp.ls_config import LanguageServerId
from solidlsp.lsp_protocol_handler.server import LSPError
from test.conftest import get_repo_path

# --- test-owned frozen strings (spec section D; duplicated on purpose, do NOT import) ---------
UNSUPPORTED_MSG = (
    "The {ls_id} language server does not support document formatting "
    "(no documentFormattingProvider capability); format_file is unavailable for this file. "
    "Use an external formatter instead."
)
NO_CHANGES_MSG = "The language server returned no formatting edits; file unchanged."
SUCCESS_MSG_TEMPLATE = "Successfully formatted {relative_path} ({num_edits} edits applied)."

# `{ls_id}` substitution: the routed server's identifier string as used in other tool messages;
# grounded in `SolidLanguageServer.ls_id.value` (see e.g. ls_manager logging / cli usage).
_FAKE_LS_ID = LanguageServerId.PYTHON


def _unsupported_for_fake_ls() -> str:
    return UNSUPPORTED_MSG.format(ls_id=_FAKE_LS_ID.value)


class _ExplodingSend:
    """Transport-boundary tripwire: ANY `server.send.<method>(...)` call raises. Used by the
    never-sent rows (T3/T3b/T3d) so that even an implementation bypassing
    `request_document_formatting` and talking to the transport directly is caught."""

    def __getattr__(self, name: str):
        def _explode(*args, **kwargs):
            raise AssertionError(f"transport must not be used when formatting is unsupported (server.send.{name} was called)")

        return _explode


def _arm_exploding_transport(fake_ls: MagicMock) -> None:
    fake_ls.server.send = _ExplodingSend()
    fake_ls.server.send_request = MagicMock(
        side_effect=AssertionError("transport must not be used when formatting is unsupported (server.send_request was called)")
    )


def _make_lsp_agent(project: Project) -> MagicMock:
    """A MagicMock agent wired just enough for `Component.create_ls_code_editor` /
    `create_language_server_symbol_retriever` / `create_code_editor` to route through the LSP
    branch, mirroring the `read_file_tool` fixture pattern in `test/serena/test_file_tools.py`.
    """
    agent = MagicMock()
    agent.get_active_project_or_raise.return_value = project
    agent.get_language_backend.return_value = LanguageBackend.LSP
    agent.is_using_language_server.return_value = True
    return agent


def _make_untouchable_retriever(project: Project) -> MagicMock:
    """A fake symbol retriever whose `get_language_server` raises if ever invoked -- used by
    T5a/T5b/T5c/T9 to prove that path-validation failures are raised before any language-server
    resolution or capability check (spec step 1: "Path validation BEFORE anything else")."""
    retriever = MagicMock()
    retriever.project = project

    def _must_not_be_called(relative_path: str):
        raise AssertionError(f"language server must not be resolved for an invalid path, got {relative_path!r}")

    retriever.get_language_server.side_effect = _must_not_be_called
    return retriever


def _build_mocked_tool(tool_class, tmp_path, file_content: str = "original content\n"):
    """Builds a `FormatFileTool` over a real tmp `Project`, plus a fully mocked `SolidLanguageServer`
    double (`fake_ls`) reachable via a fake symbol retriever (`fake_retriever`). Callers patch
    `tool.create_language_server_symbol_retriever` to return `fake_retriever` for the duration of
    the call under test.
    """
    sample_path = tmp_path / "sample.py"
    sample_path.write_text(file_content, encoding="utf-8")

    project = Project.load(str(tmp_path), serena_config=SerenaConfig(gui_log_window=False, web_dashboard=False))
    agent = _make_lsp_agent(project)
    tool = tool_class(agent)
    tool._limit_length = lambda result, max_answer_chars: result

    fake_ls = MagicMock()
    fake_ls.ls_id = _FAKE_LS_ID
    fake_retriever = MagicMock()
    fake_retriever.project = project
    fake_retriever.get_language_server.return_value = fake_ls

    return tool, fake_ls, fake_retriever, sample_path


# ---------------------------------------------------------------------------
# hybrid live-LS project (T1, T11, T12): one pyright startup per module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_python_project(tmp_path_factory: pytest.TempPathFactory):
    """A live-pyright project over a tmp COPY of the python test_repo (spec T1 seam strategy);
    module-scoped so T1/T11/T12 share one language-server startup."""
    src_repo = get_repo_path(LanguageServerId.PYTHON)
    tmp_repo = tmp_path_factory.mktemp("format-file-hybrid") / "test_repo"
    shutil.copytree(src_repo, tmp_repo, ignore=shutil.ignore_patterns(".serena"))
    project = Project.load(str(tmp_repo), serena_config=SerenaConfig(gui_log_window=False, web_dashboard=False))
    project.project_config.language_servers = [LanguageServerId.PYTHON]
    project.create_language_server_manager()  # starts pyright synchronously
    try:
        yield project, tmp_repo
    finally:
        project.shutdown(timeout=5)


def _live_tool(tool_class, project: Project, bypass_limit: bool = True):
    agent = _make_lsp_agent(project)
    tool = tool_class(agent)
    if bypass_limit:
        tool._limit_length = lambda result, max_answer_chars: result
    return tool


# same-position pair (mirrors U6) so a naive edit applier diverges from the expected literal
_HYBRID_EDITS = [
    {"range": {"start": {"line": 0, "character": 1}, "end": {"line": 0, "character": 1}}, "newText": "X"},
    {"range": {"start": {"line": 0, "character": 1}, "end": {"line": 0, "character": 1}}, "newText": "Y"},
]
_HYBRID_ORIGINAL = "ab\n"
# TEST-OWNED expected result of applying _HYBRID_EDITS to _HYBRID_ORIGINAL (rev 2.1: must NOT be
# recomputed via the production apply_text_edits_to_text -- that would be circular):
_HYBRID_EXPECTED_BYTES = b"aXYb\n"


@pytest.mark.python
def test_t1_supported_caps_writes_pure_function_result_to_disk(live_python_project) -> None:
    from serena.tools.file_tools import FormatFileTool

    project, tmp_repo = live_python_project
    target = "format_me_t1.py"
    (tmp_repo / target).write_text(_HYBRID_ORIGINAL, encoding="utf-8")

    tool = _live_tool(FormatFileTool, project)
    ls = project.get_language_server_manager_or_raise().get_language_server(target)
    with (
        patch.object(ls, "get_server_capabilities", return_value={"documentFormattingProvider": True}),
        patch.object(ls, "request_document_formatting", return_value=list(_HYBRID_EDITS)) as request_spy,
        patch.object(tool, "create_code_editor", wraps=tool.create_code_editor) as editor_spy,
    ):
        result = tool.apply(target)

    request_spy.assert_called_once()
    # the STANDARD editing context must be used (EditedFileContext over create_code_editor);
    # a bare Path.write_text implementation must fail here (rev 2.1)
    assert editor_spy.called, "expected the tool to write through create_code_editor()/EditedFileContext"

    # BYTES on disk (encoding + trailing newline), against the test-owned literal
    assert (tmp_repo / target).read_bytes() == _HYBRID_EXPECTED_BYTES

    assert result == SUCCESS_MSG_TEMPLATE.format(relative_path=target, num_edits=len(_HYBRID_EDITS))


@pytest.mark.python
def test_t11_edits_producing_identical_text_yield_no_changes_and_no_write(live_python_project) -> None:
    from serena.tools.file_tools import FormatFileTool

    project, tmp_repo = live_python_project
    target = "format_me_t11.py"
    target_path = tmp_repo / target
    target_path.write_text("ab\n", encoding="utf-8")
    original_bytes = target_path.read_bytes()
    original_mtime_ns = target_path.stat().st_mtime_ns

    # a non-empty edit list whose application reproduces the original text exactly
    identity_edits = [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 2}}, "newText": "ab"}]

    tool = _live_tool(FormatFileTool, project)
    ls = project.get_language_server_manager_or_raise().get_language_server(target)
    with (
        patch.object(ls, "get_server_capabilities", return_value={"documentFormattingProvider": True}),
        patch.object(ls, "request_document_formatting", return_value=identity_edits),
    ):
        result = tool.apply(target)

    assert result == NO_CHANGES_MSG
    assert target_path.read_bytes() == original_bytes
    assert target_path.stat().st_mtime_ns == original_mtime_ns, "file must not be rewritten when the text is unchanged"


@pytest.mark.python
def test_t12_max_answer_chars_limiting_applies_to_the_result(live_python_project) -> None:
    """Deliberately does NOT bypass `_limit_length` (rev 2.1): the standard limiting behavior of
    `Tool._limit_length` must apply to format_file's result. The success path is used because it
    is the one return the spec unambiguously routes through step 8's `_limit_length`."""
    from serena.tools.file_tools import FormatFileTool

    project, tmp_repo = live_python_project
    target = "format_me_t12.py"
    (tmp_repo / target).write_text(_HYBRID_ORIGINAL, encoding="utf-8")

    tool = _live_tool(FormatFileTool, project, bypass_limit=False)
    ls = project.get_language_server_manager_or_raise().get_language_server(target)

    expected_full = SUCCESS_MSG_TEMPLATE.format(relative_path=target, num_edits=len(_HYBRID_EDITS))
    max_answer_chars = 10
    assert len(expected_full) > max_answer_chars  # sanity: limiting will actually trigger

    with (
        patch.object(ls, "get_server_capabilities", return_value={"documentFormattingProvider": True}),
        patch.object(ls, "request_document_formatting", return_value=list(_HYBRID_EDITS)),
    ):
        result = tool.apply(target, max_answer_chars=max_answer_chars)

    # standard Tool._limit_length behavior for an over-long result with no shortening factories
    expected_limited = (
        f"The answer is too long ({len(expected_full)} characters). "
        "You can adjust your query or raise the max_answer_chars parameter."
    )
    assert result == expected_limited


# ---------------------------------------------------------------------------
# T2, T3, T3b, T3c, T3d, T4, T4b, T8, T10: mocked-LS tool tests
# ---------------------------------------------------------------------------


def test_t2_supported_caps_no_edits_leaves_disk_untouched(tmp_path) -> None:
    from serena.tools.file_tools import FormatFileTool

    tool, fake_ls, fake_retriever, sample_path = _build_mocked_tool(FormatFileTool, tmp_path)
    fake_ls.get_server_capabilities.return_value = {"documentFormattingProvider": True}
    original_bytes = sample_path.read_bytes()

    with patch.object(tool, "create_language_server_symbol_retriever", return_value=fake_retriever):
        for edits_value in ([], None):
            fake_ls.request_document_formatting.return_value = edits_value
            result = tool.apply("sample.py")
            assert result == NO_CHANGES_MSG
            assert sample_path.read_bytes() == original_bytes


def test_t3_documentOnTypeFormattingProvider_does_not_satisfy_gate_and_no_request_is_sent(tmp_path) -> None:
    from serena.tools.file_tools import FormatFileTool

    tool, fake_ls, fake_retriever, sample_path = _build_mocked_tool(FormatFileTool, tmp_path)
    # documentFormattingProvider absent; documentOnTypeFormattingProvider present -- must NOT satisfy the gate
    fake_ls.get_server_capabilities.return_value = {"documentOnTypeFormattingProvider": True}
    _arm_exploding_transport(fake_ls)
    original_bytes = sample_path.read_bytes()

    with patch.object(tool, "create_language_server_symbol_retriever", return_value=fake_retriever):
        result = tool.apply("sample.py")

    fake_ls.request_document_formatting.assert_not_called()
    assert result == _unsupported_for_fake_ls()
    assert sample_path.read_bytes() == original_bytes


def test_t3b_explicit_false_capability_is_unsupported_and_no_request_is_sent(tmp_path) -> None:
    from serena.tools.file_tools import FormatFileTool

    tool, fake_ls, fake_retriever, sample_path = _build_mocked_tool(FormatFileTool, tmp_path)
    fake_ls.get_server_capabilities.return_value = {"documentFormattingProvider": False}
    _arm_exploding_transport(fake_ls)
    original_bytes = sample_path.read_bytes()

    with patch.object(tool, "create_language_server_symbol_retriever", return_value=fake_retriever):
        result = tool.apply("sample.py")

    fake_ls.request_document_formatting.assert_not_called()
    assert result == _unsupported_for_fake_ls()
    assert sample_path.read_bytes() == original_bytes


def test_t3c_empty_options_object_is_supported_and_request_is_sent(tmp_path) -> None:
    """`{"documentFormattingProvider": {}}` is a valid (empty) DocumentFormattingOptions object --
    SUPPORTED. Kills 'truthy-check' gate implementations, which would treat {} as falsy."""
    from serena.tools.file_tools import FormatFileTool

    tool, fake_ls, fake_retriever, sample_path = _build_mocked_tool(FormatFileTool, tmp_path)
    fake_ls.get_server_capabilities.return_value = {"documentFormattingProvider": {}}
    fake_ls.request_document_formatting.return_value = None
    original_bytes = sample_path.read_bytes()

    with patch.object(tool, "create_language_server_symbol_retriever", return_value=fake_retriever):
        result = tool.apply("sample.py")

    fake_ls.request_document_formatting.assert_called_once()
    assert result == NO_CHANGES_MSG
    assert sample_path.read_bytes() == original_bytes


def test_t3d_range_only_capability_is_unsupported_and_no_request_is_sent(tmp_path) -> None:
    """`documentRangeFormattingProvider` alone must NOT satisfy the whole-document gate. Kills
    'key-exists-somewhere' / substring-matching gate implementations."""
    from serena.tools.file_tools import FormatFileTool

    tool, fake_ls, fake_retriever, sample_path = _build_mocked_tool(FormatFileTool, tmp_path)
    fake_ls.get_server_capabilities.return_value = {"documentRangeFormattingProvider": True}
    _arm_exploding_transport(fake_ls)
    original_bytes = sample_path.read_bytes()

    with patch.object(tool, "create_language_server_symbol_retriever", return_value=fake_retriever):
        result = tool.apply("sample.py")

    fake_ls.request_document_formatting.assert_not_called()
    assert result == _unsupported_for_fake_ls()
    assert sample_path.read_bytes() == original_bytes


def test_t4_mapped_method_not_found_yields_unsupported_message(tmp_path) -> None:
    from solidlsp.ls_exceptions import SolidLSPException

    from serena.tools.file_tools import FormatFileTool

    tool, fake_ls, fake_retriever, sample_path = _build_mocked_tool(FormatFileTool, tmp_path)
    fake_ls.get_server_capabilities.return_value = {"documentFormattingProvider": True}
    # fabricate the exception exactly as the (spec section B) mapped one appears: message contains
    # the mapped phrase AND cause carries the -32601 LSPError -- so a conforming tool may identify
    # the condition via either signal (rev 2.1).
    fake_ls.request_document_formatting.side_effect = SolidLSPException(
        "The python language server does not support textDocument/formatting",
        cause=LSPError(-32601, "Method not found"),
    )
    original_bytes = sample_path.read_bytes()

    with patch.object(tool, "create_language_server_symbol_retriever", return_value=fake_retriever):
        result = tool.apply("sample.py")

    assert result == _unsupported_for_fake_ls()
    assert sample_path.read_bytes() == original_bytes


def test_t4b_other_lsp_error_propagates_out_of_the_tool_unchanged(tmp_path) -> None:
    """A non--32601 SolidLSPException must NOT be converted to UNSUPPORTED_MSG -- it propagates
    out of the tool as the same instance. Kills catch-all-to-unsupported implementations."""
    from solidlsp.ls_exceptions import SolidLSPException

    from serena.tools.file_tools import FormatFileTool

    tool, fake_ls, fake_retriever, sample_path = _build_mocked_tool(FormatFileTool, tmp_path)
    fake_ls.get_server_capabilities.return_value = {"documentFormattingProvider": True}
    original_exception = SolidLSPException(
        "Error processing request textDocument/formatting with params:\n...",
        cause=LSPError(-32603, "Internal error"),
    )
    fake_ls.request_document_formatting.side_effect = original_exception
    original_bytes = sample_path.read_bytes()

    with patch.object(tool, "create_language_server_symbol_retriever", return_value=fake_retriever):
        with pytest.raises(SolidLSPException) as exc_info:
            tool.apply("sample.py")

    assert exc_info.value is original_exception
    assert sample_path.read_bytes() == original_bytes


def test_t8_caps_never_captured_still_attempts_request(tmp_path) -> None:
    from serena.tools.file_tools import FormatFileTool

    tool, fake_ls, fake_retriever, sample_path = _build_mocked_tool(FormatFileTool, tmp_path)
    fake_ls.get_server_capabilities.return_value = None  # UNKNOWN state: never captured
    fake_ls.request_document_formatting.return_value = None
    original_bytes = sample_path.read_bytes()

    with patch.object(tool, "create_language_server_symbol_retriever", return_value=fake_retriever):
        result = tool.apply("sample.py")

    # UNKNOWN must never be treated as unsupported: the request has to actually be attempted
    fake_ls.request_document_formatting.assert_called_once()
    assert result == NO_CHANGES_MSG
    assert sample_path.read_bytes() == original_bytes


def test_t10_ls_sync_file_system_changes_called_before_gate_and_request(tmp_path) -> None:
    from serena.tools.file_tools import FormatFileTool

    tool, fake_ls, fake_retriever, sample_path = _build_mocked_tool(FormatFileTool, tmp_path)
    project = fake_retriever.project

    events: list[str] = []

    def _record_caps():
        events.append("caps")
        return {"documentFormattingProvider": True}

    def _record_request(relative_path: str):
        events.append("request")
        return None

    fake_ls.get_server_capabilities.side_effect = _record_caps
    fake_ls.request_document_formatting.side_effect = _record_request

    with (
        patch.object(project, "ls_sync_file_system_changes", side_effect=lambda: events.append("sync") or 0) as sync_spy,
        patch.object(tool, "create_language_server_symbol_retriever", return_value=fake_retriever),
    ):
        result = tool.apply("sample.py")

    sync_spy.assert_called()
    assert "caps" in events and "request" in events
    assert events.index("sync") < events.index("caps"), f"ls_sync_file_system_changes must precede the capability gate, got {events}"
    assert events.index("sync") < events.index("request"), f"ls_sync_file_system_changes must precede the request, got {events}"
    assert result == NO_CHANGES_MSG


# ---------------------------------------------------------------------------
# T5a, T5b, T5c, T9: path validation before anything else
# ---------------------------------------------------------------------------


def test_t5a_nonexistent_path_raises_file_not_found_before_capability_check(tmp_path) -> None:
    from serena.tools.file_tools import FormatFileTool

    project = Project.load(str(tmp_path), serena_config=SerenaConfig(gui_log_window=False, web_dashboard=False))
    agent = _make_lsp_agent(project)
    tool = FormatFileTool(agent)
    tool._limit_length = lambda result, max_answer_chars: result

    relative_path = "does_not_exist.py"
    with patch.object(tool, "create_language_server_symbol_retriever", return_value=_make_untouchable_retriever(project)):
        with pytest.raises(FileNotFoundError) as exc_info:
            tool.apply(relative_path)

    assert relative_path in str(exc_info.value)


def test_t5b_directory_path_raises_value_error(tmp_path) -> None:
    from serena.tools.file_tools import FormatFileTool

    (tmp_path / "a_directory").mkdir()
    project = Project.load(str(tmp_path), serena_config=SerenaConfig(gui_log_window=False, web_dashboard=False))
    agent = _make_lsp_agent(project)
    tool = FormatFileTool(agent)
    tool._limit_length = lambda result, max_answer_chars: result

    relative_path = "a_directory"
    with patch.object(tool, "create_language_server_symbol_retriever", return_value=_make_untouchable_retriever(project)):
        with pytest.raises(ValueError) as exc_info:
            tool.apply(relative_path)

    assert relative_path in str(exc_info.value)


def test_t5c_external_path_raises_value_error(tmp_path) -> None:
    from serena.tools.file_tools import FormatFileTool

    project = Project.load(str(tmp_path), serena_config=SerenaConfig(gui_log_window=False, web_dashboard=False))
    agent = _make_lsp_agent(project)
    tool = FormatFileTool(agent)
    tool._limit_length = lambda result, max_answer_chars: result

    relative_path = "<ext:FileUtil.class|472e0a13>"
    with patch.object(tool, "create_language_server_symbol_retriever", return_value=_make_untouchable_retriever(project)):
        with pytest.raises(ValueError) as exc_info:
            tool.apply(relative_path)

    assert relative_path in str(exc_info.value)


def test_t9_ignored_file_is_rejected_like_other_editing_tools(tmp_path) -> None:
    """Ignored files raise ValueError, the same error class `ReplaceContentTool` produces via
    `validate_relative_path(..., require_not_ignored=True)` (verified against the current tree:
    "Path secret.py is ignored; cannot access for safety reasons")."""
    from serena.tools.file_tools import FormatFileTool

    (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "secret.py").write_text("x = 1\n", encoding="utf-8")
    project = Project.load(str(tmp_path), serena_config=SerenaConfig(gui_log_window=False, web_dashboard=False))
    agent = _make_lsp_agent(project)
    tool = FormatFileTool(agent)
    tool._limit_length = lambda result, max_answer_chars: result

    with patch.object(tool, "create_language_server_symbol_retriever", return_value=_make_untouchable_retriever(project)):
        with pytest.raises(ValueError) as exc_info:
            tool.apply("secret.py")

    assert "secret.py" in str(exc_info.value)


# ---------------------------------------------------------------------------
# T6: registration & config
# ---------------------------------------------------------------------------


def test_t6_tool_registered_beta_can_edit_and_excluded_in_jetbrains_and_planning() -> None:
    from serena.config.context_mode import SerenaAgentMode
    from serena.tools import ToolRegistry
    from serena.tools.tools_base import ToolMarkerBeta

    # this is the entry point for the row: on base, "format_file" is not a registered tool name
    tool_class = ToolRegistry().get_tool_class_by_name("format_file")

    assert tool_class.get_name_from_cls() == "format_file"
    assert issubclass(tool_class, ToolMarkerBeta)
    assert tool_class.can_edit() is True
    # default-enabled: not in the optional set
    assert "format_file" not in ToolRegistry().get_tool_names_optional()
    assert "format_file" in ToolRegistry().get_tool_names_default_enabled()

    jetbrains_mode = SerenaAgentMode.from_name_internal("jetbrains")
    assert "format_file" in jetbrains_mode.excluded_tools

    planning_mode = SerenaAgentMode.from_name("planning")
    assert "format_file" in planning_mode.excluded_tools
