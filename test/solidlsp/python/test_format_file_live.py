"""
Live tests for the `format_file` tool against real (host-cheap) Python language servers
(SPEC-format-file.md rev 2 + rev 2.1 addendum, rows L1 and L2): neither pyright nor basedpyright
advertise `documentFormattingProvider`, and basedpyright's `documentOnTypeFormattingProvider`
must not fool the capability gate into claiming support.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from serena.config.serena_config import LanguageBackend, SerenaConfig
from serena.project import Project
from solidlsp.ls_config import LanguageServerId
from test.conftest import get_repo_path

pytestmark = pytest.mark.python

# --- test-owned frozen string (spec section D; duplicated on purpose, do NOT import) ----------
UNSUPPORTED_MSG = (
    "The {ls_id} language server does not support document formatting "
    "(no documentFormattingProvider capability); format_file is unavailable for this file. "
    "Use an external formatter instead."
)

_TARGET_FILE = os.path.join("test_repo", "variables.py")  # existing, small file in the shared fixture


def _live_format_file_tool(tool_class, ls_id: LanguageServerId):
    # get_repo_path aliases PYTHON_BASEDPYRIGHT to the same python/test_repo fixture directory;
    # resolving via the actual ls_id keeps this correct even if that aliasing ever changes.
    project = Project.load(str(get_repo_path(ls_id)), serena_config=SerenaConfig(gui_log_window=False, web_dashboard=False))
    project.project_config.language_servers = [ls_id]
    project.create_language_server_manager()

    agent = MagicMock()
    agent.get_active_project_or_raise.return_value = project
    agent.get_language_backend.return_value = LanguageBackend.LSP
    agent.is_using_language_server.return_value = True

    tool = tool_class(agent)
    tool._limit_length = lambda result, max_answer_chars: result
    return tool, project


def test_l1_format_file_unsupported_for_pyright() -> None:
    from serena.tools.file_tools import FormatFileTool

    tool, project = _live_format_file_tool(FormatFileTool, LanguageServerId.PYTHON)
    try:
        result = tool.apply(_TARGET_FILE)
        ls = project.get_language_server_manager_or_raise().get_language_server(_TARGET_FILE)
        assert result == UNSUPPORTED_MSG.format(ls_id=ls.ls_id.value)
    finally:
        project.shutdown(timeout=5)


def test_l2_format_file_unsupported_for_basedpyright_despite_onType_capability() -> None:
    """basedpyright advertises `documentOnTypeFormattingProvider` but NOT
    `documentFormattingProvider` -- proves the gate is not fooled by the former."""
    from serena.tools.file_tools import FormatFileTool

    tool, project = _live_format_file_tool(FormatFileTool, LanguageServerId.PYTHON_BASEDPYRIGHT)
    try:
        ls = project.get_language_server_manager_or_raise().get_language_server(_TARGET_FILE)
        capabilities = ls.get_server_capabilities() or {}
        assert capabilities.get("documentOnTypeFormattingProvider"), (
            "expected basedpyright to advertise documentOnTypeFormattingProvider (the capability "
            "that must NOT satisfy the format_file gate); if this fails, basedpyright's advertised "
            "capabilities changed and L2 needs to be revisited"
        )
        assert not capabilities.get("documentFormattingProvider")

        result = tool.apply(_TARGET_FILE)
        assert result == UNSUPPORTED_MSG.format(ls_id=ls.ls_id.value)
    finally:
        project.shutdown(timeout=5)
