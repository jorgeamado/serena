"""Tool-level tests for CallHierarchyTool (spec 7c)."""

import json
import os
import shutil
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from serena.agent import SerenaAgent
from serena.tools import CallHierarchyTool
from serena.tools.tools_base import ToolMarkerBeta, ToolRegistry
from solidlsp.ls_config import LanguageServerId
from test.conftest import agent_for_project_context, get_repo_path

CALL_GRAPH_FILE = os.path.join("test_repo", "call_graph.py")


@pytest.fixture(scope="module")
def python_agent() -> Iterator[SerenaAgent]:
    """A SerenaAgent over the python test repo with a warm language server."""
    with agent_for_project_context(LanguageServerId.PYTHON) as agent:
        yield agent


class TestCallHierarchyToolRegistry:
    def test_t1_registered_default_enabled_and_beta(self) -> None:
        """T1: call_hierarchy is registered, default-enabled, and marked beta."""
        registry = ToolRegistry()
        assert "call_hierarchy" in registry.get_tool_names_default_enabled()
        assert issubclass(registry.get_tool_class_by_name("call_hierarchy"), ToolMarkerBeta)


class TestCallHierarchyToolValidation:
    """Validation must raise before any LS interaction, so a mock agent without any LS suffices."""

    def test_t2_invalid_direction(self) -> None:
        """T2: an invalid direction raises ValueError before any LS call."""
        tool = CallHierarchyTool(MagicMock())
        with pytest.raises(ValueError, match="direction"):
            tool.apply(name_path="leaf", relative_path=CALL_GRAPH_FILE, direction="sideways")

    @pytest.mark.parametrize("depth", [0, 11])
    def test_t3_invalid_depth(self, depth: int) -> None:
        """T3: depth outside [1, 10] raises ValueError."""
        tool = CallHierarchyTool(MagicMock())
        with pytest.raises(ValueError, match="depth"):
            tool.apply(name_path="leaf", relative_path=CALL_GRAPH_FILE, depth=depth)


@pytest.mark.python
class TestCallHierarchyToolWithLanguageServer:
    def test_t4_happy_path_json_structure(self, python_agent: SerenaAgent) -> None:
        """T4: happy path returns JSON with exactly the spec 5 keys and only the requested direction key."""
        tool = python_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="leaf", relative_path=CALL_GRAPH_FILE, direction="incoming", depth=1)

        output = json.loads(result)
        assert set(output.keys()) == {"symbol", "direction", "incoming", "truncated", "external_calls_omitted"}
        assert output["direction"] == "incoming"
        assert "outgoing" not in output

        roots = output["incoming"]
        assert len(roots) == 1
        assert roots[0]["name"] == "leaf"
        assert [child["name"] for child in roots[0]["children"]] == ["mid"]

    def test_t5_tiny_max_answer_chars(self, python_agent: SerenaAgent) -> None:
        """T5: a tiny max_answer_chars returns without exception; shortened form or the generic too-long notice."""
        tool = python_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="leaf", relative_path=CALL_GRAPH_FILE, direction="incoming", depth=1, max_answer_chars=10)

        assert isinstance(result, str)
        assert "The answer is too long" in result or "Call hierarchy" in result

    def test_t6_new_caller_file_found_while_server_warm(self, tmp_path) -> None:
        """T6: a caller file created while the server is warm is found via the tool's ls_sync_file_system_changes path."""
        # work on an isolated copy so we can freely create files under the project root
        repo_root = tmp_path / "repo"
        shutil.copytree(get_repo_path(LanguageServerId.PYTHON), repo_root)
        caller_abs = repo_root / "test_repo" / "external_ch_caller.py"

        with agent_for_project_context(LanguageServerId.PYTHON, str(repo_root)) as agent:
            tool = agent.get_tool(CallHierarchyTool)

            # warm the server and establish the file-watch baseline (first poll never notifies)
            baseline = json.loads(tool.apply(name_path="leaf", relative_path=CALL_GRAPH_FILE, direction="incoming", depth=1))
            baseline_callers = [child["name"] for child in baseline["incoming"][0]["children"]]
            assert "external_ch_caller" not in baseline_callers

            # create a new caller file while the server is warm
            caller_abs.write_text(
                "from test_repo.call_graph import leaf\n\n\ndef external_ch_caller() -> None:\n    leaf()\n",
                encoding="utf-8",
            )

            result = json.loads(tool.apply(name_path="leaf", relative_path=CALL_GRAPH_FILE, direction="incoming", depth=1))
            callers = [child["name"] for child in result["incoming"][0]["children"]]
            assert "external_ch_caller" in callers
