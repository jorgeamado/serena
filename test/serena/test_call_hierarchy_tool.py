"""Tool-level tests for CallHierarchyTool (spec 7c)."""

import json
import os
import shutil
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

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


# Frozen note text per spec 2.3: f-string template with the ls_id substituted.
APPROXIMATE_NOTE_TEMPLATE = (
    "{ls_id} has no call hierarchy support; incoming calls were derived from find-references: "
    "they may include non-call usages and may miss callers inside properties, constructors or one-line functions."
)
# Frozen outgoing-unavailable text per spec 2.3.
OUTGOING_UNAVAILABLE_TEXT = "outgoing calls cannot be derived from find-references"


def _make_approximate_retriever(ls_id: str, n_children: int = 2, with_call_sites: bool = False) -> MagicMock:
    """Retriever mock in the frozen spec 5d shape: request_call_hierarchy_by_location returns an
    approximate incoming result; get_language_server(relative_path).ls_id.value == ls_id.

    Children live in DISTINCT files (so the per-file-counts form differs meaningfully from the one-line
    summary) and optionally carry call_sites (so the without-call-sites form is shorter than the full form).
    """

    def make_child(i: int) -> dict[str, object]:
        rel_path = f"pkg/module_{i}/callers_{i}.py"
        child: dict[str, object] = {
            "name": f"caller_{i}",
            "kind": 12,
            "location": {
                "uri": f"file:///repo/{rel_path}",
                "range": {"start": {"line": i + 1, "character": 0}, "end": {"line": i + 1, "character": 10}},
                "absolutePath": f"/repo/{rel_path}",
                "relativePath": rel_path,
            },
            "children": [],
        }
        if with_call_sites:
            child["call_sites"] = {
                "relative_path": rel_path,
                "ranges": [
                    {"start": {"line": i + 2, "character": 4}, "end": {"line": i + 2, "character": 4}},
                    {"start": {"line": i + 3, "character": 8}, "end": {"line": i + 3, "character": 8}},
                ],
            }
        return child

    incoming_result = {
        "roots": [
            {
                "name": "leaf",
                "kind": 12,  # Function
                "location": {
                    "uri": "file:///repo/test.py",
                    "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 4}},
                    "absolutePath": "/repo/test.py",
                    "relativePath": "test.py",
                },
                "children": [make_child(i) for i in range(n_children)],
            }
        ],
        "truncated": False,
        "external_calls_omitted": 0,
        "approximate": True,  # Frozen: approximate is True
    }

    retriever = MagicMock()
    symbol = MagicMock()
    symbol.get_name_path.return_value = "leaf"
    retriever.find_unique.return_value = symbol
    retriever.request_call_hierarchy_by_location.return_value = incoming_result
    retriever.get_language_server.return_value.ls_id.value = ls_id
    return retriever


class TestCallHierarchyToolApproximate:
    """F10, F14: Tests for tool-level approximate output."""

    def test_f10_direction_both_with_approximate_mock(self) -> None:
        """F10: direction="both" with approximate mock returns exactly ONE request_call_hierarchy_by_location call
        and pins the FROZEN warning strings.
        """
        retriever = _make_approximate_retriever(ls_id="csharp", n_children=1)

        tool = CallHierarchyTool(MagicMock())
        with patch.object(CallHierarchyTool, "create_language_server_symbol_retriever", return_value=retriever):
            result = tool.apply(name_path="leaf", relative_path=CALL_GRAPH_FILE, direction="both", depth=1, max_answer_chars=10_000_000)

        output = json.loads(result)

        # Verify exactly ONE request_call_hierarchy_by_location call (incoming only, not outgoing)
        calls = retriever.request_call_hierarchy_by_location.call_args_list
        assert len(calls) == 1, "Should have exactly one call for incoming direction"
        assert calls[0].kwargs["direction"] == "incoming"

        # Verify output structure with the FROZEN strings pinned exactly
        assert output.get("approximate") is True, "Output should have approximate: true"
        assert output.get("approximate_note") == APPROXIMATE_NOTE_TEMPLATE.format(ls_id="csharp")
        assert output.get("outgoing") == [], "Outgoing should be empty list"
        assert output.get("outgoing_unavailable") == OUTGOING_UNAVAILABLE_TEXT
        assert output.get("truncated") is False

    def test_f14_approximate_output_at_length_limits(self) -> None:
        """F14: approximate output at TWO pinned limits, both computed at test time from actual output lengths.

        Limit (a) = len(full) - 1 forces the first shortened form that fits; limit (b) = len(form a) - 1
        forces a strictly shorter form (form 2/3). Every shortened form must carry the approximate marker,
        the frozen note text and the _limit_length too-long prefix. The generic-notice case
        (limit below the shortest form) is explicitly out of scope per spec F14.
        """
        # call_sites make the without-call-sites form strictly shorter than the full form;
        # distinct per-child files make the per-file-counts form strictly longer than the one-line summary
        retriever = _make_approximate_retriever(ls_id="python", n_children=8, with_call_sites=True)
        frozen_note = APPROXIMATE_NOTE_TEMPLATE.format(ls_id="python")

        tool = CallHierarchyTool(MagicMock())

        def apply_with_limit(limit: int) -> str:
            with patch.object(CallHierarchyTool, "create_language_server_symbol_retriever", return_value=retriever):
                return tool.apply(name_path="leaf", relative_path=CALL_GRAPH_FILE, direction="incoming", depth=1, max_answer_chars=limit)

        # baseline: the full JSON form (fits comfortably)
        result_full = apply_with_limit(10_000_000)
        output_full = json.loads(result_full)
        assert output_full["approximate"] is True
        assert output_full["approximate_note"] == frozen_note
        full_len = len(result_full)

        # limit (a): one char below the full output -> the first shortened form that fits is returned
        result_a = apply_with_limit(full_len - 1)
        assert len(result_a) <= full_len - 1
        assert result_a != result_full
        assert "The answer is too long" in result_a, "shortened forms must carry the _limit_length too-long prefix"
        assert "approximate" in result_a
        assert frozen_note in result_a, "the full frozen note must survive shortening"

        # limit (b): one char below form (a) -> a strictly shorter form (form 2/3) is returned
        limit_b = len(result_a) - 1
        result_b = apply_with_limit(limit_b)
        assert len(result_b) <= limit_b
        assert result_b != result_a
        assert "The answer is too long" in result_b, "shortened forms must carry the _limit_length too-long prefix"
        assert "approximate" in result_b
        assert frozen_note in result_b, "the full frozen note must survive shortening"
