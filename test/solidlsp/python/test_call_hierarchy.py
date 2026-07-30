"""
LS-level integration tests for request_call_hierarchy (spec 7b) against Python backends.

Runtime verification (2026-07-30, devcontainer):
- pyright 1.1.403 and basedpyright 1.39.9 answer prepareCallHierarchy (both advertise callHierarchyProvider).
- pyrefly 1.1.1 also answers prepareCallHierarchy/incomingCalls/outgoingCalls, so it joins the main parametrization.
- ty 0.0.25 advertises no callHierarchyProvider and responds to prepareCallHierarchy with -32601
  ("Received request textDocument/prepareCallHierarchy which does not have a handler") -> row L8.
"""

import os
from typing import Any, cast

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import LanguageServerId
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.ls_types import CallHierarchyNode

pytestmark = pytest.mark.python

CALL_GRAPH_FILE = os.path.join("test_repo", "call_graph.py")
VARIABLES_FILE = os.path.join("test_repo", "variables.py")

# 0-based (line, column) of the function name identifiers in call_graph.py
LEAF_POS = (3, 4)
MID_POS = (8, 4)
ENTRY_A_POS = (13, 4)
RECURSIVE_FN_POS = (23, 4)
# 0-based (line, column) of the module-level variable `module_var` in variables.py
MODULE_VAR_POS = (13, 0)

# Pinned explicitly per spec 7b (NOT the shared PYTHON_BACKEND_LANGUAGES list); pyrefly added after runtime check.
CALL_HIERARCHY_BACKENDS = [
    LanguageServerId.PYTHON,
    LanguageServerId.PYTHON_BASEDPYRIGHT,
    LanguageServerId.PYTHON_PYREFLY,
]
# Backends verified at runtime to NOT answer prepareCallHierarchy.
UNSUPPORTED_BACKENDS = [
    LanguageServerId.PYTHON_TY,
]


def _has_recursion_node(nodes: list[CallHierarchyNode]) -> bool:
    return any(node.get("recursion") or _has_recursion_node(node["children"]) for node in nodes)


class TestCallHierarchyPython:
    @pytest.mark.parametrize("language_server", CALL_HIERARCHY_BACKENDS, indirect=True)
    def test_l1_incoming_leaf_depth_1(self, language_server: SolidLanguageServer) -> None:
        """L1: incoming of leaf at depth=1 is exactly mid, with correct paths, selectionRange-derived line and call sites."""
        result = language_server.request_call_hierarchy(CALL_GRAPH_FILE, *LEAF_POS, direction="incoming", depth=1, max_nodes=200)

        assert len(result["roots"]) == 1
        root = result["roots"][0]
        assert root["name"] == "leaf"

        children = root["children"]
        assert [child["name"] for child in children] == ["mid"]
        mid = children[0]
        assert mid["location"]["relativePath"] == CALL_GRAPH_FILE
        # node line must be mid's def line, derived from selectionRange (not range, which could point at decorators)
        assert mid["location"]["range"]["start"]["line"] == MID_POS[0]
        assert "call_sites" in mid
        assert len(mid["call_sites"]["ranges"]) >= 1

    @pytest.mark.parametrize("language_server", CALL_HIERARCHY_BACKENDS, indirect=True)
    def test_l2_outgoing_mid_depth_1(self, language_server: SolidLanguageServer) -> None:
        """L2: outgoing of mid at depth=1 is exactly leaf; call_sites.relative_path is the caller's (mid's) file."""
        result = language_server.request_call_hierarchy(CALL_GRAPH_FILE, *MID_POS, direction="outgoing", depth=1, max_nodes=200)

        assert len(result["roots"]) == 1
        root = result["roots"][0]
        assert root["name"] == "mid"

        children = root["children"]
        assert [child["name"] for child in children] == ["leaf"]
        leaf = children[0]
        assert "call_sites" in leaf
        assert leaf["call_sites"]["relative_path"] == CALL_GRAPH_FILE

    @pytest.mark.parametrize("language_server", CALL_HIERARCHY_BACKENDS, indirect=True)
    def test_l3_incoming_leaf_depth_2(self, language_server: SolidLanguageServer) -> None:
        """L3: incoming of leaf at depth=2 has mid at level 1 and both entry_a and entry_b at level 2."""
        result = language_server.request_call_hierarchy(CALL_GRAPH_FILE, *LEAF_POS, direction="incoming", depth=2, max_nodes=200)

        assert len(result["roots"]) == 1
        level_1 = result["roots"][0]["children"]
        assert [node["name"] for node in level_1] == ["mid"]
        level_2_names = {node["name"] for node in level_1[0]["children"]}
        assert level_2_names == {"entry_a", "entry_b"}

    @pytest.mark.parametrize("language_server", CALL_HIERARCHY_BACKENDS, indirect=True)
    def test_l4_incoming_recursive_fn_depth_3(self, language_server: SolidLanguageServer) -> None:
        """L4: incoming of recursive_fn at depth=3 terminates and flags recursion on some node."""
        result = language_server.request_call_hierarchy(CALL_GRAPH_FILE, *RECURSIVE_FN_POS, direction="incoming", depth=3, max_nodes=200)

        assert len(result["roots"]) == 1
        assert result["roots"][0]["name"] == "recursive_fn"
        assert _has_recursion_node(result["roots"])

    @pytest.mark.parametrize("language_server", CALL_HIERARCHY_BACKENDS, indirect=True)
    def test_l5_incoming_entry_a_depth_1(self, language_server: SolidLanguageServer) -> None:
        """L5: incoming of entry_a (never called) yields the root with no children and no truncation."""
        result = language_server.request_call_hierarchy(CALL_GRAPH_FILE, *ENTRY_A_POS, direction="incoming", depth=1, max_nodes=200)

        assert len(result["roots"]) == 1
        assert result["roots"][0]["name"] == "entry_a"
        assert result["roots"][0]["children"] == []
        assert result["truncated"] is False

    @pytest.mark.parametrize("language_server", CALL_HIERARCHY_BACKENDS, indirect=True)
    def test_l6_prepare_on_module_level_variable(self, language_server: SolidLanguageServer) -> None:
        """L6: prepare on a module-level variable yields empty roots without raising."""
        result = language_server.request_call_hierarchy(VARIABLES_FILE, *MODULE_VAR_POS, direction="incoming", depth=1, max_nodes=200)

        assert result["roots"] == []
        assert result["truncated"] is False
        assert result["external_calls_omitted"] == 0

    @pytest.mark.parametrize("language_server", [LanguageServerId.PYTHON, LanguageServerId.PYTHON_BASEDPYRIGHT], indirect=True)
    def test_l7_initialize_params_advertise_call_hierarchy(self, language_server: SolidLanguageServer) -> None:
        """L7: pyright/basedpyright initialize params contain callHierarchy with dynamicRegistration False."""
        params = cast(dict[str, Any], language_server._create_initialize_params())
        assert params["capabilities"]["textDocument"]["callHierarchy"] == {"dynamicRegistration": False}

    @pytest.mark.parametrize("language_server", UNSUPPORTED_BACKENDS, indirect=True)
    def test_l8_unsupported_backend_raises(self, language_server: SolidLanguageServer) -> None:
        """L8: backends without call hierarchy support raise a descriptive SolidLSPException."""
        with pytest.raises(SolidLSPException, match="does not support call hierarchy"):
            language_server.request_call_hierarchy(CALL_GRAPH_FILE, *LEAF_POS, direction="incoming", depth=1, max_nodes=200)
