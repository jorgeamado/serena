"""
Unit tests for call hierarchy expansion mechanics (_expand_call_hierarchy, _convert_call_hierarchy_item,
the -32601 error mapping and the tool-level "both" budget sharing).
"""

import json
import os
import tempfile
from pathlib import Path
from time import monotonic
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from solidlsp import ls_types
from solidlsp.ls import (
    CALL_HIERARCHY_DEADLINE_SECONDS,
    CALL_HIERARCHY_MAX_NODES,
    SolidLanguageServer,
    SolidLSPException,
)
from solidlsp.lsp_protocol_handler.server import LSPError


@pytest.fixture
def language_server_mock() -> MagicMock:
    """Create a mock language server."""
    ls = MagicMock(spec=SolidLanguageServer)
    ls.repository_root_path = "/repo"
    ls.is_ignored_path = MagicMock(return_value=False)
    # Mock ls_id for error messages
    ls_id_mock = MagicMock()
    ls_id_mock.value = "test_server"
    ls.ls_id = ls_id_mock
    return ls


class TestCallHierarchyExpansion:
    """Tests for _expand_call_hierarchy graph expansion."""

    def test_u1_data_round_trip(self, language_server_mock: MagicMock) -> None:
        """U1: opaque data survives round-trip through fetch_calls."""
        prepare_items = [
            {
                "name": "root",
                "kind": 1,
                "selectionRange": {"start": {"line": 1, "character": 0}},
                "uri": "file:///repo/a.py",
                "data": {"custom": "value"},
            }
        ]

        # fetch_calls should receive the exact same item dict
        received_items = []

        def fetch_calls(item: dict[str, object]) -> list[dict[str, object]]:
            received_items.append(item)
            return []

        def convert_item(item: dict[str, object]) -> tuple[tuple[str, int, int], ls_types.CallHierarchyNode | None, bool]:
            node: ls_types.CallHierarchyNode = {
                "name": cast(str, item.get("name", "")),
                "kind": cast(int, item.get("kind", 1)),
                "location": ls_types.Location(
                    uri="file:///repo/a.py",
                    range={"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 5}},
                    absolutePath="/repo/a.py",
                    relativePath="a.py",
                ),
                "children": [],
            }
            return (("file:///repo/a.py", 1, 0), node, False)

        # Call _expand_call_hierarchy via the actual method
        result = SolidLanguageServer._expand_call_hierarchy(
            language_server_mock,
            prepare_items=prepare_items,
            fetch_calls=fetch_calls,
            convert_item=convert_item,
            direction="incoming",
            depth=1,
            max_nodes=CALL_HIERARCHY_MAX_NODES,
            deadline=monotonic() + CALL_HIERARCHY_DEADLINE_SECONDS,
        )

        # Verify that fetch_calls received the same object
        assert len(received_items) == 1
        assert received_items[0] is prepare_items[0]
        assert received_items[0].get("data") == {"custom": "value"}
        # Verify result is returned
        assert result["roots"]

    def test_u2_diamond_no_recursion_flag(self, language_server_mock: MagicMock) -> None:
        """U2: diamond root→a/b→shared expands both branches, no recursion flag."""
        prepare_items = [{"name": "root", "kind": 1, "selectionRange": {"start": {"line": 0, "character": 0}}, "uri": "file:///repo/a.py"}]

        def fetch_calls(item: dict[str, object]) -> list[dict[str, object]]:
            name = item.get("name")
            if name == "root":
                return [
                    {
                        "from": {
                            "name": "a",
                            "kind": 1,
                            "uri": "file:///repo/a.py",
                            "selectionRange": {"start": {"line": 1, "character": 0}},
                        },
                        "fromRanges": [],
                    },
                    {
                        "from": {
                            "name": "b",
                            "kind": 1,
                            "uri": "file:///repo/a.py",
                            "selectionRange": {"start": {"line": 2, "character": 0}},
                        },
                        "fromRanges": [],
                    },
                ]
            elif name in ("a", "b"):
                return [
                    {
                        "from": {
                            "name": "shared",
                            "kind": 1,
                            "uri": "file:///repo/a.py",
                            "selectionRange": {"start": {"line": 3, "character": 0}},
                        },
                        "fromRanges": [],
                    }
                ]
            return []

        def convert_item(item: dict[str, object]) -> tuple[tuple[str, int, int], ls_types.CallHierarchyNode | None, bool]:
            name = cast(str, item.get("name", ""))
            line = cast(dict[str, int], cast(dict[str, object], item.get("selectionRange", {})).get("start", {})).get("line", 0)
            node: ls_types.CallHierarchyNode = {
                "name": name,
                "kind": 1,
                "location": ls_types.Location(
                    uri="file:///repo/a.py",
                    range={"start": {"line": line, "character": 0}, "end": {"line": line, "character": 5}},
                    absolutePath="/repo/a.py",
                    relativePath="a.py",
                ),
                "children": [],
            }
            return (("file:///repo/a.py", line, 0), node, False)

        result = SolidLanguageServer._expand_call_hierarchy(
            language_server_mock,
            prepare_items=prepare_items,
            fetch_calls=fetch_calls,
            convert_item=convert_item,
            direction="incoming",
            depth=2,
            max_nodes=CALL_HIERARCHY_MAX_NODES,
            deadline=monotonic() + CALL_HIERARCHY_DEADLINE_SECONDS,
        )

        # Check structure: root has 2 children (a, b), each has 1 child (shared)
        assert len(result["roots"]) == 1
        root = result["roots"][0]
        assert root["name"] == "root"
        assert len(root["children"]) == 2

        # Both a and b should have shared as child, not marked as recursion
        for child in root["children"]:
            assert len(child["children"]) == 1
            shared = child["children"][0]
            assert shared["name"] == "shared"
            assert "recursion" not in shared

    def test_u3_self_cycle_and_2cycle(self, language_server_mock: MagicMock) -> None:
        """U3: self-cycle and 2-cycle are detected via ancestor_keys."""
        prepare_items = [{"name": "root", "kind": 1, "selectionRange": {"start": {"line": 0, "character": 0}}, "uri": "file:///repo/a.py"}]

        def fetch_calls(item: dict[str, object]) -> list[dict[str, object]]:
            name = item.get("name")
            if name == "root":  # noqa: SIM116
                return [
                    {
                        "from": {
                            "name": "self_caller",
                            "kind": 1,
                            "uri": "file:///repo/a.py",
                            "selectionRange": {"start": {"line": 1, "character": 0}},
                        },
                        "fromRanges": [],
                    }
                ]
            elif name == "self_caller":
                return [
                    {
                        "from": {
                            "name": "self_caller",
                            "kind": 1,
                            "uri": "file:///repo/a.py",
                            "selectionRange": {"start": {"line": 1, "character": 0}},
                        },
                        "fromRanges": [],
                    },
                    {
                        "from": {
                            "name": "other",
                            "kind": 1,
                            "uri": "file:///repo/a.py",
                            "selectionRange": {"start": {"line": 2, "character": 0}},
                        },
                        "fromRanges": [],
                    },
                ]
            elif name == "other":
                return [
                    {
                        "from": {
                            "name": "self_caller",
                            "kind": 1,
                            "uri": "file:///repo/a.py",
                            "selectionRange": {"start": {"line": 1, "character": 0}},
                        },
                        "fromRanges": [],
                    }
                ]
            return []

        def convert_item(item: dict[str, object]) -> tuple[tuple[str, int, int], ls_types.CallHierarchyNode | None, bool]:
            name = cast(str, item.get("name", ""))
            line = cast(dict[str, int], cast(dict[str, object], item.get("selectionRange", {})).get("start", {})).get("line", 0)
            node: ls_types.CallHierarchyNode = {
                "name": name,
                "kind": 1,
                "location": ls_types.Location(
                    uri="file:///repo/a.py",
                    range={"start": {"line": line, "character": 0}, "end": {"line": line, "character": 5}},
                    absolutePath="/repo/a.py",
                    relativePath="a.py",
                ),
                "children": [],
            }
            return (("file:///repo/a.py", line, 0), node, False)

        result = SolidLanguageServer._expand_call_hierarchy(
            language_server_mock,
            prepare_items=prepare_items,
            fetch_calls=fetch_calls,
            convert_item=convert_item,
            direction="incoming",
            depth=3,
            max_nodes=CALL_HIERARCHY_MAX_NODES,
            deadline=monotonic() + CALL_HIERARCHY_DEADLINE_SECONDS,
        )

        # Root -> self_caller -> [self_caller (recursion stub), other] -> [self_caller (recursion stub)]
        assert len(result["roots"]) == 1
        root = result["roots"][0]
        assert root["name"] == "root"
        assert len(root["children"]) == 1

        self_caller_1 = root["children"][0]
        assert self_caller_1["name"] == "self_caller"
        assert len(self_caller_1["children"]) == 2

        # Check for recursion stubs
        recursion_count = sum(1 for child in self_caller_1["children"] if child.get("recursion"))
        assert recursion_count >= 1  # At least the self-cycle is detected

    def test_u4_multiple_prepare_roots(self, language_server_mock: MagicMock) -> None:
        """U4: multiple prepare roots each become a root node; one shared budget spans all of them."""
        prepare_items = [
            {"name": "root1", "kind": 1, "selectionRange": {"start": {"line": 0, "character": 0}}, "uri": "file:///repo/a.py"},
            {"name": "root2", "kind": 1, "selectionRange": {"start": {"line": 1, "character": 0}}, "uri": "file:///repo/a.py"},
            {"name": "root3", "kind": 1, "selectionRange": {"start": {"line": 2, "character": 0}}, "uri": "file:///repo/a.py"},
        ]

        def fetch_calls(item: dict[str, object]) -> list[dict[str, object]]:
            # each root has 3 callers; full expansion would emit 3 roots + 9 children = 12 nodes
            parent_line = cast(dict[str, int], cast(dict[str, object], item.get("selectionRange", {})).get("start", {})).get("line", 0)
            return [
                {
                    "from": {
                        "name": f"caller_{parent_line}_{i}",
                        "kind": 1,
                        "uri": "file:///repo/a.py",
                        "selectionRange": {"start": {"line": 10 + parent_line * 10 + i, "character": 0}},
                    },
                    "fromRanges": [],
                }
                for i in range(3)
            ]

        def convert_item(item: dict[str, object]) -> tuple[tuple[str, int, int], ls_types.CallHierarchyNode | None, bool]:
            name = cast(str, item.get("name", ""))
            line = cast(dict[str, int], cast(dict[str, object], item.get("selectionRange", {})).get("start", {})).get("line", 0)
            node: ls_types.CallHierarchyNode = {
                "name": name,
                "kind": 1,
                "location": ls_types.Location(
                    uri="file:///repo/a.py",
                    range={"start": {"line": line, "character": 0}, "end": {"line": line, "character": 5}},
                    absolutePath="/repo/a.py",
                    relativePath="a.py",
                ),
                "children": [],
            }
            return (("file:///repo/a.py", line, 0), node, False)

        max_nodes = 5  # smaller than the 12 nodes a full expansion would emit
        result = SolidLanguageServer._expand_call_hierarchy(
            language_server_mock,
            prepare_items=prepare_items,
            fetch_calls=fetch_calls,
            convert_item=convert_item,
            direction="incoming",
            depth=1,
            max_nodes=max_nodes,
            deadline=monotonic() + CALL_HIERARCHY_DEADLINE_SECONDS,
        )

        # every prepare item becomes a root node
        assert [root["name"] for root in result["roots"]] == ["root1", "root2", "root3"]

        # the budget is shared across the roots: exactly max_nodes emitted in total, then truncation
        total_nodes = len(result["roots"]) + sum(len(root["children"]) for root in result["roots"])
        assert total_nodes == max_nodes
        assert result["truncated"] is True

    def test_u5_hard_budget(self, language_server_mock: MagicMock) -> None:
        """U5: response with more children than remaining budget is truncated exactly at max_nodes."""
        prepare_items = [{"name": "root", "kind": 1, "selectionRange": {"start": {"line": 0, "character": 0}}, "uri": "file:///repo/a.py"}]

        def fetch_calls(item: dict[str, object]) -> list[dict[str, object]]:
            if item.get("name") == "root":
                # Return 10 children
                return [
                    {
                        "from": {
                            "name": f"child{i}",
                            "kind": 1,
                            "uri": "file:///repo/a.py",
                            "selectionRange": {"start": {"line": i + 1, "character": 0}},
                        },
                        "fromRanges": [],
                    }
                    for i in range(10)
                ]
            return []

        def convert_item(item: dict[str, object]) -> tuple[tuple[str, int, int], ls_types.CallHierarchyNode | None, bool]:
            name = cast(str, item.get("name", ""))
            line = cast(dict[str, int], cast(dict[str, object], item.get("selectionRange", {})).get("start", {})).get("line", 0)
            node: ls_types.CallHierarchyNode = {
                "name": name,
                "kind": 1,
                "location": ls_types.Location(
                    uri="file:///repo/a.py",
                    range={"start": {"line": line, "character": 0}, "end": {"line": line, "character": 5}},
                    absolutePath="/repo/a.py",
                    relativePath="a.py",
                ),
                "children": [],
            }
            return (("file:///repo/a.py", line, 0), node, False)

        max_nodes = 5  # root (1) + 4 children
        result = SolidLanguageServer._expand_call_hierarchy(
            language_server_mock,
            prepare_items=prepare_items,
            fetch_calls=fetch_calls,
            convert_item=convert_item,
            direction="incoming",
            depth=1,
            max_nodes=max_nodes,
            deadline=monotonic() + CALL_HIERARCHY_DEADLINE_SECONDS,
        )

        total_nodes = len(result["roots"]) + sum(len(r["children"]) for r in result["roots"])
        assert total_nodes == max_nodes
        assert result["truncated"] is True

    def test_u6_ignored_path(self, language_server_mock: MagicMock) -> None:
        """U6: item in ignored path is dropped silently (no omitted counter)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temp file matching an ignore pattern (e.g., __pycache__)
            ignore_dir = os.path.join(tmpdir, "__pycache__")
            os.makedirs(ignore_dir, exist_ok=True)
            ignored_file = os.path.join(ignore_dir, "test.py")
            Path(ignored_file).touch()

            language_server_mock.repository_root_path = tmpdir
            language_server_mock.is_ignored_path = MagicMock(return_value=True)

            item = {
                "name": "ignored_func",
                "kind": 1,
                "uri": f"file://{ignored_file}",
                "selectionRange": {"start": {"line": 0, "character": 0}},
            }

            key, node, omitted = SolidLanguageServer._convert_call_hierarchy_item(language_server_mock, item)

            # Should be dropped but NOT counted as omitted
            assert node is None
            assert omitted is False

    def test_u7_outside_root_and_nonexistent(self, language_server_mock: MagicMock) -> None:
        """U7: items outside the repo root, and nonexistent paths inside it, are dropped with omitted=True."""
        with tempfile.TemporaryDirectory() as repo_dir, tempfile.TemporaryDirectory() as outside_dir:
            language_server_mock.repository_root_path = repo_dir

            # case 1: a real file that exists on disk but lies OUTSIDE the repo root (containment check, not existence)
            outside_file = os.path.join(outside_dir, "outside.py")
            Path(outside_file).touch()
            item_outside = {
                "name": "outside_func",
                "kind": 1,
                "uri": f"file://{outside_file}",
                "selectionRange": {"start": {"line": 0, "character": 0}},
            }
            _key, node, omitted = SolidLanguageServer._convert_call_hierarchy_item(language_server_mock, item_outside)
            assert node is None
            assert omitted is True

            # case 2: a path INSIDE the repo root that does not exist on disk (deleted/generated file)
            missing_file = os.path.join(repo_dir, "deleted.py")
            assert not os.path.exists(missing_file)
            item_missing = {
                "name": "deleted_func",
                "kind": 1,
                "uri": f"file://{missing_file}",
                "selectionRange": {"start": {"line": 0, "character": 0}},
            }
            _key, node, omitted = SolidLanguageServer._convert_call_hierarchy_item(language_server_mock, item_missing)
            assert node is None
            assert omitted is True

    def test_u8_outgoing_call_sites_relative_path(self, language_server_mock: MagicMock) -> None:
        """U8: outgoing call_sites.relative_path equals PARENT's file, not child's."""
        prepare_items = [
            {"name": "parent", "kind": 1, "selectionRange": {"start": {"line": 0, "character": 0}}, "uri": "file:///repo/parent.py"}
        ]

        def fetch_calls(item: dict[str, object]) -> list[dict[str, object]]:
            if item.get("name") == "parent":
                return [
                    {
                        "to": {
                            "name": "child",
                            "kind": 1,
                            "uri": "file:///repo/child.py",
                            "selectionRange": {"start": {"line": 1, "character": 0}},
                        },
                        "fromRanges": [{"start": {"line": 0, "character": 5}, "end": {"line": 0, "character": 10}}],
                    }
                ]
            return []

        def convert_item(item: dict[str, object]) -> tuple[tuple[str, int, int], ls_types.CallHierarchyNode | None, bool]:
            name = cast(str, item.get("name", ""))
            uri = cast(str, item.get("uri", ""))
            line = cast(dict[str, int], cast(dict[str, object], item.get("selectionRange", {})).get("start", {})).get("line", 0)
            rel_path = "parent.py" if "parent" in uri else "child.py"
            node: ls_types.CallHierarchyNode = {
                "name": name,
                "kind": 1,
                "location": ls_types.Location(
                    uri=uri,
                    range={"start": {"line": line, "character": 0}, "end": {"line": line, "character": 5}},
                    absolutePath=uri.replace("file://", ""),
                    relativePath=rel_path,
                ),
                "children": [],
            }
            return ((uri, line, 0), node, False)

        result = SolidLanguageServer._expand_call_hierarchy(
            language_server_mock,
            prepare_items=prepare_items,
            fetch_calls=fetch_calls,
            convert_item=convert_item,
            direction="outgoing",
            depth=1,
            max_nodes=CALL_HIERARCHY_MAX_NODES,
            deadline=monotonic() + CALL_HIERARCHY_DEADLINE_SECONDS,
        )

        assert len(result["roots"]) == 1
        parent = result["roots"][0]
        assert len(parent["children"]) == 1
        child = parent["children"][0]
        assert "call_sites" in child
        # For outgoing, call_sites.relative_path should be parent's file
        assert child["call_sites"]["relative_path"] == "parent.py"

    def test_u9_deadline_pre_expired(self, language_server_mock: MagicMock) -> None:
        """U9: deadline pre-expired stops after roots, truncated=True."""
        prepare_items = [{"name": "root", "kind": 1, "selectionRange": {"start": {"line": 0, "character": 0}}, "uri": "file:///repo/a.py"}]

        fetch_call_count = 0

        def fetch_calls(item: dict[str, object]) -> list[dict[str, object]]:
            nonlocal fetch_call_count
            fetch_call_count += 1
            return [
                {
                    "from": {
                        "name": "child",
                        "kind": 1,
                        "uri": "file:///repo/a.py",
                        "selectionRange": {"start": {"line": 1, "character": 0}},
                    },
                    "fromRanges": [],
                }
            ]

        def convert_item(item: dict[str, object]) -> tuple[tuple[str, int, int], ls_types.CallHierarchyNode | None, bool]:
            name = cast(str, item.get("name", ""))
            line = cast(dict[str, int], cast(dict[str, object], item.get("selectionRange", {})).get("start", {})).get("line", 0)
            node: ls_types.CallHierarchyNode = {
                "name": name,
                "kind": 1,
                "location": ls_types.Location(
                    uri="file:///repo/a.py",
                    range={"start": {"line": line, "character": 0}, "end": {"line": line, "character": 5}},
                    absolutePath="/repo/a.py",
                    relativePath="a.py",
                ),
                "children": [],
            }
            return (("file:///repo/a.py", line, 0), node, False)

        # Use a deadline in the past
        past_deadline = monotonic() - 1

        result = SolidLanguageServer._expand_call_hierarchy(
            language_server_mock,
            prepare_items=prepare_items,
            fetch_calls=fetch_calls,
            convert_item=convert_item,
            direction="incoming",
            depth=1,
            max_nodes=CALL_HIERARCHY_MAX_NODES,
            deadline=past_deadline,
        )

        # Should have root but no children; no expansion request may be issued after the deadline
        assert len(result["roots"]) == 1
        assert len(result["roots"][0]["children"]) == 0
        assert result["truncated"] is True
        assert fetch_call_count == 0

    @pytest.mark.parametrize("method_name", ["callHierarchy/incomingCalls", "textDocument/prepareCallHierarchy"])
    @pytest.mark.parametrize("wrapped", [True, False], ids=["solidlsp_wrapped_cause", "bare_lsp_error"])
    def test_u10_error_mapping_helper(self, language_server_mock: MagicMock, method_name: str, wrapped: bool) -> None:
        """U10: -32601 maps to a message naming the ACTUAL failed method.

        The real transport (ls_process.send_request) raises SolidLSPException(message, cause=LSPError(...));
        a bare LSPError must also still map.
        """
        lsp_error = LSPError(-32601, "method not found")
        error: Exception = SolidLSPException(f"Error processing request {method_name}", cause=lsp_error) if wrapped else lsp_error

        mapped = SolidLanguageServer._map_call_hierarchy_exception(language_server_mock, error, method_name)

        assert isinstance(mapped, SolidLSPException)
        assert "does not support call hierarchy" in str(mapped)
        assert method_name in str(mapped)
        # the message must name the actual failed method, not any of the other call hierarchy methods
        other_methods = {"callHierarchy/incomingCalls", "callHierarchy/outgoingCalls", "textDocument/prepareCallHierarchy"} - {method_name}
        assert all(other not in str(mapped) for other in other_methods)

    def test_u10_error_mapping_helper_ignores_other_codes(self, language_server_mock: MagicMock) -> None:
        """U10 complement: a non--32601 error is not mapped (returns None so the original propagates)."""
        error = SolidLSPException("Error processing request", cause=LSPError(-32603, "internal error"))
        mapped = SolidLanguageServer._map_call_hierarchy_exception(language_server_mock, error, "callHierarchy/incomingCalls")
        assert mapped is None

    def test_u11_direction_both_budget_exhausted(self) -> None:
        """U11: direction='both' with the budget exhausted by incoming skips the outgoing request entirely.

        Budget sharing lives at the TOOL level: the tool passes 200 to the first (incoming) mid-layer call and
        200 - <nodes emitted> to the second; at 0 remaining, the outgoing call is NOT made, the outgoing key is []
        and the combined result is truncated.
        """
        from serena.tools import CallHierarchyTool

        def make_node(name: str, line: int) -> ls_types.CallHierarchyNode:
            return {
                "name": name,
                "kind": cast(ls_types.SymbolKind, 12),  # Function
                "location": ls_types.Location(
                    uri="file:///repo/a.py",
                    range={"start": {"line": line, "character": 0}, "end": {"line": line, "character": 5}},
                    absolutePath="/repo/a.py",
                    relativePath="a.py",
                ),
                "children": [],
            }

        # incoming consumes the entire 200-node budget: one root with 199 children, truncated
        incoming_root = make_node("leaf", 0)
        incoming_root["children"] = [make_node(f"caller{i}", i + 1) for i in range(199)]
        incoming_result = ls_types.CallHierarchyResult(roots=[incoming_root], truncated=True, external_calls_omitted=0)

        retriever = MagicMock()
        symbol = MagicMock()
        symbol.get_name_path.return_value = "leaf"
        retriever.find_unique.return_value = symbol
        retriever.request_call_hierarchy_by_location.return_value = incoming_result

        tool = CallHierarchyTool(MagicMock())
        with patch.object(CallHierarchyTool, "create_language_server_symbol_retriever", return_value=retriever):
            result = tool.apply(name_path="leaf", relative_path="a.py", direction="both", depth=1, max_answer_chars=10_000_000)

        output = json.loads(result)
        assert output["outgoing"] == []
        assert output["truncated"] is True

        # exactly one mid-layer call was made, for the incoming direction, with the full budget;
        # the outgoing request was never made
        calls = retriever.request_call_hierarchy_by_location.call_args_list
        assert len(calls) == 1
        assert calls[0].kwargs["direction"] == "incoming"
        assert calls[0].kwargs["max_nodes"] == 200
