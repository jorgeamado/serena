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
    ReferenceInSymbol,
    SolidLanguageServer,
    SolidLSPException,
)
from solidlsp.ls_utils import PathUtils
from solidlsp.lsp_protocol_handler.lsp_types import SymbolKind
from solidlsp.lsp_protocol_handler.server import LSPError


def _stub_convert_item(item: dict[str, object]) -> tuple[tuple[str, int, int], ls_types.CallHierarchyNode | None, bool]:
    """Engine-shape converter keyed on (uri, selectionRange.start); no filesystem containment checks.

    Used in place of the real _convert_call_hierarchy_item so unit tests can use fake /repo paths
    that do not exist on disk (the real converter drops nonexistent paths with omitted=True).
    """
    name = cast(str, item.get("name", ""))
    uri = cast(str, item.get("uri", ""))
    start = cast(dict[str, int], cast(dict[str, object], item.get("selectionRange", {})).get("start", {}))
    line = start.get("line", 0)
    char = start.get("character", 0)
    abs_path = uri.removeprefix("file://")
    node: ls_types.CallHierarchyNode = {
        "name": name,
        "kind": cast(ls_types.SymbolKind, item.get("kind", 1)),
        "location": ls_types.Location(
            uri=uri,
            range={"start": {"line": line, "character": char}, "end": {"line": line, "character": char + max(len(name), 1)}},
            absolutePath=abs_path,
            relativePath=abs_path.removeprefix("/repo/"),
        ),
        "children": [],
    }
    return ((uri, line, char), node, False)


def _make_symbol(
    name: str, rel_path: str = "a.py", line: int = 0, kind: SymbolKind = SymbolKind.Function
) -> ls_types.UnifiedSymbolInformation:
    """Build a UnifiedSymbolInformation with location + selectionRange under the fake /repo root."""
    return {
        "children": [],
        "name": name,
        "kind": kind,
        "location": ls_types.Location(
            uri=f"file:///repo/{rel_path}",
            range={"start": {"line": line, "character": 0}, "end": {"line": line + 2, "character": 0}},
            absolutePath=f"/repo/{rel_path}",
            relativePath=rel_path,
        ),
        "selectionRange": {"start": {"line": line, "character": 4}, "end": {"line": line, "character": 4 + len(name)}},
    }


def _wire_fallback_engine(ls: MagicMock) -> None:
    """Wire the REAL production methods onto the mock (as MagicMock spies with real side effects)
    so that _call_hierarchy_fallback_incoming / request_call_hierarchy run end-to-end.

    Only request_referencing_symbols / request_containing_symbol / server.send remain scripted stubs,
    plus _convert_call_hierarchy_item (see _stub_convert_item).
    """
    ls._call_hierarchy_fallback_incoming = MagicMock(
        side_effect=lambda *args, **kwargs: SolidLanguageServer._call_hierarchy_fallback_incoming(ls, *args, **kwargs)
    )
    ls._fetch_incoming_calls_via_references = MagicMock(
        side_effect=lambda item, deadline, max_callers: SolidLanguageServer._fetch_incoming_calls_via_references(
            ls, item, deadline, max_callers
        )
    )
    ls._expand_call_hierarchy = MagicMock(side_effect=lambda **kwargs: SolidLanguageServer._expand_call_hierarchy(ls, **kwargs))
    ls._convert_call_hierarchy_item = MagicMock(side_effect=_stub_convert_item)
    ls._is_unsupported_error = MagicMock(side_effect=lambda e: SolidLanguageServer._is_unsupported_error(ls, e))
    ls._map_call_hierarchy_exception = MagicMock(
        side_effect=lambda e, method_name: SolidLanguageServer._map_call_hierarchy_exception(ls, e, method_name)
    )


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
    # Wire the REAL item converter so fetch/fallback tests drive production conversion code
    ls._call_hierarchy_item_from_symbol = MagicMock(
        side_effect=lambda symbol: SolidLanguageServer._call_hierarchy_item_from_symbol(ls, symbol)
    )
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


class TestCallHierarchyItemFromSymbol:
    """F1: Tests for _call_hierarchy_item_from_symbol conversion."""

    def test_f1_full_symbol_with_location(self, language_server_mock: MagicMock) -> None:
        """F1a: _call_hierarchy_item_from_symbol converts a full symbol with location."""
        symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "test_func",
            "kind": SymbolKind.Function,
            "location": ls_types.Location(
                uri="file:///repo/test.py",
                range={"start": {"line": 5, "character": 0}, "end": {"line": 5, "character": 10}},
                absolutePath="/repo/test.py",
                relativePath="test.py",
            ),
            "selectionRange": {"start": {"line": 5, "character": 4}, "end": {"line": 5, "character": 13}},
        }

        result = SolidLanguageServer._call_hierarchy_item_from_symbol(language_server_mock, symbol)

        assert result is not None
        assert result["name"] == "test_func"
        assert result["kind"] == SymbolKind.Function
        assert result["uri"] == "file:///repo/test.py"
        assert result["range"] == {"start": {"line": 5, "character": 0}, "end": {"line": 5, "character": 10}}
        assert result["selectionRange"] == {"start": {"line": 5, "character": 4}, "end": {"line": 5, "character": 13}}
        assert result["data"] is None

    def test_f1_symbol_without_location(self, language_server_mock: MagicMock) -> None:
        """F1b: _call_hierarchy_item_from_symbol returns None for symbol without location."""
        symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "test_var",
            "kind": SymbolKind.Variable,
        }

        result = SolidLanguageServer._call_hierarchy_item_from_symbol(language_server_mock, symbol)

        assert result is None

    def test_f1_symbol_without_range(self, language_server_mock: MagicMock) -> None:
        """F1c: _call_hierarchy_item_from_symbol returns None when location lacks range."""
        symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "test_func",
            "kind": SymbolKind.Function,
            "location": ls_types.Location(
                uri="file:///repo/test.py",
                range={"start": {"line": 5, "character": 0}, "end": {"line": 5, "character": 10}},
                absolutePath="/repo/test.py",
                relativePath="test.py",
            ),
        }
        # Deliberately delete range from location after creating it
        del cast(dict[str, object], symbol["location"])["range"]

        result = SolidLanguageServer._call_hierarchy_item_from_symbol(language_server_mock, symbol)

        assert result is None


class TestFallbackFetchGrouping:
    """F2: Tests for reference-based grouping in _fetch_incoming_calls_via_references."""

    def test_f2_reference_grouping(self, language_server_mock: MagicMock) -> None:
        """F2: Three references (2 from caller A, 1 from caller B) group into 2 calls."""
        # Create an item to search references for
        item: dict[str, object] = {
            "name": "target_func",
            "kind": SymbolKind.Function,
            "uri": "file:///repo/target.py",
            "selectionRange": {"start": {"line": 10, "character": 4}, "end": {"line": 10, "character": 15}},
            "range": {"start": {"line": 10, "character": 0}, "end": {"line": 10, "character": 20}},
            "data": None,
        }

        # Create mock referencing symbols: 2 in caller_a, 1 in caller_b
        caller_a_symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "caller_a",
            "kind": SymbolKind.Function,
            "location": ls_types.Location(
                uri="file:///repo/callers.py",
                range={"start": {"line": 20, "character": 0}, "end": {"line": 22, "character": 0}},
                absolutePath="/repo/callers.py",
                relativePath="callers.py",
            ),
            "selectionRange": {"start": {"line": 20, "character": 4}, "end": {"line": 20, "character": 12}},
        }

        caller_b_symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "caller_b",
            "kind": SymbolKind.Function,
            "location": ls_types.Location(
                uri="file:///repo/callers.py",
                range={"start": {"line": 25, "character": 0}, "end": {"line": 27, "character": 0}},
                absolutePath="/repo/callers.py",
                relativePath="callers.py",
            ),
            "selectionRange": {"start": {"line": 25, "character": 4}, "end": {"line": 25, "character": 12}},
        }

        # Two references in caller_a, one in caller_b
        mock_references = [
            ReferenceInSymbol(symbol=caller_a_symbol, line=21, character=5),
            ReferenceInSymbol(symbol=caller_a_symbol, line=21, character=20),
            ReferenceInSymbol(symbol=caller_b_symbol, line=26, character=8),
        ]

        # Mock the language server methods
        language_server_mock.request_referencing_symbols = MagicMock(return_value=mock_references)

        # Mock request_containing_symbol to return appropriate symbol based on line
        def mock_containing_symbol(rel_path: str, line: int, char: int, include_body: bool = False) -> ls_types.UnifiedSymbolInformation:
            if line == 20:
                return caller_a_symbol
            elif line == 25:
                return caller_b_symbol
            return caller_a_symbol

        language_server_mock.request_containing_symbol = MagicMock(side_effect=mock_containing_symbol)

        # Mock PathUtils methods
        with (
            patch.object(PathUtils, "uri_to_path", return_value="/repo/target.py"),
            patch.object(PathUtils, "get_relative_path", side_effect=lambda path, root: "callers.py"),
        ):
            # Call the fetch method with far future deadline and enough budget
            future_deadline = monotonic() + 1000
            calls, dropped = SolidLanguageServer._fetch_incoming_calls_via_references(
                language_server_mock, item, future_deadline, max_callers=100
            )

        # Should have 2 calls (one per distinct caller)
        assert len(calls) == 2
        assert dropped is False

        # Find caller_a and caller_b in results
        caller_a_call = next((c for c in calls if c["from"]["name"] == "caller_a"), None)
        caller_b_call = next((c for c in calls if c["from"]["name"] == "caller_b"), None)

        assert caller_a_call is not None
        assert caller_b_call is not None

        # Caller A should have 2 fromRanges, caller B should have 1
        assert len(caller_a_call["fromRanges"]) == 2
        assert len(caller_b_call["fromRanges"]) == 1

        # Verify ranges are zero-width
        for range_item in cast(list[dict[str, dict[str, int]]], caller_a_call["fromRanges"]):
            assert range_item["start"]["character"] == range_item["end"]["character"]
            assert range_item["start"]["line"] == range_item["end"]["line"]


class TestFallbackKindFilter:
    """F3: Tests for kind filtering in _fetch_incoming_calls_via_references."""

    def test_f3_exclude_variable_class_file_kinds(self, language_server_mock: MagicMock) -> None:
        """F3: References from Variable, Class, File kinds are excluded; Method/Function kept."""
        item: dict[str, object] = {
            "name": "target_func",
            "kind": SymbolKind.Function,
            "uri": "file:///repo/target.py",
            "selectionRange": {"start": {"line": 10, "character": 4}, "end": {"line": 10, "character": 15}},
            "range": {"start": {"line": 10, "character": 0}, "end": {"line": 10, "character": 20}},
            "data": None,
        }

        # Create symbols of different kinds
        method_symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "method_caller",
            "kind": SymbolKind.Method,
            "location": ls_types.Location(
                uri="file:///repo/callers.py",
                range={"start": {"line": 0, "character": 0}, "end": {"line": 2, "character": 0}},
                absolutePath="/repo/callers.py",
                relativePath="callers.py",
            ),
            "selectionRange": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 17}},
        }

        function_symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "function_caller",
            "kind": SymbolKind.Function,
            "location": ls_types.Location(
                uri="file:///repo/callers.py",
                range={"start": {"line": 3, "character": 0}, "end": {"line": 5, "character": 0}},
                absolutePath="/repo/callers.py",
                relativePath="callers.py",
            ),
            "selectionRange": {"start": {"line": 3, "character": 4}, "end": {"line": 3, "character": 19}},
        }

        variable_symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "variable_ref",
            "kind": SymbolKind.Variable,
            "location": ls_types.Location(
                uri="file:///repo/vars.py",
                range={"start": {"line": 10, "character": 0}, "end": {"line": 10, "character": 5}},
                absolutePath="/repo/vars.py",
                relativePath="vars.py",
            ),
            "selectionRange": {"start": {"line": 10, "character": 0}, "end": {"line": 10, "character": 5}},
        }

        class_symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "class_ref",
            "kind": SymbolKind.Class,
            "location": ls_types.Location(
                uri="file:///repo/classes.py",
                range={"start": {"line": 0, "character": 0}, "end": {"line": 10, "character": 0}},
                absolutePath="/repo/classes.py",
                relativePath="classes.py",
            ),
            "selectionRange": {"start": {"line": 0, "character": 6}, "end": {"line": 0, "character": 15}},
        }

        file_symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "file_ref",
            "kind": SymbolKind.File,
            "location": ls_types.Location(
                uri="file:///repo/modules.py",
                range={"start": {"line": 0, "character": 0}, "end": {"line": 100, "character": 0}},
                absolutePath="/repo/modules.py",
                relativePath="modules.py",
            ),
            "selectionRange": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
        }

        # Mock references from all kinds
        mock_references = [
            ReferenceInSymbol(symbol=method_symbol, line=0, character=10),
            ReferenceInSymbol(symbol=function_symbol, line=4, character=5),
            ReferenceInSymbol(symbol=variable_symbol, line=10, character=0),
            ReferenceInSymbol(symbol=class_symbol, line=5, character=0),
            ReferenceInSymbol(symbol=file_symbol, line=50, character=0),
        ]

        language_server_mock.request_referencing_symbols = MagicMock(return_value=mock_references)

        def mock_containing_by_path(rel_path: str, *args, **kw) -> ls_types.UnifiedSymbolInformation:
            if rel_path == "callers.py":
                return method_symbol if args[0] == 0 else function_symbol
            elif rel_path == "vars.py":
                return variable_symbol
            elif rel_path == "classes.py":
                return class_symbol
            elif rel_path == "modules.py":
                return file_symbol
            return method_symbol

        language_server_mock.request_containing_symbol = MagicMock(side_effect=mock_containing_by_path)

        # Mock PathUtils to route to different files based on the symbol's location
        def mock_uri_to_path(uri: str) -> str:
            uri_to_path_map = {
                "file:///repo/callers.py": "/repo/callers.py",
                "file:///repo/vars.py": "/repo/vars.py",
                "file:///repo/classes.py": "/repo/classes.py",
                "file:///repo/modules.py": "/repo/modules.py",
            }
            return uri_to_path_map.get(uri, "/repo/target.py")

        def mock_get_rel_path(abs_path: str, root: str) -> str:
            return abs_path.split("/")[-1]

        with (
            patch.object(PathUtils, "uri_to_path", side_effect=mock_uri_to_path),
            patch.object(PathUtils, "get_relative_path", side_effect=mock_get_rel_path),
        ):
            future_deadline = monotonic() + 1000
            calls, dropped = SolidLanguageServer._fetch_incoming_calls_via_references(
                language_server_mock, item, future_deadline, max_callers=100
            )

        # Only method and function should be included (2 calls total)
        assert len(calls) == 2
        caller_names = {c["from"]["name"] for c in calls}
        assert caller_names == {"method_caller", "function_caller"}


class TestApproximateFlag:
    """F4: fallback-activated result vs exact result — the approximate flag."""

    def test_f4_fallback_result_has_approximate_true(self, language_server_mock: MagicMock) -> None:
        """F4a: driving the REAL _call_hierarchy_fallback_incoming yields approximate=True with the derived callers."""
        _wire_fallback_engine(language_server_mock)

        leaf_symbol = _make_symbol("leaf", rel_path="a.py", line=0)
        mid_symbol = _make_symbol("mid", rel_path="a.py", line=10)

        language_server_mock.request_containing_symbol = MagicMock(return_value=leaf_symbol)
        language_server_mock.request_referencing_symbols = MagicMock(
            return_value=[ReferenceInSymbol(symbol=mid_symbol, line=11, character=8)]
        )

        result = SolidLanguageServer._call_hierarchy_fallback_incoming(
            language_server_mock, "a.py", 0, 4, depth=1, max_nodes=200, deadline=monotonic() + 1000
        )

        # production code was actually driven: root resolved, references fetched, tree expanded
        assert result["approximate"] is True
        assert [root["name"] for root in result["roots"]] == ["leaf"]
        assert [child["name"] for child in result["roots"][0]["children"]] == ["mid"]
        language_server_mock.request_referencing_symbols.assert_called_once()

    def test_f4_empty_fallback_result_still_approximate(self, language_server_mock: MagicMock) -> None:
        """F4b: a fallback whose root is not a Method/Function returns EMPTY roots but still approximate=True."""
        _wire_fallback_engine(language_server_mock)

        variable_symbol = _make_symbol("module_var", rel_path="a.py", line=0, kind=SymbolKind.Variable)
        language_server_mock.request_containing_symbol = MagicMock(return_value=variable_symbol)
        language_server_mock.request_referencing_symbols = MagicMock(return_value=[])

        result = SolidLanguageServer._call_hierarchy_fallback_incoming(
            language_server_mock, "a.py", 0, 4, depth=1, max_nodes=200, deadline=monotonic() + 1000
        )

        assert result["roots"] == []
        assert result["approximate"] is True
        # a rejected root never triggers a reference search
        language_server_mock.request_referencing_symbols.assert_not_called()

    def test_f4_exact_result_no_approximate_key(self, language_server_mock: MagicMock) -> None:
        """F4c: request_call_hierarchy with a WORKING scripted prepare yields a result with NO approximate key."""
        _wire_fallback_engine(language_server_mock)
        language_server_mock.server_started = True
        language_server_mock.server = MagicMock()

        leaf_item = SolidLanguageServer._call_hierarchy_item_from_symbol(language_server_mock, _make_symbol("leaf"))
        assert leaf_item is not None
        language_server_mock.server.send.prepare_call_hierarchy = MagicMock(return_value=[leaf_item])
        language_server_mock.server.send.incoming_calls = MagicMock(return_value=[])

        result = SolidLanguageServer.request_call_hierarchy(
            language_server_mock, "a.py", 0, 4, direction="incoming", depth=1, max_nodes=200
        )

        assert [root["name"] for root in result["roots"]] == ["leaf"]
        assert "approximate" not in result
        # the exact path never touched the fallback machinery
        language_server_mock._call_hierarchy_fallback_incoming.assert_not_called()
        language_server_mock._fetch_incoming_calls_via_references.assert_not_called()


class TestSelfRecursiveViaFallback:
    """F5: self-recursive caller through the fallback fetch — engine recursion stub in the RESULT tree."""

    def test_f5_self_recursive_caller(self, language_server_mock: MagicMock) -> None:
        """F5: references that make the root its own caller produce a node with recursion=True in the result."""
        _wire_fallback_engine(language_server_mock)

        recursive_symbol = _make_symbol("recursive_fn", rel_path="recursive.py", line=0)

        language_server_mock.request_containing_symbol = MagicMock(return_value=recursive_symbol)
        # the only reference to recursive_fn lies INSIDE recursive_fn itself (self-call)
        language_server_mock.request_referencing_symbols = MagicMock(
            return_value=[ReferenceInSymbol(symbol=recursive_symbol, line=2, character=8)]
        )

        result = SolidLanguageServer._call_hierarchy_fallback_incoming(
            language_server_mock, "recursive.py", 0, 4, depth=2, max_nodes=200, deadline=monotonic() + 1000
        )

        assert result["approximate"] is True
        assert [root["name"] for root in result["roots"]] == ["recursive_fn"]

        # the engine (unchanged by the fallback) must emit a recursion stub for the self-cycle
        children = result["roots"][0]["children"]
        assert [child["name"] for child in children] == ["recursive_fn"]
        stub = children[0]
        assert stub.get("recursion") is True
        assert stub["children"] == []


class TestFallbackDeadlineAndMaxCallers:
    """F11: Tests for deadline and max_callers bounds in _fetch_incoming_calls_via_references."""

    def test_f11_pre_expired_deadline(self, language_server_mock: MagicMock) -> None:
        """F11a: Pre-expired deadline yields 0 calls + dropped=True."""
        item: dict[str, object] = {
            "name": "target",
            "kind": SymbolKind.Function,
            "uri": "file:///repo/target.py",
            "selectionRange": {"start": {"line": 10, "character": 0}, "end": {"line": 10, "character": 6}},
            "range": {"start": {"line": 10, "character": 0}, "end": {"line": 10, "character": 6}},
            "data": None,
        }

        # Even if we have references, a past deadline should stop processing immediately
        caller_symbol: ls_types.UnifiedSymbolInformation = {
            "children": [],
            "name": "caller",
            "kind": SymbolKind.Function,
            "location": ls_types.Location(
                uri="file:///repo/callers.py",
                range={"start": {"line": 0, "character": 0}, "end": {"line": 2, "character": 0}},
                absolutePath="/repo/callers.py",
                relativePath="callers.py",
            ),
            "selectionRange": {"start": {"line": 0, "character": 4}, "end": {"line": 0, "character": 10}},
        }

        mock_references = [ReferenceInSymbol(symbol=caller_symbol, line=1, character=5)]

        language_server_mock.request_referencing_symbols = MagicMock(return_value=mock_references)
        language_server_mock.request_containing_symbol = MagicMock(return_value=caller_symbol)

        past_deadline = monotonic() - 1  # Already expired
        calls, dropped = SolidLanguageServer._fetch_incoming_calls_via_references(
            language_server_mock, item, past_deadline, max_callers=100
        )

        assert len(calls) == 0
        assert dropped is True

    def test_f11_max_callers_exceeded(self, language_server_mock: MagicMock) -> None:
        """F11b: max_callers=1 with 3 available callers yields 1 caller + dropped=True."""
        item: dict[str, object] = {
            "name": "target",
            "kind": SymbolKind.Function,
            "uri": "file:///repo/target.py",
            "selectionRange": {"start": {"line": 10, "character": 0}, "end": {"line": 10, "character": 6}},
            "range": {"start": {"line": 10, "character": 0}, "end": {"line": 10, "character": 6}},
            "data": None,
        }

        # Create 3 distinct callers
        callers = []
        for i in range(3):
            caller: ls_types.UnifiedSymbolInformation = {
                "children": [],
                "name": f"caller_{i}",
                "kind": SymbolKind.Function,
                "location": ls_types.Location(
                    uri="file:///repo/callers.py",
                    range={"start": {"line": i * 5, "character": 0}, "end": {"line": i * 5 + 2, "character": 0}},
                    absolutePath="/repo/callers.py",
                    relativePath="callers.py",
                ),
                "selectionRange": {"start": {"line": i * 5, "character": 4}, "end": {"line": i * 5, "character": 12}},
            }
            callers.append(caller)

        mock_references = [
            ReferenceInSymbol(symbol=callers[0], line=1, character=5),
            ReferenceInSymbol(symbol=callers[1], line=6, character=5),
            ReferenceInSymbol(symbol=callers[2], line=11, character=5),
        ]

        language_server_mock.request_referencing_symbols = MagicMock(return_value=mock_references)
        language_server_mock.request_containing_symbol = MagicMock(side_effect=lambda *args, **kw: callers[args[1] // 5])

        future_deadline = monotonic() + 1000
        calls, dropped = SolidLanguageServer._fetch_incoming_calls_via_references(
            language_server_mock, item, future_deadline, max_callers=1
        )

        # Should have 1 caller and dropped=True
        assert len(calls) == 1
        assert dropped is True

    def test_f11_truncated_flag_propagation(self, language_server_mock: MagicMock) -> None:
        """F11c: a dropped=True fetch inside the REAL fallback wrapper yields truncated=True in the final result,
        and the second fetch receives the REMAINING budget (a smaller max_callers than the first).
        """
        _wire_fallback_engine(language_server_mock)

        # call graph: target <- {caller_a, caller_b}; caller_a <- {caller_c, caller_d}
        target = _make_symbol("target", rel_path="a.py", line=0)
        caller_a = _make_symbol("caller_a", rel_path="a.py", line=10)
        caller_b = _make_symbol("caller_b", rel_path="a.py", line=20)
        caller_c = _make_symbol("caller_c", rel_path="a.py", line=30)
        caller_d = _make_symbol("caller_d", rel_path="a.py", line=40)

        def references_for(rel_path: str, line: int, column: int, **kwargs: object) -> list[ReferenceInSymbol]:
            if line == 0:  # target: two callers, fits within the first fetch's budget
                return [
                    ReferenceInSymbol(symbol=caller_a, line=11, character=8),
                    ReferenceInSymbol(symbol=caller_b, line=21, character=8),
                ]
            if line == 10:  # caller_a: TWO more callers, but only budget for one -> dropped=True
                return [
                    ReferenceInSymbol(symbol=caller_c, line=31, character=8),
                    ReferenceInSymbol(symbol=caller_d, line=41, character=8),
                ]
            return []

        language_server_mock.request_containing_symbol = MagicMock(return_value=target)
        language_server_mock.request_referencing_symbols = MagicMock(side_effect=references_for)

        # max_nodes=4: root(1) + 2 children = 3 handed out after the first fetch,
        # so the second fetch gets remaining budget 1 while 2 callers are available -> dropped=True
        result = SolidLanguageServer._call_hierarchy_fallback_incoming(
            language_server_mock, "a.py", 0, 4, depth=2, max_nodes=4, deadline=monotonic() + 1000
        )

        # the dropped flag from the bounded fetch must surface as truncated=True on the final result
        assert result["truncated"] is True
        assert result["approximate"] is True

        # the remaining-budget contract: the wrapper passes max_nodes minus nodes handed out so far,
        # so the second fetch call must receive a strictly smaller max_callers than the first
        fetch_calls = language_server_mock._fetch_incoming_calls_via_references.call_args_list
        assert len(fetch_calls) >= 2
        first_max_callers = fetch_calls[0].args[2]
        second_max_callers = fetch_calls[1].args[2]
        assert first_max_callers == 3  # max_nodes(4) - root(1)
        assert second_max_callers == 1  # max_nodes(4) - root(1) - 2 callers handed out
        assert second_max_callers < first_max_callers

        # tree shape sanity: caller_a expanded with exactly one (budget-capped) grandchild
        root = result["roots"][0]
        assert [child["name"] for child in root["children"]] == ["caller_a", "caller_b"]
        caller_a_node = root["children"][0]
        assert [child["name"] for child in caller_a_node["children"]] == ["caller_c"]


class TestErrorPropagation:
    """F12-F13: error propagation through request_call_hierarchy (fallback NOT activated)."""

    @staticmethod
    def _wire_request_path(ls: MagicMock) -> None:
        """Wire the real engine and a scripted server transport for driving request_call_hierarchy."""
        _wire_fallback_engine(ls)
        ls.server_started = True
        ls.server = MagicMock()

    def test_f12_32601_from_incoming_calls_after_prepare(self, language_server_mock: MagicMock) -> None:
        """F12: -32601 from incoming_calls AFTER a successful prepare propagates as the unsupported error;
        the fallback is NOT activated.
        """
        self._wire_request_path(language_server_mock)

        leaf_item = SolidLanguageServer._call_hierarchy_item_from_symbol(language_server_mock, _make_symbol("leaf"))
        assert leaf_item is not None
        language_server_mock.server.send.prepare_call_hierarchy = MagicMock(return_value=[leaf_item])
        language_server_mock.server.send.incoming_calls = MagicMock(
            side_effect=SolidLSPException(
                "Error processing request callHierarchy/incomingCalls", cause=LSPError(-32601, "method not found")
            )
        )

        with pytest.raises(SolidLSPException, match="does not support call hierarchy") as excinfo:
            SolidLanguageServer.request_call_hierarchy(language_server_mock, "a.py", 0, 4, direction="incoming", depth=1, max_nodes=200)

        # the message names the ACTUAL failed method (incomingCalls, after a successful prepare)
        assert "callHierarchy/incomingCalls" in str(excinfo.value)

        # fallback NOT activated: no reference search, no fallback entry point, no containing-symbol resolution
        language_server_mock._call_hierarchy_fallback_incoming.assert_not_called()
        language_server_mock.request_referencing_symbols.assert_not_called()
        language_server_mock.request_containing_symbol.assert_not_called()

    def test_f13_non_32601_prepare_failure(self, language_server_mock: MagicMock) -> None:
        """F13: a non--32601 prepare failure (cause code -32603) propagates unchanged; fallback NOT activated."""
        self._wire_request_path(language_server_mock)

        original = SolidLSPException("Error processing request textDocument/prepareCallHierarchy", cause=LSPError(-32603, "internal error"))
        language_server_mock.server.send.prepare_call_hierarchy = MagicMock(side_effect=original)

        with pytest.raises(SolidLSPException) as excinfo:
            SolidLanguageServer.request_call_hierarchy(language_server_mock, "a.py", 0, 4, direction="incoming", depth=1, max_nodes=200)

        # the ORIGINAL error propagates -- it is not remapped to the unsupported message
        assert excinfo.value is original
        assert "does not support call hierarchy" not in str(excinfo.value)

        # fallback NOT activated
        language_server_mock._call_hierarchy_fallback_incoming.assert_not_called()
        language_server_mock.request_referencing_symbols.assert_not_called()
        language_server_mock.request_containing_symbol.assert_not_called()
        language_server_mock.server.send.incoming_calls.assert_not_called()
