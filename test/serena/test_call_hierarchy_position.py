"""Tool-level tests for CallHierarchyTool's POSITION anchor (relative_path:line:column).

The position anchor resolves the symbol at a usage position via the language server's hover and, for
property/field/event members, classifies its references (this reaches framework/external members that
name-path cannot). A position that resolves to a method returns a structured message.

Reuses the C# fixture `Members.cs` (Bag.Value property, Bag.Count field, Bag.Changed event, all with
usages in BagUser) and the module-scoped `csharp_agent` fixture convention.
"""

import json
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from serena.agent import SerenaAgent
from serena.tools import CallHierarchyTool
from solidlsp.ls_config import LanguageServerId
from test.conftest import agent_for_project_context

MEMBERS_FILE = "Members.cs"


@pytest.fixture(scope="module")
def csharp_agent() -> Iterator[SerenaAgent]:
    with agent_for_project_context(LanguageServerId.CSHARP) as agent:
        yield agent


def _pos(agent: SerenaAgent, line_substr: str, token: str) -> tuple[int, int]:
    """Locate the (0-based line, 0-based column) of `token` on the first line containing `line_substr`."""
    lines = agent.get_active_project_or_raise().read_file(MEMBERS_FILE).split("\n")
    for i, ln in enumerate(lines):
        if line_substr in ln:
            return i, ln.index(token, ln.index(line_substr))
    raise AssertionError(f"line containing {line_substr!r} not found")


def _access_kinds(output: dict) -> set[str]:
    kinds: set[str] = set()

    def walk(nodes: list[dict]) -> None:
        for n in nodes:
            for r in n.get("call_sites", {}).get("ranges", []):
                kinds.add(r["access_kind"])
            walk(n.get("children", []))

    walk(output.get("incoming", []))
    return kinds


@pytest.mark.csharp
class TestPositionAnchor:
    def test_property_usage_position(self, csharp_agent: SerenaAgent) -> None:
        """Anchor on a `b.Value` usage -> Property member; references include read AND write sites."""
        line, col = _pos(csharp_agent, "=> b.Value;", "Value")
        tool = csharp_agent.get_tool(CallHierarchyTool)
        out = json.loads(tool.apply(relative_path=MEMBERS_FILE, line=line, column=col, direction="incoming"))
        assert out["member_kind"] == "Property"
        assert out["approximate"] is True
        assert out["incoming"][0]["name"] == "Value"
        assert {"read", "write"} <= _access_kinds(out)

    def test_field_usage_position(self, csharp_agent: SerenaAgent) -> None:
        line, col = _pos(csharp_agent, "=> b.Count;", "Count")
        tool = csharp_agent.get_tool(CallHierarchyTool)
        out = json.loads(tool.apply(relative_path=MEMBERS_FILE, line=line, column=col, direction="incoming"))
        assert out["member_kind"] == "Field"
        assert len(out["incoming"][0]["children"]) >= 1

    def test_event_usage_position(self, csharp_agent: SerenaAgent) -> None:
        """Anchor on a `b.Changed` usage -> Event member; references include subscribe AND unsubscribe."""
        line, col = _pos(csharp_agent, "b.Changed += h", "Changed")
        tool = csharp_agent.get_tool(CallHierarchyTool)
        out = json.loads(tool.apply(relative_path=MEMBERS_FILE, line=line, column=col, direction="incoming"))
        assert out["member_kind"] == "Event"
        assert {"subscribe", "unsubscribe"} <= _access_kinds(out)

    def test_method_position_returns_message(self, csharp_agent: SerenaAgent) -> None:
        """A position resolving to a method returns a structured message (not a member result)."""
        line, col = _pos(csharp_agent, "public void Fire()", "Fire")
        tool = csharp_agent.get_tool(CallHierarchyTool)
        out = json.loads(tool.apply(relative_path=MEMBERS_FILE, line=line, column=col, direction="incoming"))
        assert "error" in out
        assert "method" in out["error"]


class TestPositionValidation:
    def test_no_name_path_and_no_position_raises(self) -> None:
        tool = CallHierarchyTool(MagicMock())
        with pytest.raises(ValueError, match="name_path"):
            tool.apply(relative_path="x.cs")

    def test_position_without_relative_path_raises(self) -> None:
        tool = CallHierarchyTool(MagicMock())
        with pytest.raises(ValueError, match="relative_path is required"):
            tool.apply(line=1, column=1)

    def test_partial_position_raises(self) -> None:
        tool = CallHierarchyTool(MagicMock())
        with pytest.raises(ValueError, match="together"):
            tool.apply(relative_path="x.cs", line=1)  # column omitted

    def test_name_path_and_position_together_raises(self) -> None:
        tool = CallHierarchyTool(MagicMock())
        with pytest.raises(ValueError, match="not both"):
            tool.apply(name_path="Foo/Bar", relative_path="x.cs", line=1, column=1)
