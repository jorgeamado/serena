"""Tool-level tests for CallHierarchyTool on non-callable members (property/field/event).

Covers SPEC-call-hierarchy-members.md (Rev 2) §6b rows M1-M9. On base `main` the tool has no
kind-based routing: every property/field/event lookup falls into the callable "prepare returned
nothing" branch (symbol_tools.py:819) and returns the frozen F1 string instead of JSON. Rows that
expect JSON first assert `result != F1` -- that assertion is the declared, precise RED cause for this
file (as opposed to a collection-wide error).

Conventions follow test/serena/test_call_hierarchy_tool.py: a module-scoped `csharp_agent` fixture
(from agent_for_project_context) amortizes the ~2-5s Roslyn warmup across all LS-backed rows.

Rev 2 shape: the member result carries a synthetic root = the queried member as incoming[0], with the
referencing symbols as its children; each child's call_sites.ranges[*] carries an `access_kind`.
"""

import json
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from serena.agent import SerenaAgent
from serena.tools import CallHierarchyTool
from solidlsp.ls_config import LanguageServerId
from solidlsp.lsp_protocol_handler.lsp_types import SymbolKind
from test.conftest import agent_for_project_context

MEMBERS_FILE = "Members.cs"
CALL_GRAPH_FILE = "CallGraph.cs"

# Frozen strings (SPEC Rev 2 §7). Test files own DUPLICATED copies compared by equality -- do NOT
# import the production constants into the oracle.
F1 = "No callable symbol found at the given position"
F2 = (
    "incoming usages for this property, field, or event were derived from find-references and "
    "classified by access kind syntactically; results are approximate and may include non-access "
    "usages or 'unknown' where a site could not be classified."
)
F3 = "outgoing calls are not applicable to a property, field, or event"


@pytest.fixture(scope="module")
def csharp_agent() -> Iterator[SerenaAgent]:
    """A SerenaAgent over the C# test repo with a warm language server."""
    with agent_for_project_context(LanguageServerId.CSHARP) as agent:
        yield agent


def _collect_access_kinds(nodes: list[dict]) -> list[str]:
    """Flatten every call-site `access_kind` out of a (possibly nested) incoming-hierarchy node list."""
    kinds: list[str] = []
    for node in nodes:
        call_sites = node.get("call_sites")
        if call_sites:
            for rng in call_sites.get("ranges", []):
                if "access_kind" in rng:
                    kinds.append(rng["access_kind"])
        kinds.extend(_collect_access_kinds(node.get("children", [])))
    return kinds


def _key_present_anywhere(obj: object, key: str) -> bool:
    """True if `key` appears in `obj` or any nested dict/list within it."""
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_key_present_anywhere(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_key_present_anywhere(item, key) for item in obj)
    return False


class TestCallHierarchyMembersZeroReferencesMocked:
    """M9 uses a mocked retriever (same pattern as _make_approximate_retriever /
    TestCallHierarchyToolApproximate in test_call_hierarchy_tool.py) because no zero-reference member
    exists in the shared C# fixture tree.
    """

    def test_m9_property_zero_references_returns_valid_json_not_f1(self) -> None:
        """M9: a property with zero find-references usages must still return a valid JSON object with a
        synthetic root whose `children==[]` -- NOT the F1 string (SPEC §4.3.6 reserves F1 for callables
        whose prepare is genuinely empty).
        """
        symbol = MagicMock()
        symbol.symbol_kind = SymbolKind.Property  # Rev 2 routes on symbol.symbol_kind (enum, value 7)
        symbol.get_name_path.return_value = "Bag/Value"

        retriever = MagicMock()
        retriever.find_unique.return_value = symbol
        retriever.find_referencing_symbols_by_location.return_value = []
        # Base path (no member routing) still issues the callable request; empty roots -> F1.
        retriever.request_call_hierarchy_by_location.return_value = {"roots": [], "truncated": False, "external_calls_omitted": 0}
        retriever.get_language_server.return_value.ls_id.value = "csharp"

        agent = MagicMock()
        agent.serena_config.default_max_tool_answer_chars = 1_000_000  # a real int budget for the default (-1) path
        tool = CallHierarchyTool(agent)
        with patch.object(CallHierarchyTool, "create_language_server_symbol_retriever", return_value=retriever):
            result = tool.apply(name_path="Bag/Value", relative_path=MEMBERS_FILE, direction="incoming", depth=1)

        assert result != F1, "a zero-reference property must not collapse to the callable 'No callable...' message"
        output = json.loads(result)
        assert output["member_kind"] == "Property"
        assert output["approximate"] is True
        assert output["incoming"][0]["children"] == []

    def test_m10_member_incoming_small_max_answer_chars_shortened_form(self) -> None:
        """M10: a member incoming result whose full JSON exceeds a small `max_answer_chars` must fall back
        (via _limit_length) to a shortened form that STILL carries `member_kind`, `approximate`, and the
        exact F2 note, and that DROPS the per-site `access_kind` detail (SPEC §4.4). Several referencing
        sites are mocked so the full form is comfortably over the limit.
        """
        symbol = MagicMock()
        symbol.symbol_kind = SymbolKind.Property  # Rev 2 routes on symbol.symbol_kind (enum, value 7)
        symbol.get_name_path.return_value = "Bag/Value"

        retriever = MagicMock()
        retriever.find_unique.return_value = symbol
        # Many distinct referencing sites so the full JSON (with per-site access_kind) is large.
        refs = []
        for i in range(20):
            ref = MagicMock()
            ref.line = i + 10
            ref.character = 8
            containing = MagicMock()
            containing.get_name_path.return_value = f"BagUser/Use{i}"
            containing.symbol_kind = SymbolKind.Method
            ref.symbol = containing
            refs.append(ref)
        retriever.find_referencing_symbols_by_location.return_value = refs
        # Base path (no member routing) still issues the callable request; empty roots -> F1.
        retriever.request_call_hierarchy_by_location.return_value = {"roots": [], "truncated": False, "external_calls_omitted": 0}
        retriever.get_language_server.return_value.ls_id.value = "csharp"

        tool = CallHierarchyTool(MagicMock())
        with patch.object(CallHierarchyTool, "create_language_server_symbol_retriever", return_value=retriever):
            result = tool.apply(name_path="Bag/Value", relative_path=MEMBERS_FILE, direction="incoming", depth=1, max_answer_chars=1000)

        assert result != F1, "a member incoming query must not collapse to the callable 'No callable...' message"
        assert "member_kind" in result
        assert "approximate" in result
        assert F2 in result
        assert "access_kind" not in result


@pytest.mark.csharp
class TestCallHierarchyMembersWithLanguageServer:
    def test_m1_property_incoming_synthetic_root(self, csharp_agent: SerenaAgent) -> None:
        """M1: property incoming -> JSON; member_kind=='Property'; approximate==True;
        incoming[0].name=='Value' (synthetic root); >=1 child.
        """
        tool = csharp_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="Bag/Value", relative_path=MEMBERS_FILE, direction="incoming", depth=1)

        assert result != F1, "base falls through to the callable 'No callable...' message for properties"
        output = json.loads(result)
        assert output["member_kind"] == "Property"
        assert output["approximate"] is True
        assert output["incoming"][0]["name"] == "Value"
        assert len(output["incoming"][0]["children"]) >= 1

    def test_m2_property_read_and_write_sites_classified(self, csharp_agent: SerenaAgent) -> None:
        """M2: across all child call-site ranges of Bag/Value, the access_kind set includes 'read' (Read)
        AND 'write' (Write).
        """
        tool = csharp_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="Bag/Value", relative_path=MEMBERS_FILE, direction="incoming", depth=1)

        assert result != F1
        output = json.loads(result)
        kinds = _collect_access_kinds(output["incoming"])
        assert "read" in kinds
        assert "write" in kinds

    def test_m3_field_incoming(self, csharp_agent: SerenaAgent) -> None:
        """M3: Bag/Count (field) -> member_kind=='Field'; >=1 child."""
        tool = csharp_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="Bag/Count", relative_path=MEMBERS_FILE, direction="incoming", depth=1)

        assert result != F1
        output = json.loads(result)
        assert output["member_kind"] == "Field"
        assert len(output["incoming"][0]["children"]) >= 1

    def test_m4_event_subscribe_and_unsubscribe_classified(self, csharp_agent: SerenaAgent) -> None:
        """M4: Bag/Changed (event) -> child ranges' access_kind set includes 'subscribe' (Sub) and
        'unsubscribe' (Unsub).
        """
        tool = csharp_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="Bag/Changed", relative_path=MEMBERS_FILE, direction="incoming", depth=1)

        assert result != F1
        output = json.loads(result)
        assert output["member_kind"] == "Event"
        kinds = _collect_access_kinds(output["incoming"])
        assert "subscribe" in kinds
        assert "unsubscribe" in kinds

    def test_m5_method_control_unchanged(self, csharp_agent: SerenaAgent) -> None:
        """M5 (characterization control): a Method result has NO `member_kind` key and NO per-site
        `access_kind` key anywhere. Must call the real, unmocked tool.apply(); GREEN on base.
        """
        tool = csharp_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="CallGraph/Leaf", relative_path=CALL_GRAPH_FILE, direction="incoming", depth=1)

        output = json.loads(result)
        assert "member_kind" not in output
        assert not _key_present_anywhere(output, "access_kind")

    def test_m6_direction_both_on_property(self, csharp_agent: SerenaAgent) -> None:
        """M6: direction='both' on a property -> outgoing==[] and outgoing_unavailable==F3 present;
        `incoming` present.
        """
        tool = csharp_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="Bag/Value", relative_path=MEMBERS_FILE, direction="both", depth=1)

        assert result != F1
        output = json.loads(result)
        assert output["outgoing"] == []
        assert output["outgoing_unavailable"] == F3
        assert "incoming" in output

    def test_m7_direction_outgoing_on_property(self, csharp_agent: SerenaAgent) -> None:
        """M7: direction='outgoing' on a property -> outgoing==[], outgoing_unavailable==F3, NO
        `incoming` key, NO `approximate` key. Base returns F1 for outgoing member too -> assert != F1.
        """
        tool = csharp_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="Bag/Value", relative_path=MEMBERS_FILE, direction="outgoing", depth=1)

        assert result != F1
        output = json.loads(result)
        assert output["outgoing"] == []
        assert output["outgoing_unavailable"] == F3
        assert "incoming" not in output
        assert "approximate" not in output

    def test_m8_approximate_note_equals_frozen_f2(self, csharp_agent: SerenaAgent) -> None:
        """M8: approximate_note equals F2 exactly (property incoming; frozen-string equality)."""
        tool = csharp_agent.get_tool(CallHierarchyTool)
        result = tool.apply(name_path="Bag/Value", relative_path=MEMBERS_FILE, direction="incoming", depth=1)

        assert result != F1
        output = json.loads(result)
        assert output["approximate_note"] == F2
