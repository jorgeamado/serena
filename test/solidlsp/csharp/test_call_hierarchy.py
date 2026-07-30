"""C# call hierarchy tests (spec 7b).

Runtime verification (2026-07-30, devcontainer, .NET 10.0.10 + libicu installed, server fully operational —
test_csharp_basic passes): Microsoft.CodeAnalysis.LanguageServer 5.5.0-2.26078.4 (the version pinned by this
repo) does NOT advertise `callHierarchyProvider` in its initialize response and answers
`textDocument/prepareCallHierarchy` with LSPError -32601
("No method by the name 'textDocument/prepareCallHierarchy' is found.").

The spec's L1+L2 mirror therefore cannot pass against this server; per spec 7b ("if it does not answer
prepare, stop and report instead of forcing it") this file instead pins the row-8 behaviour: the mid-layer
maps the -32601 into a descriptive SolidLSPException. The CallGraph.cs fixture (EntryA/EntryB -> Mid -> Leaf)
stays in place so the L1+L2 mirror can be restored once a Roslyn version with call hierarchy support is adopted.
"""

import os

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import LanguageServerId
from solidlsp.ls_exceptions import SolidLSPException
from test.conftest import find_identifier_position, get_repo_path

CALL_GRAPH_FILE = os.path.join("CallGraph.cs")


@pytest.mark.csharp
class TestCSharpCallHierarchy:
    @pytest.mark.parametrize("language_server", [LanguageServerId.CSHARP], indirect=True)
    def test_f8_csharp_incoming_with_fallback(self, language_server: SolidLanguageServer) -> None:
        """F8: C# incoming of Mid at depth=2; fallback activated; EntryA and EntryB present; approximate True."""
        repo_path = get_repo_path(LanguageServerId.CSHARP)
        mid_pos = find_identifier_position(repo_path / "CallGraph.cs", "Mid")
        assert mid_pos is not None, "Could not find CallGraph.Mid in fixture"

        result = language_server.request_call_hierarchy(CALL_GRAPH_FILE, *mid_pos, direction="incoming", depth=2, max_nodes=200)

        # Should NOT raise; should return a result
        assert isinstance(result, dict)
        assert len(result["roots"]) == 1
        root = result["roots"][0]
        assert root["name"] == "Mid"

        # Should have approximate flag (fallback activated)
        assert result.get("approximate") is True

        # At depth=2: Mid -> [EntryA, EntryB]
        level_1_callers = root["children"]
        level_1_names = {caller["name"] for caller in level_1_callers}
        assert level_1_names == {"EntryA", "EntryB"}

    @pytest.mark.parametrize("language_server", [LanguageServerId.CSHARP], indirect=True)
    def test_f9_csharp_outgoing_raises(self, language_server: SolidLanguageServer) -> None:
        """F9: C# outgoing of Mid raises unsupported error (fallback not used for outgoing)."""
        repo_path = get_repo_path(LanguageServerId.CSHARP)
        mid_pos = find_identifier_position(repo_path / "CallGraph.cs", "Mid")
        assert mid_pos is not None, "Could not find CallGraph.Mid in fixture"

        with pytest.raises(SolidLSPException, match="does not support call hierarchy"):
            language_server.request_call_hierarchy(CALL_GRAPH_FILE, *mid_pos, direction="outgoing", depth=1, max_nodes=200)
