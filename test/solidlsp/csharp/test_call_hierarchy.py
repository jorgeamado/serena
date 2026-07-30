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
    def test_call_hierarchy_unsupported_raises(self, language_server: SolidLanguageServer) -> None:
        """The C# server does not implement prepareCallHierarchy; the mid-layer maps the -32601 descriptively."""
        repo_path = get_repo_path(LanguageServerId.CSHARP)
        leaf_pos = find_identifier_position(repo_path / "CallGraph.cs", "Leaf")
        assert leaf_pos is not None, "Could not find CallGraph.Leaf in fixture"

        with pytest.raises(SolidLSPException, match="does not support call hierarchy"):
            language_server.request_call_hierarchy(CALL_GRAPH_FILE, *leaf_pos, direction="incoming", depth=1, max_nodes=200)
