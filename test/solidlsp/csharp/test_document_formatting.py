"""
Live C# tests for document formatting against the pinned Roslyn
(`Microsoft.CodeAnalysis.LanguageServer`) language server (SPEC-format-file.md rev 2, rows L3
and L4). These require a working .NET/Roslyn toolchain and are intended to run in the
devcontainer, NOT on the host (see the C# fixture `Misformatted.cs`, deliberately misformatted
but valid, compilable C#, added alongside `Program.cs` / `Models/Person.cs` in this test_repo).

Host-side, only `pytest --collect-only` is exercised for this file: on the current tree,
`request_document_formatting` and `get_server_capabilities` do not exist on `SolidLanguageServer`,
so running these tests (in the devcontainer) fails at the first feature-specific line with
`AttributeError: 'SolidLanguageServer' object has no attribute ...` (or the corresponding
`apply_text_edits_to_text` ImportError for L3) -- never a collection error, since all such access
happens inside the test bodies.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import LanguageServerId


def _import_apply_text_edits_to_text() -> Callable[[str, list[dict[str, Any]]], str]:
    try:
        from solidlsp.ls_utils import apply_text_edits_to_text  # type: ignore[attr-defined]

        return apply_text_edits_to_text
    except ImportError:
        pass
    from solidlsp.ls import apply_text_edits_to_text  # type: ignore[attr-defined]

    return apply_text_edits_to_text


@pytest.mark.csharp
class TestCSharpDocumentFormatting:
    @pytest.mark.parametrize("language_server", [LanguageServerId.CSHARP], indirect=True)
    def test_l3_request_document_formatting_and_pure_apply_on_misformatted_fixture(
        self, language_server: SolidLanguageServer
    ) -> None:
        relative_path = "Misformatted.cs"
        apply_text_edits_to_text = _import_apply_text_edits_to_text()

        abs_path = Path(language_server.language_server.repository_root_path) / relative_path
        original_bytes = abs_path.read_bytes()
        original_hash = hashlib.sha256(original_bytes).hexdigest()

        with language_server.open_file(relative_path, open_in_ls=False) as file_buffer:
            original_text = file_buffer.contents

        edits = language_server.request_document_formatting(relative_path)
        assert edits, "expected non-empty edits for the deliberately misformatted fixture"

        formatted_text = apply_text_edits_to_text(original_text, edits)

        # precise expected normalization: the cramped constructor-body line becomes exactly its
        # 4-space-indented form (namespace > class > constructor body = 3 levels = 12 spaces),
        # matching the indentation convention already used throughout this fixture repo (see
        # Program.cs: namespace > class > method body is likewise indented 12 spaces).
        assert "\n            Value = value;\n" in formatted_text

        # further normalizations Roslyn's default formatter applies regardless of brace style
        # (spacing around binary/assignment operators is on by default):
        assert "Value * 2" in formatted_text
        assert "Value > 0" in formatted_text

        # rev 2.1: the applied result must preserve the fixture's full non-whitespace content --
        # guards against an applier that drops or duplicates content. Compared as the SEQUENCE OF
        # NON-WHITESPACE CHARACTERS (whitespace-token lists would be too strict: a formatter
        # legitimately inserts whitespace inside previously glued tokens such as `Value=value;`).
        assert "".join(formatted_text.split()) == "".join(original_text.split()), (
            "formatting must be whitespace-only: the non-whitespace character sequence changed"
        )

        # the fixture file on disk must be untouched -- compare BYTES via hash, not `git status`
        # (this repo generates local state dirs such as `.serena/cache`, `obj/`, which would make
        # a git-status-based check noisy/misleading).
        assert hashlib.sha256(abs_path.read_bytes()).hexdigest() == original_hash

    @pytest.mark.parametrize("language_server", [LanguageServerId.CSHARP], indirect=True)
    def test_l4_get_server_capabilities_reports_document_formatting_provider(
        self, language_server: SolidLanguageServer
    ) -> None:
        capabilities = language_server.get_server_capabilities()
        assert capabilities is not None, "expected capabilities to have been captured from the live initialize response"
        assert "documentFormattingProvider" in capabilities
        assert capabilities["documentFormattingProvider"] is not False
