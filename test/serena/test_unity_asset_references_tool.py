"""
Tool-layer tests for ``UnityAssetReferencesTool``.

These exercise the tool's JSON contract and error branches with a fully mocked agent/project (no
language server, no real SerenaAgent) against a tiny synthetic Unity project. The underlying scan
logic is covered separately in ``test/serena/util/test_unity_assets.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from serena.tools.unity_tools import UnityAssetReferencesTool

GUID_PLAYER = "11111111111111111111111111111111"
GUID_PREFAB = "33333333333333333333333333333333"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def unity_root(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    _write(root / "ProjectSettings" / "ProjectVersion.txt", "m_EditorVersion: 6000.3.16f1\n")
    _write(root / "ProjectSettings" / "EditorSettings.asset", "EditorSettings:\n  m_SerializationMode: 2\n")
    _write(root / "Assets" / "Scripts" / "Player.cs", "public class Player {}\n")
    _write(root / "Assets" / "Scripts" / "Player.cs.meta", f"fileFormatVersion: 2\nguid: {GUID_PLAYER}\n")
    _write(root / "Assets" / "Prefabs" / "Player.prefab.meta", f"fileFormatVersion: 2\nguid: {GUID_PREFAB}\n")
    _write(
        root / "Assets" / "Prefabs" / "Player.prefab",
        f"%YAML 1.1\n--- !u!114 &1\nMonoBehaviour:\n  m_Script: {{fileID: 11500000, guid: {GUID_PLAYER}, type: 3}}\n",
    )
    return root


def _make_tool(project_root: Path) -> UnityAssetReferencesTool:
    """Builds a tool instance backed by a mocked agent/project rooted at ``project_root``."""
    tool = object.__new__(UnityAssetReferencesTool)
    agent = MagicMock()
    project = MagicMock()
    project.project_root = str(project_root)
    project.is_ignored_path.return_value = False
    agent.get_active_project_or_raise.return_value = project
    agent.serena_config.default_max_tool_answer_chars = 200_000
    tool.agent = agent  # type: ignore[attr-defined]
    return tool


def test_apply_reverse_reference(unity_root: Path) -> None:
    out = json.loads(_make_tool(unity_root).apply(relative_path="Assets/Scripts/Player.cs"))
    assert out["target"] == {"relative_path": "Assets/Scripts/Player.cs", "guid": GUID_PLAYER}
    assert out["serialization_mode"] == "ForceText"
    assert out["reference_count"] == 1
    (ref,) = out["references"]
    assert ref["referencing_file"] == "Assets/Prefabs/Player.prefab"
    assert ref["reference_kind"] == "script"
    assert ref["property"] == "m_Script"
    assert "notes" not in out  # ForceText project => no caveats


def test_apply_by_guid(unity_root: Path) -> None:
    out = json.loads(_make_tool(unity_root).apply(guid=GUID_PLAYER.upper()))  # case-insensitive input
    assert out["target"] == {"guid": GUID_PLAYER}
    assert out["reference_count"] == 1


def test_apply_not_unity_project(tmp_path: Path) -> None:
    out = json.loads(_make_tool(tmp_path / "empty").apply(guid=GUID_PLAYER))
    assert "error" in out and "Unity project" in out["error"]


def test_apply_invalid_guid(unity_root: Path) -> None:
    out = json.loads(_make_tool(unity_root).apply(guid="not-a-valid-guid"))
    assert "error" in out and "GUID" in out["error"]


def test_apply_unresolved_path(unity_root: Path) -> None:
    out = json.loads(_make_tool(unity_root).apply(relative_path="Assets/Scripts/DoesNotExist.cs"))
    assert "error" in out and ".meta" in out["error"]


def test_apply_requires_input(unity_root: Path) -> None:
    out = json.loads(_make_tool(unity_root).apply())
    assert "error" in out
