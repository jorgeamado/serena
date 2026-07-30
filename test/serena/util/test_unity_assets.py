"""Unit tests for the static Unity asset-reference index (:mod:`serena.util.unity_assets`).

These tests are pure (no language server, no agent, no running Unity): they build a tiny synthetic
Unity project on disk and assert the index's behavior against it.
"""

from pathlib import Path

import pytest

from serena.util.unity_assets import UnityAssetIndex

GUID_PLAYER = "11111111111111111111111111111111"
GUID_ENEMY = "22222222222222222222222222222222"  # unreferenced
GUID_PREFAB = "33333333333333333333333333333333"
GUID_MAT = "44444444444444444444444444444444"
GUID_MISSING = "99999999999999999999999999999999"  # referenced but has no asset


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def unity_project(tmp_path: Path) -> Path:
    """Builds a minimal text-serialized Unity project and returns its root."""
    root = tmp_path / "proj"

    _write(root / "ProjectSettings" / "ProjectVersion.txt", "m_EditorVersion: 6000.3.16f1\n")
    _write(root / "ProjectSettings" / "EditorSettings.asset", "EditorSettings:\n  m_SerializationMode: 2\n")

    # Scripts (identity lives in the adjacent .meta)
    _write(root / "Assets" / "Scripts" / "Player.cs", "public class Player {}\n")
    _write(root / "Assets" / "Scripts" / "Player.cs.meta", f"fileFormatVersion: 2\nguid: {GUID_PLAYER}\nMonoImporter:\n")
    _write(root / "Assets" / "Scripts" / "Enemy.cs", "public class Enemy {}\n")
    _write(root / "Assets" / "Scripts" / "Enemy.cs.meta", f"fileFormatVersion: 2\nguid: {GUID_ENEMY}\n")

    _write(root / "Assets" / "Art" / "Red.mat.meta", f"fileFormatVersion: 2\nguid: {GUID_MAT}\n")
    _write(
        root / "Assets" / "Art" / "Red.mat",
        "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!21 &2100000\nMaterial:\n  m_Name: Red\n",
    )

    # A prefab that attaches the Player MonoBehaviour.
    _write(root / "Assets" / "Prefabs" / "Player.prefab.meta", f"fileFormatVersion: 2\nguid: {GUID_PREFAB}\n")
    _write(
        root / "Assets" / "Prefabs" / "Player.prefab",
        "%YAML 1.1\n"
        "%TAG !u! tag:unity3d.com,2011:\n"
        "--- !u!1 &100\n"
        "GameObject:\n"
        "  m_Component:\n"
        "  - component: {fileID: 114001}\n"
        "--- !u!114 &114001\n"
        "MonoBehaviour:\n"
        f"  m_Script: {{fileID: 11500000, guid: {GUID_PLAYER}, type: 3}}\n"
        "  m_Name: Player\n",
    )

    # A scene that: attaches Player directly, instantiates Player.prefab, references a material,
    # and holds a dangling reference to a missing GUID.
    _write(root / "Assets" / "Scenes" / "Main.unity.meta", "fileFormatVersion: 2\nguid: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
    _write(
        root / "Assets" / "Scenes" / "Main.unity",
        "%YAML 1.1\n"
        "%TAG !u! tag:unity3d.com,2011:\n"
        "--- !u!114 &500\n"
        "MonoBehaviour:\n"
        f"  m_Script: {{fileID: 11500000, guid: {GUID_PLAYER}, type: 3}}\n"
        "--- !u!1001 &600\n"
        "PrefabInstance:\n"
        "  m_Modification:\n"
        f"    m_SourcePrefab: {{fileID: 100100000, guid: {GUID_PREFAB}, type: 3}}\n"
        "--- !u!23 &700\n"
        "MeshRenderer:\n"
        "  m_Materials:\n"
        f"  - {{fileID: 2100000, guid: {GUID_MAT}, type: 2}}\n"
        f"  m_MissingRef: {{fileID: 0, guid: {GUID_MISSING}, type: 3}}\n",
    )

    # A Library-cache copy that also attaches Player — must be skipped (Library is a Unity cache dir).
    _write(
        root / "Library" / "junk.prefab",
        f"--- !u!114 &1\nMonoBehaviour:\n  m_Script: {{fileID: 11500000, guid: {GUID_PLAYER}, type: 3}}\n",
    )

    return root


def test_is_unity_project(unity_project: Path, tmp_path: Path) -> None:
    assert UnityAssetIndex(unity_project).is_unity_project() is True
    assert UnityAssetIndex(tmp_path / "not_unity").is_unity_project() is False


def test_serialization_mode(unity_project: Path, tmp_path: Path) -> None:
    assert UnityAssetIndex(unity_project).get_serialization_mode() == "ForceText"
    assert UnityAssetIndex(tmp_path / "empty").get_serialization_mode() is None


def test_guid_for_asset(unity_project: Path) -> None:
    idx = UnityAssetIndex(unity_project)
    assert idx.guid_for_asset("Assets/Scripts/Player.cs") == GUID_PLAYER
    assert idx.guid_for_asset("Assets/Scripts/DoesNotExist.cs") is None  # no .meta


def test_reverse_script_attachment(unity_project: Path) -> None:
    """A MonoBehaviour script's GUID is found wherever it is attached, as reference_kind 'script'."""
    refs = UnityAssetIndex(unity_project).find_references_to_guid(GUID_PLAYER)
    by_file = {r.referencing_file for r in refs}
    # Both the prefab and the scene attach it; the Library copy is skipped.
    assert by_file == {"Assets/Prefabs/Player.prefab", "Assets/Scenes/Main.unity"}
    assert all(r.reference_kind == "script" and r.property == "m_Script" for r in refs)
    assert all(r.object_class_name == "MonoBehaviour" and r.object_class_id == 114 for r in refs)
    assert not any("Library" in r.referencing_file for r in refs)


def test_reverse_prefab_instantiation(unity_project: Path) -> None:
    """A prefab's GUID is found where a scene instantiates it, via m_SourcePrefab on a PrefabInstance."""
    refs = UnityAssetIndex(unity_project).find_references_to_asset("Assets/Prefabs/Player.prefab")
    assert [r.referencing_file for r in refs] == ["Assets/Scenes/Main.unity"]
    (ref,) = refs
    assert ref.property == "m_SourcePrefab"
    assert ref.reference_kind == "reference"
    assert ref.object_class_name == "PrefabInstance" and ref.object_class_id == 1001


def test_property_attribution_for_sequence_item(unity_project: Path) -> None:
    """A reference that is a YAML sequence item is attributed to its enclosing mapping key."""
    (ref,) = UnityAssetIndex(unity_project).find_references_to_guid(GUID_MAT)
    assert ref.referencing_file == "Assets/Scenes/Main.unity"
    assert ref.property == "m_Materials"
    assert ref.object_class_name == "MeshRenderer"


def test_dangling_reference_is_reported(unity_project: Path) -> None:
    """A reference to a GUID with no backing asset is still reported (useful for finding broken refs)."""
    (ref,) = UnityAssetIndex(unity_project).find_references_to_guid(GUID_MISSING)
    assert ref.referencing_file == "Assets/Scenes/Main.unity"
    assert ref.property == "m_MissingRef"


def test_unreferenced_guid(unity_project: Path) -> None:
    assert UnityAssetIndex(unity_project).find_references_to_guid(GUID_ENEMY) == []


def test_scope_narrows_search(unity_project: Path) -> None:
    refs = UnityAssetIndex(unity_project).find_references_to_guid(GUID_PLAYER, scope_relative_path="Assets/Scenes")
    assert [r.referencing_file for r in refs] == ["Assets/Scenes/Main.unity"]


def test_find_references_to_asset_excludes_self(unity_project: Path) -> None:
    """Searching references to an asset never reports the asset file itself."""
    refs = UnityAssetIndex(unity_project).find_references_to_asset("Assets/Prefabs/Player.prefab")
    assert "Assets/Prefabs/Player.prefab" not in {r.referencing_file for r in refs}
