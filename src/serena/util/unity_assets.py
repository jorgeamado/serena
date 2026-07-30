"""
Static, LSP-independent analysis of Unity's asset graph.

Unity serializes scenes (``.unity``), prefabs (``.prefab``), ScriptableObjects and other
assets (``.asset``, ``.mat``, ``.controller``, ...) as YAML that references scripts and other
assets *by GUID* (and a numeric ``fileID``), not by name. The GUID of an asset is declared in
its adjacent ``.meta`` file. No language server understands these references — a C# server has
no idea that a ``.prefab`` attaches a given ``MonoBehaviour`` — which is exactly why this
module exists.

This module reads files on disk only. It does not require a running Unity Editor, a license, or
any Unity package. It works on a plain VCS checkout (including a checkout that would not import
cleanly), which is the niche the Unity CLI's live/editor tooling does not cover.

Scope (v1, deliberately narrow): *direct* GUID references in text-serialized Unity YAML. It does
NOT resolve prefab-variant inheritance transitively, does not resolve UnityEvent method overloads,
and cannot see binary-serialized assets. Those are explicitly out of scope for v1.
"""

import os
import re
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# A Unity GUID is 32 lowercase hex characters.
_GUID_PATTERN = r"[0-9a-f]{32}"
_META_GUID_RE = re.compile(rf"^guid:\s*({_GUID_PATTERN})", re.MULTILINE | re.IGNORECASE)
# Object document separator, e.g. "--- !u!114 &1234567890123456789" (classID 114 = MonoBehaviour)
_DOC_SEP_RE = re.compile(r"^--- !u!(\d+) &(\d+)")
# A YAML mapping key at some indentation, optionally a sequence item ("- key:"). Captures indent + key.
_KEY_RE = re.compile(r"^(\s*)(?:- )?([A-Za-z_][A-Za-z0-9_]*):")

# Text-serialized Unity asset file extensions we scan as *reference sources* (compared lowercased).
# ".meta" is intentionally excluded: it declares an asset's own identity/import settings, not
# scene/prefab references, and would otherwise report every asset as "referencing itself".
UNITY_YAML_ASSET_EXTS: frozenset[str] = frozenset(
    {
        ".unity",  # scenes
        ".prefab",  # prefabs
        ".asset",  # ScriptableObjects, settings, sprite atlases, ...
        ".mat",  # materials
        ".controller",  # animator controllers
        ".overridecontroller",  # animator override controllers
        ".anim",  # animation clips
        ".playable",  # playable/timeline
        ".preset",  # presets
        ".mask",  # avatar masks
        ".physicmaterial",  # 3D physic materials
        ".physicsmaterial2d",  # 2D physics materials
        ".spriteatlas",  # sprite atlases
        ".guiskin",  # legacy GUI skins
        ".fontsettings",  # legacy font settings
        ".rendertexture",  # render textures
        ".mixer",  # audio mixers
        ".terrainlayer",  # terrain layers
        ".signal",  # timeline signals
        ".lighting",  # lighting settings
        ".giparams",  # GI parameters
        ".cubemap",  # cubemaps
        ".flare",  # lens flares
        ".shadervariants",  # shader variant collections
        ".brush",  # terrain/polybrush brushes
    }
)

# Directories that never contain authored asset references (Unity/IDE/build caches, VCS).
_SKIP_DIRS: frozenset[str] = frozenset(
    {".git", "Library", "Temp", "Logs", "obj", "Obj", "Bin", "Build", "Builds", ".vs", ".idea", ".gradle", "node_modules"}
)

# Common Unity native class IDs → readable names (best-effort; unknown IDs fall back to "ClassID<n>").
_UNITY_CLASS_NAMES: dict[int, str] = {
    1: "GameObject",
    2: "Component",
    4: "Transform",
    20: "Camera",
    21: "Material",
    23: "MeshRenderer",
    25: "Renderer",
    28: "Texture2D",
    33: "MeshFilter",
    43: "Mesh",
    48: "Shader",
    49: "TextAsset",
    54: "Rigidbody",
    58: "Collider",
    64: "MeshCollider",
    65: "BoxCollider",
    74: "AnimationClip",
    81: "AudioListener",
    82: "AudioSource",
    83: "AudioClip",
    90: "Avatar",
    91: "AnimatorController",
    95: "Animator",
    102: "TextMesh",
    108: "Light",
    114: "MonoBehaviour",
    115: "MonoScript",
    120: "LineRenderer",
    128: "Font",
    135: "SphereCollider",
    136: "CapsuleCollider",
    137: "SkinnedMeshRenderer",
    143: "CharacterController",
    198: "ParticleSystem",
    199: "ParticleSystemRenderer",
    212: "SpriteRenderer",
    213: "Sprite",
    222: "CanvasRenderer",
    223: "Canvas",
    224: "RectTransform",
    225: "CanvasGroup",
    320: "PlayableDirector",
    328: "VideoPlayer",
    850595691: "LightingSettings",
    1001: "PrefabInstance",
    1660057539: "SceneRoots",
}

# The well-known fileID that identifies a MonoScript reference (i.e. a script *attachment*).
MONO_SCRIPT_FILE_ID = 11500000

SerializationMode = str  # "ForceText" | "ForceBinary" | "Mixed"
_SERIALIZATION_MODES: dict[int, SerializationMode] = {0: "Mixed", 1: "ForceBinary", 2: "ForceText"}


@dataclass
class UnityAssetReference:
    """A single occurrence of a target GUID inside a Unity YAML asset file."""

    referencing_file: str
    """Project-relative path (forward slashes) of the asset that contains the reference."""
    line: int
    """1-based line number of the occurrence."""
    property: str | None
    """Best-effort YAML property the reference belongs to (e.g. ``m_Script``, ``m_Materials``)."""
    reference_kind: str
    """``"script"`` if this is a MonoBehaviour/MonoScript attachment (``m_Script``), else ``"reference"``."""
    object_class_id: int | None
    """Unity native class ID of the enclosing serialized object (from the ``--- !u!<id>`` header)."""
    object_class_name: str | None
    """Readable name for ``object_class_id`` (best-effort)."""
    object_file_id: int | None
    """The enclosing object's local fileID (the ``&<id>`` anchor)."""


class UnityAssetIndex:
    """
    Static reader/scanner for a Unity project's on-disk asset graph.

    All methods operate purely on files under ``project_root``; nothing here launches Unity or a
    language server.
    """

    def __init__(self, project_root: str | os.PathLike[str]):
        self.project_root = Path(project_root)

    # --- project detection / metadata -------------------------------------------------------

    def is_unity_project(self) -> bool:
        """:return: whether ``project_root`` looks like a Unity project."""
        return (self.project_root / "ProjectSettings" / "ProjectVersion.txt").is_file() or (self.project_root / "Assets").is_dir()

    def get_serialization_mode(self) -> SerializationMode | None:
        """
        :return: the project's asset serialization mode (``"ForceText"``/``"ForceBinary"``/``"Mixed"``),
            or ``None`` if it cannot be determined. Only ``"ForceText"`` guarantees that all asset
            references are visible to static scanning.
        """
        settings = self.project_root / "ProjectSettings" / "EditorSettings.asset"
        if not settings.is_file():
            return None
        try:
            text = settings.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        m = re.search(r"m_SerializationMode:\s*(\d+)", text)
        if m is None:
            return None
        return _SERIALIZATION_MODES.get(int(m.group(1)))

    def guid_for_asset(self, relative_path: str | os.PathLike[str]) -> str | None:
        """
        Resolves the GUID of an asset (script, prefab, ...) from its adjacent ``.meta`` file.

        :param relative_path: path to the asset, relative to ``project_root`` (absolute also accepted).
        :return: the 32-char lowercase GUID, or ``None`` if no ``.meta`` file exists / has no GUID.
        """
        p = Path(relative_path)
        abs_asset = p if p.is_absolute() else self.project_root / p
        meta = Path(str(abs_asset) + ".meta")
        if not meta.is_file():
            return None
        try:
            text = meta.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        m = _META_GUID_RE.search(text)
        return m.group(1).lower() if m else None

    # --- scanning ---------------------------------------------------------------------------

    def iter_asset_files(
        self,
        scope_relative_path: str = "",
        is_ignored: Callable[[str], bool] | None = None,
    ) -> Iterator[tuple[str, str]]:
        """
        Yields ``(absolute_path, project_relative_path)`` for every text-serialized Unity YAML asset
        under the given scope, skipping cache/VCS directories.

        :param scope_relative_path: optional subtree (relative to ``project_root``) to limit the walk.
        :param is_ignored: optional predicate on the project-relative path; matching files are skipped
            (e.g. to honor ``.gitignore``).
        """
        root = self.project_root / scope_relative_path if scope_relative_path else self.project_root
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() not in UNITY_YAML_ASSET_EXTS:
                    continue
                abs_path = os.path.join(dirpath, fn)
                rel = os.path.relpath(abs_path, self.project_root).replace(os.sep, "/")
                if is_ignored is not None and is_ignored(rel):
                    continue
                yield abs_path, rel

    def find_references_to_guid(
        self,
        guid: str,
        scope_relative_path: str = "",
        is_ignored: Callable[[str], bool] | None = None,
        exclude_relpaths: tuple[str, ...] = (),
        max_file_bytes: int = 64 * 1024 * 1024,
    ) -> list[UnityAssetReference]:
        """
        Finds every direct occurrence of ``guid`` across text-serialized Unity YAML assets.

        Uses a cheap byte-level pre-filter (no decoding) so that the vast majority of files, which do
        not contain the GUID, are rejected without parsing; only matching files are parsed line by line.

        :param guid: the 32-char GUID to search for (case-insensitive).
        :param scope_relative_path: optional subtree to limit the search.
        :param is_ignored: optional predicate to skip files (e.g. gitignore).
        :param exclude_relpaths: project-relative paths to omit from results.
        :param max_file_bytes: files larger than this are skipped (defensive against pathological inputs).
        :return: references sorted by (file, line).
        """
        guid = guid.lower()
        guid_bytes = guid.encode("ascii")
        exclude = set(exclude_relpaths)
        candidates = [(a, r) for a, r in self.iter_asset_files(scope_relative_path, is_ignored) if r not in exclude]

        def read_if_match(item: tuple[str, str]) -> tuple[str, str] | None:
            abs_path, rel = item
            try:
                if os.path.getsize(abs_path) > max_file_bytes:
                    return None
                with open(abs_path, "rb") as fh:
                    data = fh.read()
            except OSError:
                return None
            # Case-sensitive pre-filter: Unity always serializes GUIDs as canonical lowercase hex,
            # so this avoids allocating a lowercased copy of every file just to reject non-matches.
            if guid_bytes not in data:
                return None
            return rel, data.decode("utf-8", errors="replace")

        # The scan is I/O-bound (read + memchr per file); file reads release the GIL, so a thread
        # pool over the many non-matching files is a large win on SSDs. Only matching files are parsed.
        workers = min(32, (os.cpu_count() or 4) * 4)
        results: list[UnityAssetReference] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for match in pool.map(read_if_match, candidates):
                if match is not None:
                    rel, text = match
                    results.extend(self._scan_text_for_guid(rel, text, guid))
        results.sort(key=lambda r: (r.referencing_file, r.line))
        return results

    def find_references_to_asset(
        self,
        relative_path: str | os.PathLike[str],
        **kwargs: object,
    ) -> list[UnityAssetReference]:
        """
        Convenience wrapper: resolves ``relative_path`` to its GUID and finds references to it, while
        excluding the asset itself from the results. Returns an empty list if the GUID cannot be resolved.
        """
        guid = self.guid_for_asset(relative_path)
        if guid is None:
            return []
        rel = str(relative_path).replace(os.sep, "/") if not Path(relative_path).is_absolute() else ""
        exclude = (rel,) if rel else ()
        return self.find_references_to_guid(guid, exclude_relpaths=exclude, **kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _scan_text_for_guid(rel: str, text: str, guid: str) -> list[UnityAssetReference]:
        """Parses an already-matched file, attributing each GUID occurrence to its enclosing object and property."""
        out: list[UnityAssetReference] = []
        cur_class_id: int | None = None
        cur_file_id: int | None = None
        key_stack: list[tuple[int, str]] = []  # (indent, key) of enclosing mapping keys
        for i, line in enumerate(text.splitlines(), start=1):
            sep = _DOC_SEP_RE.match(line)
            if sep is not None:
                cur_class_id = int(sep.group(1))
                cur_file_id = int(sep.group(2))
                key_stack.clear()
                continue

            key_match = _KEY_RE.match(line)
            if key_match is not None:
                indent = len(key_match.group(1))
                while key_stack and key_stack[-1][0] >= indent:
                    key_stack.pop()
                key_stack.append((indent, key_match.group(2)))

            if guid in line.lower():
                if key_match is not None:
                    prop: str | None = key_match.group(2)
                elif key_stack:
                    prop = key_stack[-1][1]
                else:
                    prop = None
                out.append(
                    UnityAssetReference(
                        referencing_file=rel,
                        line=i,
                        property=prop,
                        reference_kind="script" if prop == "m_Script" else "reference",
                        object_class_id=cur_class_id,
                        object_class_name=(
                            _UNITY_CLASS_NAMES.get(cur_class_id, f"ClassID{cur_class_id}") if cur_class_id is not None else None
                        ),
                        object_file_id=cur_file_id,
                    )
                )
        return out
