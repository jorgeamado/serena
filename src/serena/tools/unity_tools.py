"""
Unity-specific tools that operate statically on the asset graph (no running Unity Editor, no
language server). See :mod:`serena.util.unity_assets` for the underlying index.
"""

import os
import re
from dataclasses import asdict

from serena.tools import Tool, ToolMarkerBeta
from serena.util.unity_assets import UnityAssetIndex


class UnityAssetReferencesTool(Tool, ToolMarkerBeta):
    """
    Finds which Unity scenes, prefabs, and other assets reference a given script or asset, by
    statically scanning text-serialized Unity YAML files on disk. Requires no running Unity Editor
    and no language server; answers questions a language server cannot (e.g. which scenes/prefabs
    attach a MonoBehaviour, or which scenes instantiate a prefab).
    """

    def apply(self, relative_path: str = "", guid: str = "", scope_relative_path: str = "", max_answer_chars: int = -1) -> str:
        """
        Finds references to a Unity asset (reverse reference search) across the project's
        text-serialized Unity YAML assets (`.unity` scenes, `.prefab`, `.asset`, `.mat`,
        `.controller`, `.anim`, and similar). The search is purely static (files on disk) and does
        NOT require a running Unity Editor, a license, or any Unity package.

        Typical uses:
        - Pass a C# MonoBehaviour script path (e.g. "Assets/Scripts/Player.cs") to find every scene
          and prefab that attaches it (reported with reference_kind "script").
        - Pass a prefab path (e.g. "Assets/Prefabs/Enemy.prefab") to find every scene/prefab that
          references or instantiates it.

        The target's GUID is read from its adjacent ".meta" file. Note: this finds only *direct*
        references in text-serialized assets; it does not resolve prefab-variant inheritance
        transitively, does not resolve UnityEvent method overloads, and cannot see binary-serialized
        assets (see the returned "serialization_mode" and "notes").

        :param relative_path: path (relative to the project root) of the script/asset whose usages to
            find. Provide this OR `guid`.
        :param guid: alternatively, the 32-character Unity GUID to search for directly. Ignored if
            `relative_path` is given.
        :param scope_relative_path: optional subdirectory (relative to the project root, e.g.
            "Assets/UI") to limit the search to, for speed on very large projects. Empty = whole project.
        :param max_answer_chars: if the JSON result exceeds this length, a shortened per-file summary
            is returned instead. -1 uses the configured default.
        :return: a JSON object with the resolved target, the project's asset serialization mode, any
            notes/caveats, the reference count, and the list of references (each with referencing_file,
            line, enclosing object class/fileID, property, and reference_kind).
        """
        index = UnityAssetIndex(self.get_project_root())
        if not index.is_unity_project():
            return self._to_json(
                {
                    "error": "The active project does not look like a Unity project "
                    "(no ProjectSettings/ProjectVersion.txt and no Assets/ directory)."
                }
            )

        # Resolve the target GUID from either an asset path or a raw GUID.
        if relative_path:
            resolved = index.guid_for_asset(relative_path)
            if resolved is None:
                return self._to_json(
                    {"error": f"Could not resolve a GUID for '{relative_path}'. Expected a '{relative_path}.meta' file next to it."}
                )
            target_guid = resolved
            target = {"relative_path": relative_path.replace(os.sep, "/"), "guid": resolved}
        elif guid:
            g = guid.strip().lower()
            if re.fullmatch(r"[0-9a-f]{32}", g) is None:
                return self._to_json({"error": f"'{guid}' is not a valid 32-character Unity GUID."})
            target_guid = g
            target = {"guid": g}
        else:
            return self._to_json({"error": "Provide either `relative_path` or `guid`."})

        # Serialization-mode guard: static scanning only sees text-serialized references.
        notes: list[str] = []
        mode = index.get_serialization_mode()
        if mode == "ForceBinary":
            notes.append(
                "Project asset serialization mode is 'Force Binary'; references are not stored as text "
                "and cannot be found by static scanning. Switch to 'Force Text' "
                "(Edit > Project Settings > Editor > Asset Serialization) to use this tool."
            )
        elif mode == "Mixed":
            notes.append("Project asset serialization mode is 'Mixed'; some binary-serialized assets may be invisible to static scanning.")
        elif mode is None:
            notes.append("Could not determine the project's asset serialization mode; results assume text serialization.")

        exclude = (relative_path.replace(os.sep, "/"),) if relative_path else ()
        is_ignored = lambda rel: self.project.is_ignored_path(rel, ignore_non_source_files=False)
        refs = index.find_references_to_guid(
            target_guid, scope_relative_path=scope_relative_path, is_ignored=is_ignored, exclude_relpaths=exclude
        )

        result: dict = {
            "target": target,
            "serialization_mode": mode,
            "reference_count": len(refs),
            "references": [asdict(r) for r in refs],
        }
        if notes:
            result["notes"] = notes

        def per_file_summary() -> str:
            counts: dict[str, int] = {}
            for r in refs:
                counts[r.referencing_file] = counts.get(r.referencing_file, 0) + 1
            summary = {
                "target": target,
                "serialization_mode": mode,
                "reference_count": len(refs),
                "references_by_file": counts,
                "notes": [*notes, "Full per-reference details omitted due to size; showing counts per file."],
            }
            return self._to_json(summary)

        return self._limit_length(self._to_json(result), max_answer_chars, [per_file_summary])
