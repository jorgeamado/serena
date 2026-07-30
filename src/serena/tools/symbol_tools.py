"""
Language server-related tools
"""

import copy
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any, cast

from serena.symbol import (
    LanguageServerSymbol,
    LanguageServerSymbolDictGrouper,
    LanguageServerSymbolLocation,
    ReferenceInLanguageServerSymbol,
)
from serena.tools import (
    SUCCESS_RESULT,
    EditingToolWithDiagnostics,
    Tool,
    ToolMarkerSymbolicEdit,
    ToolMarkerSymbolicRead,
)
from serena.tools.tools_base import ToolMarkerBeta, ToolMarkerOptional
from serena.util.ls_diagnostics import GroupedDiagnostics
from serena.util.text_utils import find_text_coordinates
from serena.util.value_access import classify_member_access
from solidlsp import ls_types
from solidlsp.ls_types import SymbolKind


class RestartLanguageServerTool(Tool, ToolMarkerOptional):
    """Restarts the language server(s)."""

    def apply(self) -> str:
        """Use this tool only on explicit user request or after confirmation.
        It may be necessary to restart the language server if it hangs.
        """
        self.agent.reset_language_server_manager()
        return SUCCESS_RESULT


class GetSymbolsOverviewTool(Tool, ToolMarkerSymbolicRead):
    """
    Gets an overview of the top-level symbols defined in a given file.
    """

    symbol_dict_grouper = LanguageServerSymbolDictGrouper(["kind"], ["kind"], collapse_singleton=True)

    def apply(self, relative_path: str, depth: int = -1, max_answer_chars: int = -1) -> str:
        """
        Use this tool to get a high-level understanding of the code symbols in a file.
        This should be the first tool to call when you want to understand a new file, unless you already know
        what you are looking for.

        :param relative_path: the relative path to the file to get the overview of
        :param depth: depth up to which descendants shall be retrieved.
            Default (-1) results in a language specific choice: 1 for java and kotlin and 0 for other languages
        :param max_answer_chars: if the overview is longer than this number of characters,
            no content will be returned. -1 means the default value from the config will be used.
            Don't adjust unless there is really no other way to get the content required for the task.
        :return: a JSON object containing symbols grouped by kind in a compact format.
        """
        # Note: file system sync not required (relevant file is opened in the language server explicitly)

        if depth == -1:
            if relative_path.endswith((".java", ".kt")):
                depth = 1
            else:
                depth = 0

        result = self.get_symbol_overview(relative_path, depth=depth)

        # capture kind names and depth-0 snapshots before grouping, which mutates the dicts
        kind_names = [d.get("kind", "unknown") for d in result]
        if depth > 0:
            depth_0_result = [d.copy() for d in result]
            for d in depth_0_result:
                d.pop("children", None)

        compact_result = self.symbol_dict_grouper.group(result)
        result_json_str = self._to_json(compact_result)

        # shortened result closures
        def make_kind_counts() -> str:
            return f"Symbol counts by kind:\n{self._to_json(Counter(kind_names))}"

        if depth == 0:
            shortened_results = [make_kind_counts]
        else:

            def make_depth_0_result() -> str:
                compact_depth_0_result = self.symbol_dict_grouper.group(depth_0_result)
                return "Depth 0 overview:\n" + self._to_json(compact_depth_0_result)

            shortened_results = [make_depth_0_result, make_kind_counts]

        return self._limit_length(result_json_str, max_answer_chars, shortened_result_factories=shortened_results)

    def get_symbol_overview(self, relative_path: str, depth: int = 0) -> list[LanguageServerSymbol.OutputDict]:
        """
        :param relative_path: relative path to a source file
        :param depth: the depth up to which descendants shall be retrieved
        :return: a list of symbol dictionaries representing the symbol overview of the file
        """
        symbol_retriever = self.create_language_server_symbol_retriever()

        # The symbol overview is capable of working with both files and directories,
        # but we want to ensure that the user provides a file path.
        file_path = os.path.join(self.project.project_root, relative_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File or directory {relative_path} does not exist in the project.")
        if os.path.isdir(file_path):
            raise ValueError(f"Expected a file path, but got a directory path: {relative_path}. ")
        if not symbol_retriever.can_analyze_file(relative_path):
            raise ValueError(
                f"Cannot extract symbols from file {relative_path}. Active language servers: {[l.value for l in self.agent.get_active_language_server_ids()]}"
            )

        symbols = symbol_retriever.get_symbol_overview(relative_path)[relative_path]

        def child_inclusion_predicate(s: LanguageServerSymbol) -> bool:
            return not s.is_low_level()

        symbol_dicts = []
        for symbol in symbols:
            symbol_dicts.append(
                symbol.to_dict(
                    name_path=False,
                    name=True,
                    depth=depth,
                    kind=True,
                    relative_path=False,
                    location=False,
                    child_inclusion_predicate=child_inclusion_predicate,
                )
            )
        return symbol_dicts


class FindSymbolTool(Tool, ToolMarkerSymbolicRead):
    """
    Performs a global (or local) search using the language server backend.
    """

    # group children by kind, keeping just the name (the parent's name_path makes it unambiguous);
    # we don't group the top-level result list because many tests rely on it being a flat list of symbol dicts
    symbol_dict_grouper = LanguageServerSymbolDictGrouper([], ["kind"], collapse_singleton=True)

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path_pattern: str,
        depth: int = 0,
        relative_path: str = "",
        include_body: bool = False,
        include_info: bool = False,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        substring_matching: bool = False,
        max_matches: int = -1,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds symbols or other code entities (classes, methods, etc.) based on the given name path pattern.
        The returned symbol information can be used for edits or further queries.
        Specify `depth > 0` to also retrieve children/descendants (e.g., methods of a class).

        A name path is a path in the symbol tree *within a source file*.
        For example, the method `my_method` defined in class `MyClass` would have the name path `MyClass/my_method`.
        If a symbol is overloaded (e.g., in Java), a 0-based index is appended (e.g. "MyClass/my_method[0]") to
        uniquely identify it.

        To search for a symbol, you provide a name path pattern that is used to match against name paths.
        It can be
         * a simple name (e.g. "method"), which will match any symbol with that name
         * a relative path like "class/method", which will match any symbol with that name path suffix
         * an absolute name path "/class/method" (absolute name path), which requires an exact match of the full name path within the source file.
        Append an index `[i]` to match a specific overload only, e.g. "MyClass/my_method[1]".

        :param name_path_pattern: the name path matching pattern (see above)
        :param depth: depth up to which descendants shall be retrieved (e.g. use 1 to also retrieve immediate children;
            for the case where the symbol is a class, this will return its methods).
            Ignored if `include_body=True`. Default 0.
        :param relative_path: (optional) restrict search to this file or directory. If None, searches entire codebase.
            If a directory is passed, the search will be restricted to the files in that directory.
            If a file is passed, the search will be restricted to that file.
        :param include_body: whether to include the symbol's source code. Use judiciously.
        :param include_info: whether to include additional info (hover-like, typically including docstring and signature),
            about the symbol (ignored if include_body is True). Info is never included for child symbols.
            Note: Depending on the language, this can be slow (e.g., C/C++).
        :param include_kinds: (optional) limits results to the given LSP symbol kinds (integers)
        :param exclude_kinds: (optional) list of LSP symbol kinds (integers) to exclude.
        :param substring_matching: If True, use substring matching for the last element of the pattern, such that
            "Foo/get" would match "Foo/getValue" and "Foo/getData".
        :param max_matches: maximum number of permitted matches. If exceeded, a shortened result is returned
             which allows refining the search. -1 (default) means no limit. Set to 1 if you search for a single symbol.
        :param max_answer_chars: max result length; -1 for default
        :return: symbols (with locations) matching the name.
        """
        # Note: file system sync not required; the symbol finder opens all relevant source files explicitly in the case of changes

        if include_body:
            depth = 0  # ignore user-specified depth if include_body is True
        assert max_matches != 0, "max_matches must be > 0 or equal to -1."
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()
        symbols = symbol_retriever.find(
            name_path_pattern,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
            substring_matching=substring_matching,
            within_relative_path=relative_path,
        )
        n_matches = len(symbols)

        def create_short_result_relative_path_to_name_paths() -> str:
            relative_path_to_name_paths: defaultdict[str, list[str]] = defaultdict(list)
            for s in symbols:
                relative_path_to_name_paths[s.location.relative_path or "unknown"].append(s.get_name_path())
            return f"Shortened result:\n{self._to_json(relative_path_to_name_paths)}"

        if 0 < max_matches < n_matches:
            return f"Matched {n_matches}>{max_matches=} symbols.\n" + create_short_result_relative_path_to_name_paths()

        symbol_dicts = [
            s.to_dict(
                kind=True,
                name_path=True,
                name=False,
                relative_path=True,
                body_location=True,
                depth=depth,
                body=include_body,
                children_name=True,
                children_name_path=False,
            )
            for s in symbols
        ]
        if not include_body and include_info:
            info_by_symbol = symbol_retriever.request_info_for_symbol_batch(symbols)
            for s, s_dict in zip(symbols, symbol_dicts, strict=True):
                if symbol_info := info_by_symbol.get(s):
                    # In python 3.15 we could specify extra_items=True in the TypedDict definition,
                    # https://peps.python.org/pep-0728/
                    # If we ever upgrade to 3.15, we can remove the type: ignore[typeddict-unknown-key]
                    s_dict["info"] = symbol_info

        grouped_symbol_dicts = self.symbol_dict_grouper.group(symbol_dicts)
        result = self._to_json(grouped_symbol_dicts)
        return self._limit_length(result, max_answer_chars, shortened_result_factories=[create_short_result_relative_path_to_name_paths])

    @classmethod
    def get_param_aliases(cls) -> dict[str, str]:
        return {"name_path": "name_path_pattern"}


class FindReferencingSymbolsTool(Tool, ToolMarkerSymbolicRead):
    """
    Finds symbols that reference the given symbol using the language server backend
    """

    symbol_dict_grouper = LanguageServerSymbolDictGrouper(["relative_path", "kind"], ["kind"], collapse_singleton=True)

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path: str,
        relative_path: str,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds references to the symbol at the given `name_path`. The result will contain metadata about the referencing symbols
        as well as a short code snippet around the reference.

        :param name_path: name path of the symbol
        :param relative_path: the relative path to the file containing the symbol for which to find references.
        :param include_kinds: (optional) limits results to the given LSP symbol kinds (integers)
        :param exclude_kinds: optional list of LSP symbol kinds (integers) to exclude.
        :param max_answer_chars: max result length; -1 for default
        :return: a list of JSON objects with the symbols referencing the requested symbol
        """
        # file system sync needed for case where symbol finder does not perform a global search, updating everything
        if relative_path:
            self.project.ls_sync_file_system_changes()

        include_body = False  # It is probably never a good idea to include the body of the referencing symbols
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None

        symbol_retriever = self.create_language_server_symbol_retriever()
        references_in_symbols = symbol_retriever.find_referencing_symbols(
            name_path,
            relative_file_path=relative_path,
            include_body=include_body,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
        )

        reference_dicts = []
        for ref in references_in_symbols:
            ref_dict_orig = ref.symbol.to_dict(kind=True, relative_path=True, depth=0, body=include_body, body_location=True)
            ref_dict = dict(ref_dict_orig)
            if not include_body:
                ref_relative_path = ref.symbol.location.relative_path
                assert ref_relative_path is not None, f"Referencing symbol {ref.symbol.name} has no relative path, this is likely a bug."
                content_around_ref = self.project.retrieve_content_around_line(
                    relative_file_path=ref_relative_path, line=ref.line, context_lines_before=1, context_lines_after=1
                )
                ref_dict["content_around_reference"] = content_around_ref.to_display_string()
            reference_dicts.append(ref_dict)

        # capture lightweight reference data before grouping
        ref_summaries = []
        for ref, d in zip(references_in_symbols, reference_dicts, strict=True):
            ref_summaries.append(
                {
                    "name_path": d.get("name_path"),
                    "kind": d.get("kind"),
                    "relative_path": d.get("relative_path"),
                    "reference_line": ref.line,
                }
            )

        result = self.symbol_dict_grouper.group(reference_dicts)

        # shortened result closures, from least to most aggressive shortening
        def make_refs_without_context() -> str:
            """References with name_path and reference line, without surrounding code lines"""
            grouped = self.symbol_dict_grouper.group(copy.deepcopy(ref_summaries))
            return f"References without surrounding lines:\n{self._to_json(grouped)}"

        def make_per_file_counts() -> str:
            counts = Counter(str(r["relative_path"]) for r in ref_summaries)
            return f"Reference counts per file:\n{self._to_json(counts)}"

        def make_summary() -> str:
            return f"Found {len(ref_summaries)} references."

        shortened_results = [make_refs_without_context, make_per_file_counts, make_summary]

        result_json = self._to_json(result)
        return self._limit_length(result_json, max_answer_chars, shortened_result_factories=shortened_results)


class FindImplementationsTool(Tool, ToolMarkerSymbolicRead):
    """
    Finds symbols that implement the given symbol using the language server backend.
    """

    # noinspection PyDefaultArgument
    def apply(
        self,
        name_path: str,
        relative_path: str,
        include_info: bool = False,
        include_kinds: list[int] = [],  # noqa: B006
        exclude_kinds: list[int] = [],  # noqa: B006
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds implementations of the symbol at the given `name_path`.

        :param name_path: the symbol's name path
        :param relative_path: the relative path to the file containing the symbol for which to find implementations.
            Note that here you can't pass a directory but must pass a file.
        :param include_info: whether to include additional info (hover-like, typically including docstring and signature),
            about the implementing symbols.
        :param include_kinds: (optional) limits results to the given LSP symbol kinds (integers)
        :param exclude_kinds: (optional) list of LSP symbol kinds (integers) to exclude.
        :param max_answer_chars: max result length; -1 for default
        :return: a list of JSON objects with the symbols implementing the requested symbol
        """
        self.project.ls_sync_file_system_changes()

        include_body = False
        parsed_include_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in include_kinds] if include_kinds else None
        parsed_exclude_kinds: Sequence[SymbolKind] | None = [SymbolKind(k) for k in exclude_kinds] if exclude_kinds else None
        symbol_retriever = self.create_language_server_symbol_retriever()

        implementing_symbols = symbol_retriever.find_implementing_symbols(
            name_path,
            relative_file_path=relative_path,
            include_body=include_body,
            include_kinds=parsed_include_kinds,
            exclude_kinds=parsed_exclude_kinds,
        )

        symbol_dicts = [
            dict(s.to_dict(kind=True, relative_path=True, depth=0, body=include_body, body_location=True)) for s in implementing_symbols
        ]
        if include_info:
            info_by_symbol = symbol_retriever.request_info_for_symbol_batch(implementing_symbols)
            for s, s_dict in zip(implementing_symbols, symbol_dicts, strict=True):
                if symbol_info := info_by_symbol.get(s):
                    s_dict["info"] = symbol_info
                    s_dict.pop("name", None)  # name is included in the info

        result = self._to_json(symbol_dicts)
        return self._limit_length(result, max_answer_chars)


class FindDeclarationTool(Tool, ToolMarkerSymbolicRead):
    """
    Finds the declaration/definition of a symbol
    """

    def apply(
        self,
        relative_path: str,
        regex: str,
        containing_symbol_name_path: str | None = None,
        include_body: bool = False,
        include_info: bool = False,
    ) -> str:
        r"""
        Finds the declaration of a symbol.

        :param relative_path: the relative path to the source file containing the symbol for which to find the declaration.
        :param regex: a regular expression with one group, where the group matches the symbol for which to perform the lookup.
            For example, to find the declaration of the `process` method in a call like `obj.process()`,
            pass an expression like "obj\.(process)\(process_input_arg=37\)".
            Prefer regexes with sufficiently large context around the group to render the match unambiguous.
            Uses Python syntax with MULTILINE and DOTALL flags enabled.
        :param containing_symbol_name_path: optional name path of a containing symbol whose body shall be searched instead of the full file.
        :param include_body: whether to include the symbol's body in the result. Default False.
        :param include_info: whether to include additional info (hover-like). Default False.
        """
        self.project.ls_sync_file_system_changes()

        symbol_retriever = self.create_language_server_symbol_retriever()
        relative_path = self._sanitize_input_param(relative_path)
        regex = self._sanitize_input_param(regex)

        # find relevant location for lookup
        editor = self.create_code_editor()
        if not containing_symbol_name_path:
            content = editor.read_file(relative_path)
            coords = find_text_coordinates(content, regex, require_unique=True)
            assert coords is not None
        else:
            symbol = symbol_retriever.find_unique(name_path_pattern=containing_symbol_name_path, within_relative_path=relative_path)
            body_line_numers = symbol.get_body_line_numbers_or_raise()
            content = editor.read_file(relative_path, lines=body_line_numers)
            coords = find_text_coordinates(content, regex, require_unique=True)
            assert coords is not None
            coords.line += body_line_numers[0]

        # retrieve declaration
        defining_symbol = symbol_retriever.find_declaration(
            relative_file_path=relative_path,
            line=coords.line,
            column=coords.col,
            include_body=include_body,
        )
        if defining_symbol is None:
            raise ValueError(
                f"No symbol declaration found at the location of the regex match. Location: {relative_path}:{coords.line}:{coords.col}."
            )

        # create output
        symbol_dict = self._defining_symbol_to_result_dict(
            symbol_retriever,
            defining_symbol,
            include_body,
            include_info,
        )
        result = self._to_json(symbol_dict)
        return result

    @staticmethod
    def _defining_symbol_to_result_dict(
        symbol_retriever: Any,
        defining_symbol: LanguageServerSymbol,
        include_body: bool,
        include_info: bool,
    ) -> dict[str, Any]:
        symbol_dict = dict(defining_symbol.to_dict(kind=True, relative_path=True, depth=0, body=include_body, body_location=True))
        if not include_body and include_info:
            if symbol_info := symbol_retriever.request_info_for_symbol(defining_symbol):
                symbol_dict["info"] = symbol_info
                symbol_dict.pop("name", None)
        return symbol_dict


class GetDiagnosticsForFileTool(Tool, ToolMarkerSymbolicRead):
    """
    Gets diagnostics for a file, optionally restricted to a line range, grouped by file, severity, and containing symbol.
    """

    FILE_LEVEL_DIAGNOSTIC_BUCKET = "<file>"

    def apply(
        self,
        relative_path: str,
        start_line: int = 0,
        end_line: int = -1,
        min_severity: int = 4,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Gets diagnostics for a file. Diagnostics are grouped as `relative_path -> severity -> name_path -> diagnostics_results`.
        If a diagnostic cannot be mapped to a symbol, it is grouped under the special name path `<file>`.

        :param relative_path: the relative path to the file to inspect.
        :param start_line: the first 0-based line to include. Defaults to 0.
        :param end_line: the last 0-based line to include. Defaults to -1, which means until the end of the file.
        :param min_severity: minimum LSP severity to include, where 1=Error, 2=Warning, 3=Information, 4=Hint.
            Diagnostics with lower-or-equal numeric severity are returned.
        :param max_answer_chars: max result length; -1 for default
        :return: grouped diagnostics for the requested file.
        """
        self.project.ls_sync_file_system_changes()

        symbol_retriever = self.create_language_server_symbol_retriever()
        diagnostics = symbol_retriever.get_file_diagnostics(
            relative_file_path=relative_path,
            start_line=start_line,
            end_line=end_line,
            min_severity=min_severity,
        )

        grouped_diagnostics = GroupedDiagnostics()
        for diagnostic in diagnostics:
            diag_range = diagnostic["range"]["start"]
            name_path = self.FILE_LEVEL_DIAGNOSTIC_BUCKET
            owner_symbol = symbol_retriever.find_diagnostic_owner_symbol(
                relative_file_path=relative_path,
                line=diag_range["line"],
                column=diag_range["character"],
            )
            if owner_symbol is not None:
                name_path = owner_symbol.get_name_path()
            grouped_diagnostics.add(relative_path, name_path, diagnostic)

        result = self._to_json(grouped_diagnostics.get_dict())
        return self._limit_length(result, max_answer_chars)


class GetDiagnosticsForSymbolTool(Tool, ToolMarkerSymbolicRead, ToolMarkerOptional):
    """
    Gets diagnostics for a symbol and, optionally, for symbols that reference it.
    """

    def apply(
        self,
        name_path: str,
        reference_file: str = "",
        check_symbol_references: bool = False,
        min_severity: int = 4,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Gets diagnostics for the specified symbol. When `check_symbol_references` is true, diagnostics for all
        referencing symbols are also included. The result is grouped as
        `relative_path -> severity -> name_path -> diagnostics_results`.

        :param name_path: the name path of the symbol to inspect.
        :param reference_file: optional file path used to disambiguate the symbol search.
        :param check_symbol_references: whether to additionally collect diagnostics for symbols that reference the symbol.
        :param min_severity: minimum LSP severity to include, where 1=Error, 2=Warning, 3=Information, 4=Hint.
            Diagnostics with lower-or-equal numeric severity are returned.
        :param max_answer_chars: max result length; -1 for default
        :return: grouped diagnostics for the requested symbol and, optionally, its referencing symbols.
        """
        self.project.ls_sync_file_system_changes()

        symbol_retriever = self.create_language_server_symbol_retriever()
        diagnostics_by_symbol = symbol_retriever.get_symbol_diagnostics(
            name_path=name_path,
            reference_file=reference_file or None,
            check_symbol_references=check_symbol_references,
            min_severity=min_severity,
        )

        grouped_diagnostics = GroupedDiagnostics()
        for symbol, diagnostics in diagnostics_by_symbol.items():
            relative_path = symbol.relative_path
            if relative_path is None:
                continue
            symbol_name_path = symbol.get_name_path()
            for diagnostic in diagnostics:
                grouped_diagnostics.add(relative_path, symbol_name_path, diagnostic)

        result = self._to_json(grouped_diagnostics.get_dict())
        return self._limit_length(result, max_answer_chars)


class ReplaceSymbolBodyTool(EditingToolWithDiagnostics):
    """
    Replaces the full definition of a symbol using the language server backend.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        r"""
        Replaces the body of the given symbol.

        IMPORTANT: Only replace symbol bodies if you have previously made a retrieval with include_body=True and thus know what
        constitutes the body!

        :param name_path: name path of the symbol whose body to replace
        :param relative_path: the relative path to the file containing the symbol
        :param body: the new symbol body. The symbol body is the definition of a symbol
            in the programming language, including e.g. the signature line for functions.
            Depending on the language, it may or may not include a preceding docstring or other preceding annotations.
        """
        with self.DiagnosticsContext(self, relative_path) as diagnostics_context:
            code_editor = self.create_code_editor()
            code_editor.replace_body(
                name_path,
                relative_file_path=relative_path,
                body=body,
            )
            return diagnostics_context.format_result(SUCCESS_RESULT)


class InsertAfterSymbolTool(EditingToolWithDiagnostics):
    """
    Inserts content after the end of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Use this to insert code after a class/method/function definition.
        Don't use to insert after assignments (constants, fields).

        :param name_path: name path of the symbol after which to insert content
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted. The inserted code shall begin with the next line after
            the symbol.
        """
        with self.DiagnosticsContext(self, relative_path) as diagnostics_context:
            code_editor = self.create_code_editor()
            code_editor.insert_after_symbol(name_path, relative_file_path=relative_path, body=body)
            return diagnostics_context.format_result(SUCCESS_RESULT)


class InsertBeforeSymbolTool(EditingToolWithDiagnostics):
    """
    Inserts content before the beginning of the definition of a given symbol.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        body: str,
    ) -> str:
        """
        Inserts the given content before the beginning of the definition of the given symbol (via the symbol's location).
        A typical use case is to insert a new class, function, method, field or variable assignment; or
        a new import statement before the first symbol in the file.

        :param name_path: name path of the symbol before which to insert content
        :param relative_path: the relative path to the file containing the symbol
        :param body: the body/content to be inserted before the line in which the referenced symbol is defined
        """
        with self.DiagnosticsContext(self, relative_path) as diagnostics_context:
            code_editor = self.create_code_editor()
            code_editor.insert_before_symbol(name_path, relative_file_path=relative_path, body=body)
            return diagnostics_context.format_result(SUCCESS_RESULT)


class RenameSymbolTool(Tool, ToolMarkerSymbolicEdit):
    """
    Renames a symbol throughout the codebase using language server refactoring capabilities.
    For JB, we use a separate tool.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        new_name: str,
    ) -> str:
        """
        Renames the symbol with the given `name_path` to `new_name` throughout the entire codebase.
        Note: for languages with method overloading, like Java, name_path may have to include a method's
        signature to uniquely identify a method.

        :param name_path: name path of the symbol to rename
        :param relative_path: the relative path to the file containing the symbol to rename
        :param new_name: the new name for the symbol
        :return: result summary indicating success or failure
        """
        self.project.ls_sync_file_system_changes()
        code_editor = self.create_ls_code_editor()
        status_message = code_editor.rename_symbol(name_path, relative_path=relative_path, new_name=new_name)
        return status_message


class SafeDeleteSymbol(Tool, ToolMarkerSymbolicEdit):
    def apply(
        self,
        name_path_pattern: str,
        relative_path: str,
    ) -> str:
        """
        Deletes the symbol if it is safe to do so (i.e., if there are no references to it)
        or returns a list of references to it.

        :param name_path_pattern: name path of the symbol to delete
        :param relative_path: the relative path to the file containing the symbol to delete
        """
        self.project.ls_sync_file_system_changes()

        ls_symbol_retriever = self.create_language_server_symbol_retriever()
        symbol = ls_symbol_retriever.find_unique(name_path_pattern, substring_matching=False, within_relative_path=relative_path)
        symbol_rel_path = symbol.relative_path
        assert symbol_rel_path is not None, f"Symbol {name_path_pattern} has no relative path, this is likely a bug."
        assert symbol_rel_path == relative_path, f"Symbol {name_path_pattern} is not in the expected relative path {relative_path}."
        symbol_name_path = symbol.get_name_path()

        symbol_line = symbol.line
        symbol_col = symbol.column
        assert symbol_line is not None and symbol_col is not None, (
            f"Symbol {name_path_pattern} has no identifier position, this is likely a bug."
        )
        lang_server = ls_symbol_retriever.get_language_server(symbol_rel_path)
        references_locations = lang_server.request_references(symbol_rel_path, symbol_line, symbol_col)
        file_to_lines: dict[str, list[int]] = defaultdict(list)
        if references_locations:
            for ref_loc in references_locations:
                ref_relative_path = ref_loc.get("relativePath")
                if ref_relative_path is None:
                    continue
                file_to_lines[ref_relative_path].append(ref_loc["range"]["start"]["line"])
        if file_to_lines:
            return f"Cannot delete, the symbol {symbol_name_path} is referenced in: {self._to_json(file_to_lines)}"
        code_editor = self.create_ls_code_editor()
        code_editor.delete_symbol(symbol_name_path, relative_file_path=symbol_rel_path)
        return SUCCESS_RESULT


# Member (property/field/event) path constants -- SPEC-call-hierarchy-members.md (Rev 2).
# Kinds routed to the member path instead of the callable call-hierarchy/find-references-fallback path.
MEMBER_KINDS = {SymbolKind.Property, SymbolKind.Field, SymbolKind.Event}
# Node budget for member incoming children (SPEC §4.3.5); mirrors CALL_HIERARCHY_MAX_NODES for callables.
MEMBER_MAX_NODES = 200
# Frozen note text (SPEC §7, F2). Member incoming results are always approximate; this note explains why.
MEMBER_APPROXIMATE_NOTE = (
    "incoming usages for this property, field, or event were derived from find-references and "
    "classified by access kind syntactically; results are approximate and may include non-access "
    "usages or 'unknown' where a site could not be classified."
)
# Frozen note text (SPEC §7, F3). Outgoing calls make no sense for a property/field/event.
MEMBER_OUTGOING_UNAVAILABLE = "outgoing calls are not applicable to a property, field, or event"


class CallHierarchyTool(Tool, ToolMarkerSymbolicRead, ToolMarkerBeta):
    """
    Finds callers (incoming) and/or callees (outgoing) of a function or method,
    transitively up to a given depth, using the language server's call hierarchy.

    Also supports properties, fields, and events (SPEC-call-hierarchy-members.md, Rev 2): for these
    "member" kinds, incoming usages are derived from find-references (there is no accessor-level call
    hierarchy) and each usage site is classified into an access kind (read/write/read_write/subscribe/
    unsubscribe/invoke/unknown) by syntactically inspecting the source line. This member path is
    reference-based and single-line syntactic, so its results are always reported as approximate, and
    sites that cannot be classified confidently are "unknown" rather than guessed. Treating a plain
    read as "get" and a plain write as "set" is a heuristic, not an identity: a ref-returning property
    getter can expose storage that callers mutate without ever calling a setter, and such a site would
    still read as a "read" here. Multi-line expressions, comments interleaved in tokens, deconstruction
    targets, indexers, `&addr`/`__makeref`, and delegate copies of events are not modeled. Member mode
    is depth-1 only (transitively expanding a member is not semantically sound: reads/writes of a
    wrapping member do not both execute the inner access), and outgoing calls do not apply to members.
    """

    def apply(
        self,
        name_path: str,
        relative_path: str,
        direction: str = "incoming",
        depth: int = 1,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Finds callers (incoming) and/or callees (outgoing) of a function or method,
        transitively up to a given depth, using the language server's call hierarchy.
        When the language server does not support call hierarchy, incoming calls are automatically derived from find-references,
        which may include non-call usages and may miss callers inside properties, constructors, or one-line functions.

        If `name_path` resolves to a property, field, or event, a separate "member" path is used instead
        (SPEC-call-hierarchy-members.md, Rev 2): incoming usages are derived from find-references and each
        usage site is classified by access kind (read/write/read_write/subscribe/unsubscribe/invoke/unknown)
        via single-line syntactic inspection -- results are always approximate, `depth` is treated as 1, and
        "outgoing" is not applicable (see the class docstring for the full list of caveats).

        :param name_path: name path of the symbol
        :param relative_path: the relative path to the file containing the symbol
        :param direction: "incoming" (who calls this), "outgoing" (what this calls), or "both"
        :param depth: maximum depth to expand (1-10); default 1
        :param max_answer_chars: max result length; -1 for default
        :return: JSON with incoming/outgoing call hierarchies
        """
        # validate direction
        if direction not in ("incoming", "outgoing", "both"):
            raise ValueError(f'direction must be "incoming", "outgoing", or "both", got "{direction}"')

        # validate depth
        if depth < 1 or depth > 10:
            raise ValueError(f"depth must be between 1 and 10, got {depth}")

        # sync file system before any LS call
        if relative_path:
            self.project.ls_sync_file_system_changes()

        # resolve symbol via retriever
        symbol_retriever = self.create_language_server_symbol_retriever()
        symbol = symbol_retriever.find_unique(name_path, substring_matching=False, within_relative_path=relative_path)
        resolved_name = symbol.get_name_path()

        # member path (property/field/event): early, member-specific branch. The callable code below
        # (call-hierarchy requests, F1 empty-check, _hierarchy_to_json_list) is not entered for members
        # and is left byte-for-byte unchanged (SPEC-call-hierarchy-members.md Rev 2, §4.1).
        if symbol.symbol_kind in MEMBER_KINDS:
            return self._apply_member(symbol, resolved_name, relative_path, direction, max_answer_chars)

        # request call hierarchy
        max_nodes_per_direction = 200  # CALL_HIERARCHY_MAX_NODES
        result_incoming: ls_types.CallHierarchyResult | None = None
        result_outgoing: ls_types.CallHierarchyResult | None = None

        if direction in ("incoming", "both"):
            result_incoming = symbol_retriever.request_call_hierarchy_by_location(
                symbol.location, direction="incoming", depth=depth, max_nodes=max_nodes_per_direction
            )

        if direction in ("outgoing", "both"):
            # Skip outgoing request if this is a "both" direction with approximate incoming
            if direction == "both" and result_incoming is not None and result_incoming.get("approximate"):
                # Approximate incoming: do not issue outgoing request
                result_outgoing = None
            else:
                remaining_nodes = max_nodes_per_direction
                if direction == "both" and result_incoming is not None:
                    remaining_nodes -= sum(self._count_nodes(node) for node in result_incoming["roots"])
                if remaining_nodes > 0:
                    result_outgoing = symbol_retriever.request_call_hierarchy_by_location(
                        symbol.location,
                        direction="outgoing",
                        depth=depth,
                        max_nodes=remaining_nodes,
                    )
                else:
                    # shared budget exhausted by the incoming direction: no request is made
                    result_outgoing = ls_types.CallHierarchyResult(roots=[], truncated=True, external_calls_omitted=0)

        # empty prepare on a supported server: distinct message, never an empty-but-plausible result (spec §3.3)
        requested_results = [r for r in (result_incoming, result_outgoing) if r is not None]
        if all(not r["roots"] and not r["truncated"] and r["external_calls_omitted"] == 0 for r in requested_results):
            return "No callable symbol found at the given position"

        # build output; every requested direction key is always present
        output: dict[str, object] = {
            "symbol": resolved_name,
            "direction": direction,
        }

        if result_incoming is not None:
            output["incoming"] = self._hierarchy_to_json_list(result_incoming["roots"])

        if result_outgoing is not None:
            output["outgoing"] = self._hierarchy_to_json_list(result_outgoing["roots"])

        # Handle approximate incoming (fallback from find-references)
        if result_incoming is not None and result_incoming.get("approximate"):
            ls_id = symbol_retriever.get_language_server(relative_path).ls_id.value
            approximate_note = f"{ls_id} has no call hierarchy support; incoming calls were derived from find-references: they may include non-call usages and may miss callers inside properties, constructors or one-line functions."
            output["approximate"] = True
            output["approximate_note"] = approximate_note
            # For direction=="both", suppress outgoing request and set unavailable message
            if direction == "both":
                output["outgoing"] = []
                output["outgoing_unavailable"] = "outgoing calls cannot be derived from find-references"

        # combine truncated and external_calls_omitted
        truncated = False
        external_calls_omitted = 0
        if result_incoming:
            truncated = truncated or result_incoming["truncated"]
            external_calls_omitted += result_incoming["external_calls_omitted"]
        if result_outgoing:
            truncated = truncated or result_outgoing["truncated"]
            external_calls_omitted += result_outgoing["external_calls_omitted"]

        output["truncated"] = truncated
        output["external_calls_omitted"] = external_calls_omitted

        # shortening factories; all forms preserve the approximate/outgoing_unavailable warnings
        def make_tree_without_call_sites() -> str:
            """Tree without call_sites for each node"""
            result_copy = copy.deepcopy(output)
            for direction_key in ("incoming", "outgoing"):
                if direction_key in result_copy:
                    self._remove_call_sites(cast(list[dict[str, object]], result_copy[direction_key]))
            return f"Call hierarchy (without call site details):\n{self._to_json(result_copy)}"

        def make_per_file_counts() -> str:
            """Per-file counts of incoming/outgoing calls"""
            counts_by_direction: dict[str, dict[str, int]] = {}
            for direction_key in ("incoming", "outgoing"):
                if direction_key in output:
                    file_counts: dict[str, int] = defaultdict(int)
                    self._count_nodes_by_file(cast(list[dict[str, object]], output[direction_key]), file_counts)
                    if file_counts:
                        counts_by_direction[direction_key] = dict(file_counts)
            summary = {"symbol": resolved_name, "direction": direction}
            summary.update(counts_by_direction)
            # Add approximate fields if present
            if "approximate" in output:
                summary["approximate"] = output["approximate"]
                summary["approximate_note"] = output["approximate_note"]
            if "outgoing_unavailable" in output:
                summary["outgoing_unavailable"] = output["outgoing_unavailable"]
            return f"Call hierarchy summary (counts per file):\n{self._to_json(summary)}"

        def make_summary() -> str:
            """One-line summary with total counts"""
            parts = []
            if result_incoming is not None:
                parts.append(f"{sum(self._count_nodes(n) for n in result_incoming['roots'])} incoming node(s)")
            if result_outgoing is not None:
                parts.append(f"{sum(self._count_nodes(n) for n in result_outgoing['roots'])} outgoing node(s)")
            summary_text = f"Call hierarchy for {resolved_name}: {', '.join(parts)}; truncated={truncated}"
            # Append approximate note if present
            if "approximate" in output:
                summary_text += f" [approximate: {output['approximate_note']}]"
            if "outgoing_unavailable" in output:
                summary_text += f"; outgoing unavailable: {output['outgoing_unavailable']}"
            return summary_text

        shortened_results = [make_tree_without_call_sites, make_per_file_counts, make_summary]

        result_json = self._to_json(output)
        return self._limit_length(result_json, max_answer_chars, shortened_result_factories=shortened_results)

    def _apply_member(
        self,
        symbol: LanguageServerSymbol,
        resolved_name: str,
        relative_path: str,
        direction: str,
        max_answer_chars: int,
    ) -> str:
        """
        Member (property/field/event) path -- SPEC-call-hierarchy-members.md (Rev 2) §4.2/§4.3.

        Produces a member-specific JSON shape (NOT `_hierarchy_to_json_list`): a synthetic root node for
        the queried member itself, whose children are the (grouped, deduped, deterministically sorted)
        referencing symbols, each carrying `call_sites` with a syntactically classified `access_kind` per
        site. `direction == "outgoing"` short-circuits before any reference query (outgoing calls do not
        apply to members). The F1 "No callable symbol..." string is never returned from this path -- a
        member with zero references still yields a valid synthetic root with `children: []`.
        """
        member_kind = self._member_kind_name(symbol)
        member_name = self._member_symbol_name(symbol, resolved_name)

        if direction == "outgoing":
            output: dict[str, object] = {
                "symbol": resolved_name,
                "direction": "outgoing",
                "member_kind": member_kind,
                "outgoing": [],
                "outgoing_unavailable": MEMBER_OUTGOING_UNAVAILABLE,
            }
            return self._limit_length(self._to_json(output), max_answer_chars)

        # incoming (or both): derive incoming usages from find-references.
        symbol_retriever = self.create_language_server_symbol_retriever()
        refs = symbol_retriever.find_referencing_symbols_by_location(symbol.location)
        root, truncated = self._build_member_incoming_root(
            symbol=symbol,
            member_kind=member_kind,
            member_name=member_name,
            relative_path=relative_path,
            refs=refs,
        )

        output = {
            "symbol": resolved_name,
            "direction": direction,
            "member_kind": member_kind,
            "incoming": [root],
            "approximate": True,
            "approximate_note": MEMBER_APPROXIMATE_NOTE,
            "truncated": truncated,
            "external_calls_omitted": 0,  # find_referencing_symbols_by_location exposes no drop count
        }
        if direction == "both":
            output["outgoing"] = []
            output["outgoing_unavailable"] = MEMBER_OUTGOING_UNAVAILABLE

        def make_tree_without_call_sites() -> str:
            """Tree without call_sites for the member root (drops per-site access_kind)."""
            result_copy = copy.deepcopy(output)
            self._remove_call_sites(cast(list[dict[str, object]], result_copy["incoming"]))
            return f"Call hierarchy (without call site details):\n{self._to_json(result_copy)}"

        def make_per_file_counts() -> str:
            """Per-file counts of incoming member usages."""
            file_counts: dict[str, int] = defaultdict(int)
            self._count_nodes_by_file(cast(list[dict[str, object]], output["incoming"]), file_counts)
            summary: dict[str, object] = {
                "symbol": resolved_name,
                "direction": direction,
                "member_kind": member_kind,
                "approximate": True,
                "approximate_note": MEMBER_APPROXIMATE_NOTE,
            }
            if file_counts:
                summary["incoming"] = dict(file_counts)
            if "outgoing_unavailable" in output:
                summary["outgoing_unavailable"] = output["outgoing_unavailable"]
            return f"Call hierarchy summary (counts per file):\n{self._to_json(summary)}"

        def make_minimal() -> str:
            """Smallest fallback: minimal JSON that STILL carries member_kind/approximate/approximate_note
            (SPEC §4.4) and drops call_sites/access_kind.
            """
            minimal: dict[str, object] = {
                "symbol": resolved_name,
                "direction": direction,
                "member_kind": member_kind,
                "approximate": True,
                "approximate_note": MEMBER_APPROXIMATE_NOTE,
                "incoming_usage_count": len(cast(list[object], root["children"])),
                "truncated": truncated,
            }
            if "outgoing_unavailable" in output:
                minimal["outgoing_unavailable"] = output["outgoing_unavailable"]
            return self._to_json(minimal)

        shortened_results: list[Callable[[], str]] = [make_tree_without_call_sites, make_per_file_counts, make_minimal]
        result_json = self._to_json(output)
        return self._limit_length(result_json, max_answer_chars, shortened_result_factories=shortened_results)

    def _build_member_incoming_root(
        self,
        symbol: LanguageServerSymbol,
        member_kind: str,
        member_name: str,
        relative_path: str,
        refs: list[ReferenceInLanguageServerSymbol],
    ) -> tuple[dict[str, object], bool]:
        """
        Build the synthetic member root + grouped/deduped/sorted children (SPEC §4.3, points 2-6).
        :return: (root node dict, truncated flag)
        """
        is_event = symbol.symbol_kind == SymbolKind.Event

        # 1) dedup identical sites (relative_path, line, character); 2) group by containing-symbol identity.
        seen_sites: set[tuple[str, int, int]] = set()
        groups: dict[object, tuple[LanguageServerSymbol, list[tuple[str, int, int]]]] = {}
        group_order: list[object] = []

        for ref in refs:
            containing = ref.symbol
            site_rel_path = self._member_location_relative_path(containing, relative_path)
            site_key = (site_rel_path, ref.line, ref.character)
            if site_key in seen_sites:
                continue
            seen_sites.add(site_key)

            group_key = self._member_group_key(containing)
            if group_key not in groups:
                groups[group_key] = (containing, [])
                group_order.append(group_key)
            groups[group_key][1].append(site_key)

        # 3) sort sites within each group and build lightweight (first_site, group_key) descriptors;
        #    sort groups deterministically and apply the node budget BEFORE any file reads/classification,
        #    so a symbol referenced by thousands of containers does not read+classify all of them just to
        #    discard the overflow.
        descriptors: list[tuple[tuple[str, int, int], object]] = []
        for group_key in group_order:
            _, sites = groups[group_key]
            sites.sort(key=lambda s: (s[1], s[2]))
            descriptors.append((sites[0], group_key))
        descriptors.sort(key=lambda d: d[0])
        truncated = len(descriptors) > MEMBER_MAX_NODES
        retained = descriptors[:MEMBER_MAX_NODES]

        # 4) classify sites (reading source lines) for the RETAINED groups only.
        file_lines_cache: dict[str, list[str] | None] = {}

        def get_line_text(rel_path: str, line: int) -> str | None:
            if rel_path not in file_lines_cache:
                try:
                    file_lines_cache[rel_path] = self.project.read_file(rel_path).split("\n")
                except Exception:
                    file_lines_cache[rel_path] = None
            lines = file_lines_cache[rel_path]
            if lines is None or not (0 <= line < len(lines)):
                return None
            return lines[line]

        children: list[dict[str, object]] = []
        for first_site, group_key in retained:
            containing, sites = groups[group_key]
            ranges: list[dict[str, object]] = []
            for site_rel_path, site_line, site_char in sites:
                line_text = get_line_text(site_rel_path, site_line)
                if line_text is not None:
                    access_kind = classify_member_access(line_text, site_char, member_name, is_event)
                else:
                    access_kind = "unknown"
                ranges.append(
                    {
                        "start": {"line": site_line, "character": site_char},
                        "end": {"line": site_line, "character": site_char + len(member_name)},
                        "access_kind": access_kind,
                    }
                )

            child_rel_path = first_site[0]
            children.append(
                {
                    "name": self._member_symbol_name(containing, member_name),
                    "kind": self._member_kind_name(containing),
                    "relative_path": child_rel_path,
                    "line": self._member_location_line(containing, first_site[1]),
                    "call_sites": {"relative_path": child_rel_path, "ranges": ranges},
                    "children": [],
                }
            )

        # 6) synthetic root = the queried member itself.
        root: dict[str, object] = {
            "name": member_name,
            "kind": member_kind,
            "relative_path": self._member_location_relative_path(symbol, relative_path),
            "line": self._member_location_line(symbol, 0),
            "children": children,
        }
        return root, truncated

    @staticmethod
    def _member_kind_name(sym: object) -> str:
        """Robustly resolve a `LanguageServerSymbol`-like object's symbol-kind name.

        Prefers the real `symbol_kind_name` property (already the enum name); falls back to wrapping
        `symbol_kind` in `SymbolKind(...)` (covers both raw ints and mocked test doubles that only set
        `.symbol_kind`, not `.symbol_kind_name`).
        """
        kind_name = getattr(sym, "symbol_kind_name", None)
        if isinstance(kind_name, str):
            return kind_name
        try:
            return SymbolKind(getattr(sym, "symbol_kind", None)).name
        except Exception:
            return "Unknown"

    @staticmethod
    def _member_symbol_name(sym: object, default: str) -> str:
        """Robustly resolve a `LanguageServerSymbol`-like object's unqualified name.

        Prefers the real `.name` property; falls back to the last name-path component (from
        `get_name_path()` when callable, else `default`, e.g. `resolved_name`) for test doubles that
        only stub `get_name_path()`.
        """
        name = getattr(sym, "name", None)
        if isinstance(name, str):
            return name
        get_name_path = getattr(sym, "get_name_path", None)
        if callable(get_name_path):
            try:
                name_path = get_name_path()
            except Exception:
                name_path = None
            if isinstance(name_path, str) and name_path:
                return name_path.rsplit("/", 1)[-1]
        return default.rsplit("/", 1)[-1]

    @staticmethod
    def _member_location_relative_path(sym: object, default: str) -> str:
        """Robustly resolve a `LanguageServerSymbol`-like object's file path, falling back to `default`
        (the relative_path the tool was invoked with) when `.location` is not a real
        `LanguageServerSymbolLocation` (e.g. an unconfigured mock in tests).
        """
        location = getattr(sym, "location", None)
        if isinstance(location, LanguageServerSymbolLocation) and isinstance(location.relative_path, str):
            return location.relative_path
        return default

    @staticmethod
    def _member_location_line(sym: object, default: int) -> int:
        """Robustly resolve a `LanguageServerSymbol`-like object's identifier line, falling back to
        `default` when `.location` is not a real `LanguageServerSymbolLocation`.
        """
        location = getattr(sym, "location", None)
        if isinstance(location, LanguageServerSymbolLocation) and isinstance(location.line, int):
            return location.line
        return default

    @staticmethod
    def _member_group_key(sym: object) -> tuple[object, ...]:
        """Grouping key for containing-symbol identity (SPEC §4.3.2): the containing symbol's own
        location + kind when it is a real, positioned `LanguageServerSymbolLocation`; otherwise the
        object's identity (safe fallback for test doubles whose `.location` is not configured).
        """
        location = getattr(sym, "location", None)
        if (
            isinstance(location, LanguageServerSymbolLocation)
            and isinstance(location.relative_path, str)
            and isinstance(location.line, int)
        ):
            return ("loc", location.relative_path, location.line, location.column, getattr(sym, "symbol_kind", None))
        return ("id", id(sym))

    @staticmethod
    def _count_nodes(node: ls_types.CallHierarchyNode) -> int:
        """Count total nodes in a subtree."""
        count = 1
        for child in node.get("children", []):
            count += CallHierarchyTool._count_nodes(child)
        return count

    @staticmethod
    def _hierarchy_to_json_list(nodes: list[ls_types.CallHierarchyNode]) -> list[dict[str, object]]:
        """Convert hierarchy nodes to JSON-compatible dicts."""
        result = []
        for node in nodes:
            node_dict: dict[str, object] = {
                "name": node["name"],
                "kind": SymbolKind(node["kind"]).name,  # render as enum name
                "relative_path": node["location"]["relativePath"],
                "line": node["location"]["range"]["start"]["line"],  # line from location
            }
            if "detail" in node:
                node_dict["detail"] = node["detail"]
            if "call_sites" in node:
                node_dict["call_sites"] = node["call_sites"]
            if "recursion" in node:
                node_dict["recursion"] = node["recursion"]
            if node.get("children"):
                node_dict["children"] = CallHierarchyTool._hierarchy_to_json_list(node["children"])
            else:
                node_dict["children"] = []
            result.append(node_dict)
        return result

    @staticmethod
    def _remove_call_sites(nodes: list[dict[str, object]]) -> None:
        """Recursively remove call_sites from all nodes in-place."""
        for node in nodes:
            if "call_sites" in node:
                del node["call_sites"]
            if "children" in node and isinstance(node["children"], list):
                CallHierarchyTool._remove_call_sites(cast(list[dict[str, object]], node["children"]))

    @staticmethod
    def _count_nodes_by_file(nodes: list[dict[str, object]], file_counts: dict[str, int]) -> None:
        """Recursively count nodes by file."""
        for node in nodes:
            rel_path = node.get("relative_path", "unknown")
            if isinstance(rel_path, str):
                file_counts[rel_path] = file_counts.get(rel_path, 0) + 1
            if "children" in node and isinstance(node["children"], list):
                CallHierarchyTool._count_nodes_by_file(cast(list[dict[str, object]], node["children"]), file_counts)
