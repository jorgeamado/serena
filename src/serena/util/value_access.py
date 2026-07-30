"""
Syntactic value-access classification for property/field/event usage sites.

Pure, single-line, best-effort. Given a source line and the column where a member-name token starts,
classify how that member is used at the site (read / write / read+write / event subscribe /
unsubscribe / invoke). This is the honest, LSP-only approximation of a semantic access model: it
returns ``"unknown"`` whenever a site cannot be classified confidently, and the whole call-hierarchy
member result is therefore reported as ``approximate``.

See SPEC-call-hierarchy-members.md §5 for the frozen contract.
"""

import re

# Frozen vocabulary (SPEC §5).
ACCESS_KINDS = ("read", "write", "read_write", "subscribe", "unsubscribe", "invoke", "unknown")

# Compound-assignment operators, longest-prefix first so e.g. ``>>>=`` is matched before ``>>=``.
_COMPOUND_OPS = (">>>=", "<<=", ">>=", "??=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=")

# A trailing receiver-qualifier segment: ``ident.`` / ``ident?.`` / ``).`` / ``].`` (with optional ws).
_QUALIFIER_SEGMENT_RE = re.compile(r"(?:[A-Za-z_]\w*|\)|\])\s*\??\.\s*$")
# The last identifier token at the end of a string (allowing trailing whitespace).
_IDENT_TAIL_RE = re.compile(r"([A-Za-z_]\w*)\s*$")


def _normalize_context_before(before: str) -> str:
    """Strip the receiver-qualifier chain (``w.``, ``this.``, ``foo?.bar.``) and trailing whitespace,
    so the token immediately governing the member access is exposed (SPEC §5).
    """
    ctx = before.rstrip()
    while True:
        m = _QUALIFIER_SEGMENT_RE.search(ctx)
        if m is None:
            return ctx
        ctx = ctx[: m.start()].rstrip()


def classify_member_access(line: str, char_start: int, member_name: str, is_event: bool) -> str:
    """
    Classify the access kind of a member usage at ``char_start`` on ``line``.

    :param line: the full source line (no trailing newline).
    :param char_start: 0-based column where the member NAME token starts.
    :param member_name: the member's name (used to guard against column drift / non-name sites).
    :param is_event: whether the member is an event.
    :return: one of :data:`ACCESS_KINDS`. Never raises.
    """
    # G. Name guard: if the span does not spell the member name, we cannot trust the position
    # (UTF-16 vs code-point drift, indexers with no name token, bad spans) -> unknown.
    end = char_start + len(member_name)
    if char_start < 0 or end > len(line) or line[char_start:end] != member_name:
        return "unknown"

    after = line[end:].lstrip(" \t")
    context_before = _normalize_context_before(line[:char_start])

    # N. nameof(...) -> no runtime access.
    if context_before.endswith("("):
        ident = _IDENT_TAIL_RE.search(context_before[:-1])
        if ident is not None and ident.group(1) == "nameof":
            return "unknown"

    # D. single-line deconstruction / tuple-assignment target: (a, X, b) = ...  -> cannot tell which
    # slots are targets on one line -> unknown (do not guess). Match the TOP-LEVEL closing paren of the
    # leading tuple (via a depth counter), so nested tuples like ``(a, (b, c)) = ...`` are handled.
    if line.lstrip().startswith("("):
        open_paren = line.index("(")
        depth = 0
        close = -1
        for i in range(open_paren, len(line)):
            if line[i] == "(":
                depth += 1
            elif line[i] == ")":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close != -1 and char_start < close:
            rest = line[close + 1 :].lstrip(" \t")
            if rest.startswith("=") and not rest.startswith("==") and not rest.startswith("=>"):
                return "unknown"

    # Event-specific forms (checked before generic compound assignment so += / -= mean subscribe).
    if is_event:
        if after.startswith("+="):
            return "subscribe"
        if after.startswith("-="):
            return "unsubscribe"
        invoke_tail = after[1:].lstrip(" \t") if after.startswith("!") else after  # null-forgiving E!()
        if invoke_tail.startswith(("(", "?(", "?.Invoke", ".Invoke")):
            return "invoke"

    # Compound assignment -> read + write.
    if any(after.startswith(op) for op in _COMPOUND_OPS):
        return "read_write"

    # Simple assignment target -> write (but not ==, =>).
    if after.startswith("=") and not after.startswith("==") and not after.startswith("=>"):
        return "write"

    # Postfix / prefix increment / decrement -> read + write.
    if after.startswith(("++", "--")):
        return "read_write"
    if context_before.endswith(("++", "--")):
        return "read_write"

    # By-reference argument modifiers (caller perspective): out=write, in=read, ref=read+write.
    modifier = _IDENT_TAIL_RE.search(context_before)
    if modifier is not None:
        word = modifier.group(1)
        if word == "out":
            return "write"
        if word == "in":
            return "read"
        if word == "ref":
            return "read_write"

    # Default: a plain value read (property getter / field read).
    return "read"


# --- hover-based symbol-kind detection (for the position-anchored call_hierarchy) -------------

# The fenced C# code block inside an LSP hover markdown value.
_HOVER_CODE_RE = re.compile(r"```(?:csharp|c#|cs)?[ \t]*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)

MEMBER_HOVER_KINDS = ("event", "property", "field")


def classify_hover_symbol(hover_value: str | None) -> tuple[str | None, str]:
    """
    Parse an LSP hover markdown value into ``(member_name, kind)`` for a C# symbol, so a
    position-anchored query can determine what it is pointing at.

    Recognizes the Roslyn hover signature shapes, e.g.
    ``CancellationToken CancellationTokenSource.Token { get; }`` (property),
    ``event Action Owner.OnX`` (event), ``void Owner.M(int x)`` (method), ``int Owner.F`` (field).

    :param hover_value: the hover ``contents.value`` markdown (or ``None``).
    :return: ``(simple member name or None, kind)`` where kind is one of
        ``"event" | "property" | "field" | "method" | "unknown"``. Never raises.
    """
    if not hover_value:
        return None, "unknown"
    m = _HOVER_CODE_RE.search(hover_value)
    signature = (m.group(1) if m else hover_value).strip()
    if not signature:
        return None, "unknown"
    first = signature.splitlines()[0].strip()

    is_event = first.startswith("event ")
    stripped = first.rstrip()
    # kind: event -> property (accessor block) -> method (trailing parameter list) -> field.
    # Using a TRAILING ')' (not any '(') avoids mis-reading a tuple return type like "(int, int) Foo.F".
    if is_event:
        kind = "event"
    elif "{" in first:
        kind = "property"
    elif stripped.endswith(")"):
        kind = "method"
    else:
        kind = "field"

    # member name: strip the suffix that belongs to the kind (accessor block / trailing param list),
    # then take the last '.'-segment's last identifier.
    head = first[len("event ") :] if is_event else first
    if kind == "property":
        head = re.sub(r"\s*\{.*$", "", head)
    elif kind == "method":
        lp = head.rfind("(")
        if lp != -1:
            head = head[:lp]
    head = head.strip()
    name: str | None = None
    if head:
        last_segment = head.split(".")[-1].strip()
        tokens = last_segment.split()
        if tokens:
            candidate = re.sub(r"<.*>$", "", tokens[-1])  # drop trailing generic args
            if re.fullmatch(r"@?[A-Za-z_]\w*", candidate):
                name = candidate.lstrip("@")
    return name, kind
