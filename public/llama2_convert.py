# llama2_convert.py
# Hugging Face Llama checkpoint -> what llama2_numpy.py loads, with nothing but NumPy, one piece of a tensor at a
# time: the weights are read through read(offset, length) and written into a buffer that has its final size from
# the start, float32, float16 or int8. So it never holds more than the output and a few megabytes, which is what
# lets the same code run when the site is built (convert_hf.py, quantize.py) and inside the browser, where the
# WebAssembly memory has 32 bits and never shrinks.
import json
import re
import struct
import time

import numpy as np

# the RoPE angles are the engine's, which computes them itself when a file leaves the tables out (int8)
from llama2_numpy import FORM, RMS_EPS, form_of, pack6, quantize6, rope_frequencies

# Pieces of at most this many values are converted at a time: 4 MB as float32. Measured on llm-jp-3-150m, the
# peak is the output plus 14 MB with this, plus 52 MB with pieces four times as large, at the same speed.
PIECE = 1024 * 1024


# ------------------------------------------------------------------------------------ the checkpoint format
def group_size(row_length):
    size = 32
    while row_length % size:
        size //= 2
    return size


def layout(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, bias=False, arch="llama", qk_norm=False,
           head_dim=0):
    """(shape, is a matrix) of every tensor, in file order. llama2_numpy.py reads the same order.

    is a matrix: True for what int8 quantizes, False for the norm weights, None for the RoPE tables.
    bias: the model adds a bias after the q, k and v projections (Qwen2). Those three vectors per layer go last,
    so that a checkpoint without them is byte for byte the file it always was.
    qk_norm: the model normalizes every head of q and k before RoPE (Qwen3, T124): the two weights of a head's size
    per layer go after the biases, for the same reason.
    head_dim: the size of a head where it is not dim / n_heads (0: it is). Then q and the attention's output are
    n_heads * head_dim wide, not dim (Qwen3 0.6B: 16 heads of 128 in a dim of 1024, T124).
    """
    head_size = head_dim or dim // n_heads
    q_dim, kv_dim = n_heads * head_size, n_kv_heads * head_size
    if arch in ("gpt2", "neox"):
        # GPT-2: LayerNorm (a weight and a bias), a bias after every projection, learned positions instead of
        # RoPE, and an FFN of two matrices instead of three (no gate). Same attention.
        # GPT-NeoX is the same, except that it rotates part of each head (so it keeps the RoPE tables of the
        # Llama layout in place of the table of positions).
        vector = lambda n=dim: ((n_layers, n), False)
        positions = [((seq_len, head_size // 2), None), ((seq_len, head_size // 2), None)] if arch == "neox" \
            else [((seq_len, dim), True)]
        tensors = [((abs(vocab_size), dim), True), *positions,
                   vector(), vector(),
                   ((n_layers, dim, dim), True), ((n_layers, dim, dim), True), ((n_layers, dim, dim), True),
                   vector(), vector(), vector(),
                   ((n_layers, dim, dim), True), vector(),
                   vector(), vector(),
                   ((n_layers, hidden_dim, dim), True), vector(hidden_dim),
                   ((n_layers, dim, hidden_dim), True), vector(),
                   ((dim,), False), ((dim,), False)]
        if vocab_size < 0:
            tensors.append(((abs(vocab_size), dim), True))
        return tensors
    tensors = [((abs(vocab_size), dim), True), ((n_layers, dim), False),
               ((n_layers, q_dim, dim), True), ((n_layers, kv_dim, dim), True), ((n_layers, kv_dim, dim), True),
               ((n_layers, dim, q_dim), True), ((n_layers, dim), False),
               ((n_layers, hidden_dim, dim), True), ((n_layers, dim, hidden_dim), True), ((n_layers, hidden_dim, dim), True),
               ((dim,), False), ((seq_len, head_size // 2), None), ((seq_len, head_size // 2), None)]
    if vocab_size < 0:
        tensors.append(((abs(vocab_size), dim), True))
    if bias:
        tensors += [((n_layers, q_dim), False), ((n_layers, kv_dim), False), ((n_layers, kv_dim), False)]
    if qk_norm:
        tensors += [((n_layers, head_size), False), ((n_layers, head_size), False)]
    return tensors


QUANTIZED = ("int8", "int6")  # the dtypes with groups and scales; int6 is T98's, see llama2_numpy.pack6


def dtype_name(dtype):
    """"float32", "float16", "int8" or "int6" from a name or a NumPy dtype (NumPy has no six-bit type)."""
    return "int6" if str(dtype) == "int6" else np.dtype(dtype).name


def check_dtype(dtype):
    if str(dtype) != "int6" and np.dtype(dtype) not in (np.float32, np.float16, np.int8):
        raise ValueError(f"dtype must be float32, float16, int8 or int6, not {dtype}.")


def tensor_bytes(shape, is_matrix, dtype):
    """How many bytes a tensor of layout() takes in a checkpoint of that dtype."""
    count, dtype = int(np.prod(shape)), dtype_name(dtype)
    if dtype not in QUANTIZED:
        return count * np.dtype(dtype).itemsize
    if is_matrix is None:
        return 0  # int8 and int6 checkpoints leave the RoPE tables out
    if dtype == "int6":
        # 24 bytes of values and a float32 scale per group of 32; the norm weights stay float32
        return count // 32 * 28 if is_matrix else 4 * count
    # int8 values and one float32 scale per group; the norm weights stay float32
    return count + 4 * (count // group_size(shape[-1])) if is_matrix else 4 * count


def checkpoint_size(header, dtype, form=None):
    """The bytes of a checkpoint with this header, dtype and form (llama2_numpy.FORM: what layout() takes besides
    the header)."""
    return 28 + sum(tensor_bytes(shape, is_matrix, dtype) for shape, is_matrix in layout(*header, **form_of(form)))


def quantize(values):
    """float32 values, whole rows -> (int8 values, float32 scales), one scale per group of the row."""
    groups = values.reshape(-1, group_size(values.shape[-1]))
    scales = (np.abs(groups).max(axis=1) / 127.0).astype(np.float32)
    inverse = np.divide(1.0, scales, out=np.zeros_like(scales), where=scales > 0)
    return np.rint(groups * inverse[:, None]).astype(np.int8), scales


class Writer:
    """Puts pieces of the tensors of layout(), in any order, where they belong in the checkpoint buffer."""

    def __init__(self, out, header, dtype, form=None, sink=None, quantize_rows=None):
        """out: a buffer of the checkpoint's size, or None with sink: an object with open(size, header, dtype,
        form) and write(offset, array of bytes), for a checkpoint that lives outside Python (T93: the WebAssembly
        memory of public/forward.js, which the header and the rest size, T115). Pyodide's own memory never shrinks,
        so a converted model that went through a Python buffer on its way there would keep taking its size twice.
        form (llama2_numpy.FORM, see checkpoint_form()): what the file does not say, which lays out its tensors and
        sizes the forward pass besides the header; the sink gets it whole."""
        # quantize_rows: quantize() on the SIMD kernels (llama2_numpy.kernel_quantizer), the same bytes six times
        # faster, for rows of whole groups of 32; NumPy's quantize() for anything else, and where there are no kernels
        self.dtype, self.sink, self.quantize_rows = dtype_name(dtype), sink, quantize_rows
        form = form_of(form)
        tensors = layout(*header, **form)
        if self.dtype == "int6" and any(is_matrix and shape[-1] % 32 for shape, is_matrix in tensors):
            raise ValueError("Six bits a weight needs rows of whole groups of 32, and this model has other rows.")
        size = checkpoint_size(header, dtype, form)
        if sink is not None:
            self.out = None
            sink.open(size, list(header), self.dtype, form)
        else:
            self.out = np.frombuffer(out, dtype=np.uint8)
            assert self.out.size == size, "the buffer has not the size of the checkpoint"
        self.put(0, np.frombuffer(struct.pack("<7i", *header), dtype=np.uint8))
        self.tensors, offset = [], 28
        for shape, is_matrix in tensors:
            self.tensors.append((offset, shape, is_matrix))
            offset += tensor_bytes(shape, is_matrix, dtype)

    def put(self, offset, array):
        raw = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
        if self.sink is not None:
            self.sink.write(offset, raw)
        else:
            self.out[offset:offset + raw.size] = raw

    def write(self, index, first, values):
        """values: whole rows of tensor number index, beginning at its element number first."""
        offset, shape, is_matrix = self.tensors[index]
        if self.dtype not in QUANTIZED:
            self.put(offset + first * np.dtype(self.dtype).itemsize, np.asarray(values).astype(self.dtype, copy=False))
        elif is_matrix and self.dtype == "int6":
            rows = np.asarray(values, dtype=np.float32).reshape(-1, shape[-1])
            if self.quantize_rows is not None:
                packed, scales = self.quantize_rows(rows, six=True)  # the same bytes on the kernel (T98)
            else:
                quantized, scales = quantize6(rows)
                packed = pack6(quantized)
            self.put(offset + first * 3 // 4, packed)
            self.put(offset + int(np.prod(shape)) * 3 // 4 + 4 * (first // 32), scales)
        elif is_matrix:
            fast = self.quantize_rows is not None and shape[-1] % 32 == 0
            quantized, scales = (self.quantize_rows if fast else quantize)(np.asarray(values, dtype=np.float32).reshape(-1, shape[-1]))
            self.put(offset + first, quantized)
            self.put(offset + int(np.prod(shape)) + 4 * (first // group_size(shape[-1])), scales)
        elif is_matrix is False:
            self.put(offset + 4 * first, np.asarray(values, dtype=np.float32))


# ---------------------------------------------------------------------------------- the chat template
# A chat_template is Jinja. This reads the part of Jinja those templates actually use: a loop over the
# messages, if / elif / else with the usual comparisons, set, string concatenation, the trim filter, and the
# whitespace control of {%- -%}; since T127 also the filters length, list and selectattr, namespace() and the
# setting of its attributes, integer arithmetic, the tests of "is", slices and string methods with arguments,
# which Qwen3's, Mistral v0.3's and sarashina2.2's templates use. A macro is skipped where it is defined (the
# templates define them for tools, which one turn has none of); calling one is Unsupported. Anything else raises
# Unsupported, and then the caller keeps whatever format src/models.js has for that model. Chosen over a real
# Jinja (jinja2 through micropip) to add no dependency.

class Unsupported(Exception):
    """This template uses something this reader does not know."""


def tokenize_template(text):
    """The template as a list of ("text", str) | ("say", expression) | ("do", statement)."""
    if text.endswith("\n"):
        text = text[:-1]   # Jinja drops one trailing newline of the source (keep_trailing_newline=False)
    out, i = [], 0
    while i < len(text):
        start = min([p for p in (text.find("{{", i), text.find("{%", i), text.find("{#", i)) if p != -1], default=-1)
        if start == -1:
            out.append(("text", text[i:]))
            break
        if start > i:
            out.append(("text", text[i:start]))
        opening, closing = {"{{": ("{{", "}}"), "{%": ("{%", "%}"), "{#": ("{#", "#}")}[text[start:start + 2]]
        # a comment is prose, and may hold an apostrophe (Llama 3.1's "user's"): no quotes inside it
        end = text.find(closing, start + 2) if opening == "{#" else find_outside_quotes(text, closing, start + 2)
        if end == -1:
            raise Unsupported(f"a {opening} that never closes")
        inner = text[start + len(opening):end]
        # {%- and -%} strip the whitespace next to them
        if inner.startswith("-"):
            inner = inner[1:]
            if out and out[-1][0] == "text":
                out[-1] = ("text", out[-1][1].rstrip())
        elif opening != "{{" and not inner.startswith("+"):
            # lstrip_blocks (transformers renders with it): the spaces and tabs from the start of its line to a
            # block or a comment go, when there is nothing else on the line before it
            indent = text[text.rfind("\n", 0, start) + 1:start]
            if indent and not indent.strip(" \t") and out and out[-1][0] == "text" and out[-1][1].endswith(indent):
                out[-1] = ("text", out[-1][1][:-len(indent)])
        strip_after = inner.endswith("-")
        if strip_after:
            inner = inner[:-1]
        # a comment says nothing, but it stands between: a {%- after it strips up to it, not the text before it
        out.append(("say" if opening == "{{" else "do", inner.strip()) if opening != "{#" else ("text", ""))
        i = end + len(closing)
        if strip_after:
            while i < len(text) and text[i] in " \t\r\n":
                i += 1
        elif opening != "{{" and text.startswith("\n", i):
            i += 1  # trim_blocks (transformers renders with it): the newline right after a block or a comment goes
    return out


# expressions, from the loosest to the tightest: or, and, not, the tests of "is", in, comparisons, + and -, * / %
# (integers), filters (|), and single values: 'text', "text", numbers, name, name['key'], name[a:b], name.attribute,
# name.method(...), namespace(...)
def evaluate(expression, scope):
    expression = expression.strip()
    while expression.startswith("(") and expression.endswith(")") and balanced(expression[1:-1]):
        expression = expression[1:-1].strip()
    for joiner, decided in ((" or ", truthy), (" and ", lambda value: not truthy(value))):
        # as in Jinja, the operand that decides, not True or False ('x' or 'default' is 'x'), and what follows it
        # unread (the review of T127: (system_message or 'You are ...') wrote "True")
        parts = split_outside_quotes(expression, joiner)
        if len(parts) > 1:
            for part in parts:
                value = evaluate(part, scope)
                if decided(value):
                    break
            return value
    if expression.startswith("not ") or (expression.startswith("not(") and balanced(expression[3:])):
        return not truthy(evaluate(expression[3:], scope))
    parts = split_outside_quotes(expression, " is ")
    if len(parts) == 2:
        test = parts[1].strip()
        negated = test.startswith("not ")
        return is_test(evaluate(parts[0], scope), test[4:].strip() if negated else test) != negated
    parts = split_outside_quotes(expression, " not in ")
    if len(parts) == 2:
        return not evaluate(f"({parts[0]}) in ({parts[1]})", scope)
    parts = split_outside_quotes(expression, " in ")
    if len(parts) == 2:
        needle, haystack = (evaluate(part, scope) for part in parts)
        return needle in haystack if isinstance(haystack, (str, list, tuple, dict)) else False
    for operator in ("==", "!=", ">=", "<=", ">", "<"):
        parts = split_outside_quotes(expression, operator)
        if len(parts) == 2:
            left, right = (evaluate(part, scope) for part in parts)
            return compare(left, right, operator)
    for operators in (("+", " - "), ("*", "/", "%")):
        terms = split_operators(expression, operators)
        if len(terms) > 1:
            value = evaluate(terms[0][1], scope)
            for operator, term in terms[1:]:
                value = arithmetic(value, operator.strip(), evaluate(term, scope))
            return value
    parts = split_outside_quotes(expression, "|")   # a | inside quotes, as in '<|im_start|>', is not a filter
    if len(parts) > 1:
        value, *filters = parts
        result = evaluate(value, scope)
        for spec in filters:
            result = apply_filter(result, spec.strip(), scope)
        return result
    return value_of(expression, scope)


def is_test(value, test):
    """value is test: defined, none, string, number, mapping, iterable, sequence, true, false. None counts as not
    defined: one_turn() gives tools and documents as None where transformers leaves them out or passes None, and
    the templates test them with "is defined" (T73 matched transformers on 23 templates so)."""
    tests = {"defined": lambda v: v is not MISSING and v is not None, "undefined": lambda v: v is MISSING or v is None,
             "none": lambda v: v is None or v is MISSING,
             "string": lambda v: isinstance(v, str), "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
             "mapping": lambda v: isinstance(v, (dict, Namespace)), "iterable": lambda v: isinstance(v, (str, list, tuple, dict)),
             "sequence": lambda v: isinstance(v, (str, list, tuple)), "true": lambda v: v is True, "false": lambda v: v is False}
    if test not in tests:
        raise Unsupported(f"the test is {test}")
    return tests[test](value)


def compare(left, right, operator):
    if operator in ("==", "!="):
        same = left == right and (left is MISSING) == (right is MISSING)
        return same if operator == "==" else not same
    if not (isinstance(left, (int, float)) and isinstance(right, (int, float))) and not (isinstance(left, str) and isinstance(right, str)):
        raise Unsupported(f"{left!r} {operator} {right!r}")
    return {">": left > right, "<": left < right, ">=": left >= right, "<=": left <= right}[operator]


def arithmetic(left, operator, right):
    """+ joins text (Jinja's templates add strings far more than numbers) and adds numbers; the others are integers'."""
    numbers = all(isinstance(v, int) and not isinstance(v, bool) for v in (left, right))
    if operator == "+":
        return left + right if numbers else as_text(left) + as_text(right)
    if not numbers:
        raise Unsupported(f"{left!r} {operator} {right!r}")
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if right == 0:
        raise Unsupported("a division by zero")
    return left % right if operator == "%" else left / right


def apply_filter(value, spec, scope):
    """The filters the templates use on one turn: trim, length, list, first, last, and selectattr / rejectattr
    (attribute, test[, value]) with the tests equalto (==), defined and none."""
    name, _, rest = spec.partition("(")
    name = name.strip()
    arguments = [evaluate(argument, scope) for argument in split_outside_quotes(rest[:-1], ",") if argument.strip()] if rest else []
    if name == "trim" and not arguments:
        return as_text(value).strip()
    if name in ("length", "count") and not arguments and isinstance(value, (str, list, tuple, dict)):
        return len(value)
    if name == "list" and not arguments and isinstance(value, (str, list, tuple)):
        return list(value)
    if name in ("first", "last") and not arguments and isinstance(value, (list, tuple)):
        return (value[0] if name == "first" else value[-1]) if value else MISSING
    if name in ("selectattr", "rejectattr") and isinstance(value, (list, tuple)) and 1 <= len(arguments) <= 3:
        attribute, test, *expected = arguments + ([] if len(arguments) > 1 else ["defined"])
        def passes(item):
            got = item.get(attribute, MISSING) if isinstance(item, dict) else getattr(item, str(attribute), MISSING)
            if test in ("equalto", "eq", "==", "sameas"):
                return expected and got == expected[0]
            if test in ("defined", "none") and not expected:
                return is_test(got, test)
            raise Unsupported(f"selectattr with the test {test}")
        return [item for item in value if passes(item) == (name == "selectattr")]
    raise Unsupported(f"the filter {spec}")


STRFTIME = object()   # so that "strftime_now is defined" is true, as it is in transformers
DAY = "\x00day"  # in the scope: the day strftime_now() writes, instead of {date:format} (see one_turn())
# two days that differ in every field strftime_now() may write, the first and the last of a year
CHECK_DAYS = [time.strptime(day, "%Y-%m-%d %H:%M:%S") for day in ("2025-01-01 00:00:00", "2026-12-31 23:59:59")]
MISSING = object()  # a name the template asks for and nothing set: Jinja calls it undefined, and it is false


def truthy(value):
    return bool(value) and value is not MISSING


def as_text(value):
    if value is MISSING or value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def balanced(text):
    """Whether the brackets of text are balanced, so that its outer pair can be dropped."""
    depth, quote, escaped = 0, "", False
    for char in text:
        if quote:
            quote, escaped = ("" if char == quote and not escaped else quote), char == "\\" and not escaped
        elif char in "'\"":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def find_outside_quotes(text, needle, start):
    """Where needle is, skipping what is inside quotes: a template may write }} or %} inside a string."""
    quote, i = "", start
    while i < len(text):
        char = text[i]
        if quote:
            if char == "\\":
                i += 1  # an escaped character, as in 'the user\'s'
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif text.startswith(needle, i):
            return i
        i += 1
    return -1


def split_outside_quotes(text, separator):
    """text.split(separator), but not inside quotes or brackets."""
    parts, depth, quote, start, i = [], 0, "", 0, 0
    while i < len(text):
        char = text[i]
        if quote:
            if char == "\\":
                i += 1  # an escaped character
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif depth == 0 and text.startswith(separator, i):
            parts.append(text[start:i])
            i += len(separator)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return [part for part in parts]


class Namespace:
    """What namespace(a=1, b=2) makes: attributes a {% set ns.a = ... %} may change inside a loop."""

    def __init__(self, **values):
        self.__dict__.update(values)


def split_operators(expression, operators):
    """[(operator, term)] of a chain of + and - (or * / %), outside quotes and brackets; the first operator is ""."""
    marks = []
    for operator in operators:
        at = 0
        for part in split_outside_quotes(expression, operator)[:-1]:
            at += len(part)
            marks.append((at, operator))
            at += len(operator)
    marks.sort()
    terms, start, previous = [], 0, ""
    for at, operator in marks:
        term = expression[start:at]
        if not term.strip():  # a sign, as in -1: not an operator
            return [("", expression)]
        terms.append((previous, term))
        start, previous = at + len(operator), operator
    terms.append((previous, expression[start:]))
    return terms if all(term.strip() for _, term in terms) else [("", expression)]


def call_arguments(text, scope):
    """The values of "a, 'b', c=1" (the part between the brackets of a call): ([positional], {keyword})."""
    positional, keyword = [], {}
    for argument in split_outside_quotes(text, ","):
        if not argument.strip():
            continue
        name, equals, value = argument.partition("=")
        if equals and name.strip().isidentifier() and not value.startswith("="):
            keyword[name.strip()] = evaluate(value, scope)
        else:
            positional.append(evaluate(argument, scope))
    return positional, keyword


def string_end(text):
    """Where the string that text begins with ends (its closing quote), a backslash escaping the next character."""
    i = 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == text[0]:
            return i
        i += 1
    raise Unsupported(f"a string that never closes: {text!r}")


def closing_bracket(text, start):
    """Where the bracket opened at text[start] closes, outside quotes."""
    depth, quote, escaped = 0, "", False
    for i in range(start, len(text)):
        char = text[i]
        if quote:
            quote, escaped = ("" if char == quote and not escaped else quote), char == "\\" and not escaped
        elif char in "'\"":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth == 0:
                return i
    raise Unsupported(f"a bracket that never closes in {text!r}")


# the string methods a template may call on one turn, and how many arguments each takes at most
METHODS = {"capitalize": 0, "lower": 0, "upper": 0, "title": 0, "strip": 1, "lstrip": 1, "rstrip": 1,
           "startswith": 1, "endswith": 1, "split": 2, "replace": 2}


def value_of(expression, scope):
    """One value: a literal, a name, or a name with ['key'], [a:b], .attribute and .method(...) after it."""
    expression = expression.strip()
    if not expression:
        raise Unsupported("an empty expression")
    if expression[0] in "'\"" and string_end(expression) == len(expression) - 1:
        return unescape(expression[1:-1])
    if expression.lstrip("-").isdigit():
        return int(expression)
    if expression.startswith("namespace(") and closing_bracket(expression, len("namespace")) == len(expression) - 1:
        positional, keyword = call_arguments(expression[len("namespace("):-1], scope)
        if positional:
            raise Unsupported("namespace() with values without names")
        return Namespace(**keyword)
    if expression in ("true", "True"):
        return True
    if expression in ("false", "False"):
        return False
    if expression in ("none", "None"):
        return None
    if expression.startswith("strftime_now(") and expression.endswith(")"):
        # the one function these templates call: the date of today, for a system prompt. Written as {date:format},
        # which filled() in src/models.js makes the visitor's day when the prompt is sent: the format is kept with
        # the conversion, and a date written now went stale from the next day on (the review of T127)
        argument = expression[len("strftime_now("):-1].strip()
        if not argument or argument[0] not in "'\"" or string_end(argument) != len(argument) - 1:
            raise Unsupported(f"strftime_now of {argument!r}")
        form = unescape(argument[1:-1])
        directives = form.replace("%%", "").split("%")[1:]  # what follows each %: the ones filled() knows
        if "}" in form or not all(directive[:1] and directive[0] in "dmYybBaAHMS" for directive in directives):
            raise Unsupported(f"the date format {form!r}")
        day = scope.get(DAY)  # a day to write, when one_turn() checks the template against real dates
        return "{date:" + form + "}" if day is None else time.strftime(form, day)
    name, rest = expression, ""
    for cut in ("[", "."):
        at = expression.find(cut)
        if at != -1 and at < len(name):
            name, rest = expression[:at], expression[at:]
    if not name.replace("_", "").isalnum():
        raise Unsupported(f"the expression {expression!r}")
    value = scope.get(name, MISSING)
    while rest:
        if rest.startswith("["):
            end = closing_bracket(rest, 0)
            inside, rest = rest[1:end], rest[end + 1:]
            if ":" in split_outside_quotes(inside, ":")[0] or len(split_outside_quotes(inside, ":")) > 1:
                # a slice, as in messages[1:] and messages[::-1]
                bounds = [evaluate(bound, scope) if bound.strip() else None for bound in split_outside_quotes(inside, ":")]
                if len(bounds) > 3 or not all(bound is None or isinstance(bound, int) for bound in bounds):
                    raise Unsupported(f"the slice [{inside}]")
                value = value[slice(*bounds)] if isinstance(value, (str, list, tuple)) else MISSING
                continue
            key = evaluate(inside, scope)
        elif rest.startswith("."):
            piece = rest[1:]
            length = len(piece) - len(piece.lstrip("_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
            key, rest = piece[:length], piece[length:]
            if rest.startswith("("):  # a method, of the few the templates call
                end = closing_bracket(rest, 0)
                arguments, keyword = call_arguments(rest[1:end], scope)
                rest = rest[end + 1:]
                if key not in METHODS or keyword or len(arguments) > METHODS[key]:
                    raise Unsupported(f"the method {key}()")
                if value is not MISSING:
                    value = getattr(as_text(value), key)(*arguments)
                continue
        else:
            raise Unsupported(f"the expression {expression!r}")
        if value is MISSING:
            continue
        if isinstance(value, dict):
            value = value.get(key, MISSING)
        elif isinstance(value, (list, tuple)) and isinstance(key, int):
            value = value[key] if -len(value) <= key < len(value) else MISSING
        else:
            value = getattr(value, str(key), MISSING)
    return value


def unescape(text):
    """A string literal's escapes, as Jinja's lexer reads them: Python's unicode-escape, left to right (the review of
    T127: "\\\\n" was a backslash and a newline, and \\x41 and \\u2581 stayed as they were)."""
    return text.replace("\r\n", "\n").encode("ascii", "backslashreplace").decode("unicode-escape")


class Loop:
    """What {% for %} puts in scope as loop."""

    def __init__(self, index, total):
        self.index0, self.index = index, index + 1
        self.first, self.last = index == 0, index == total - 1
        self.length = total


def render(template, scope):
    """The template, with the names of scope. Raises Unsupported for anything this reader does not know."""
    pieces = tokenize_template(template)
    out = []
    run(pieces, 0, len(pieces), scope, out)
    return "".join(out)


def run(pieces, start, stop, scope, out):
    i = start
    while i < stop:
        kind, body = pieces[i]
        if kind == "text":
            out.append(body)
            i += 1
        elif kind == "say":
            out.append(as_text(evaluate(body, scope)))
            i += 1
        elif body.startswith("for "):
            end = matching(pieces, i, stop, "for ", "endfor")
            name, _, source = body[4:].partition(" in ")
            if "," in name:
                raise Unsupported("a for over pairs")
            values = evaluate(source, scope)
            if not isinstance(values, (list, tuple)):
                raise Unsupported(f"a for over {source.strip()!r}")
            for index, value in enumerate(values):
                run(pieces, i + 1, end, {**scope, name.strip(): value, "loop": Loop(index, len(values))}, out)
            i = end + 1
        elif body.startswith("if "):
            end = matching(pieces, i, stop, "if ", "endif")
            branches, at = [], i
            while at < end:  # if / elif / elif / else, as (condition or None, where the body begins)
                statement = pieces[at][1]
                if statement.startswith(("if ", "elif ")):
                    branches.append((statement.split(" ", 1)[1], at + 1))
                elif statement == "else":
                    branches.append((None, at + 1))
                at = next_branch(pieces, at, end)
            ends = [begin - 1 for _, begin in branches[1:]] + [end]
            for (condition, begin), finish in zip(branches, ends):
                if condition is None or truthy(evaluate(condition, scope)):
                    run(pieces, begin, finish, scope, out)
                    break
            i = end + 1
        elif body.startswith("set "):
            name, _, expression = body[4:].partition("=")
            target, _, attribute = name.strip().partition(".")
            if attribute:  # {% set ns.index = ... %}: an attribute of a namespace()
                if not isinstance(scope.get(target), Namespace) or not attribute.isidentifier():
                    raise Unsupported(f"{{% {body} %}}")
                setattr(scope[target], attribute, evaluate(expression, scope))
            else:
                scope[target] = evaluate(expression, scope)
            i += 1
        elif body.startswith("macro "):
            # defined for tools, which one turn has none of: skipped (a call of it is an expression it cannot read)
            i = matching(pieces, i, stop, "macro ", "endmacro") + 1
        elif body in ("endfor", "endif", "else", "endmacro") or body.startswith("elif "):
            raise Unsupported(f"{body!r} without its opening")
        else:
            raise Unsupported(f"{{% {body} %}}")


def matching(pieces, start, stop, opening, closing):
    """Where the {% endfor %} or {% endif %} of the statement at start is."""
    depth = 0
    for i in range(start, stop):
        kind, body = pieces[i]
        if kind != "do":
            continue
        if body.startswith(opening):
            depth += 1
        elif body == closing:
            depth -= 1
            if depth == 0:
                return i
    raise Unsupported(f"a {opening.strip()} without its {closing}: {pieces[start][1][:40]!r} at {start}, looking to {stop}")


def next_branch(pieces, start, end):
    """The elif / else / endif that follows the branch beginning at start, at the same level."""
    depth = 0
    for i in range(start + 1, end + 1):
        kind, body = pieces[i]
        if kind != "do":
            continue
        if body.startswith(("if ", "for ")):
            depth += 1
        elif body in ("endif", "endfor"):
            if depth == 0:
                return i
            depth -= 1
        elif depth == 0 and (body == "else" or body.startswith("elif ")):
            return i
    return end


def one_turn_template(tokenizer_config, chat_template=None):
    """The template of one user turn from a tokenizer_config.json, or None when there is none this can read.
    chat_template: the text of chat_template.jinja, where the repository has one (T127: transformers now saves the
    template there rather than in tokenizer_config.json, and reads it first); the tokens it names are still the
    tokenizer_config's."""
    try:
        config = json.loads(tokenizer_config) if isinstance(tokenizer_config, (str, bytes)) else tokenizer_config
    except ValueError:
        config = None
    if not isinstance(config, dict):
        if not chat_template:
            return None
        config = {}
    template = chat_template or config.get("chat_template")
    if isinstance(template, list):  # some models publish several; the first is the chat one
        template = template[0].get("template") if template and isinstance(template[0], dict) else None
    if not isinstance(template, str) or not template.strip():
        return None
    token = lambda name: config.get(name) if isinstance(config.get(name), str) else (config.get(name) or {}).get("content", "")
    bos = token("bos_token") or ""
    turn = one_turn(template, {"bos_token": bos, "eos_token": token("eos_token") or ""})
    # generate() starts every run with the BOS token already: one written by the template would be a second one
    return turn[len(bos):] if turn and bos and turn.startswith(bos) else turn


def one_turn(template, specials, mark="\x00prompt\x00"):
    """The template of one user turn, as src/models.js writes it: the text around a {prompt}.

    specials: the tokenizer's special tokens, for bos_token and eos_token. Returns None when the template uses
    something this reader does not know, and then the caller keeps the format it has.
    """
    scope = {"messages": [{"role": "user", "content": mark}], "add_generation_prompt": True,
             "bos_token": specials.get("bos_token", ""), "eos_token": specials.get("eos_token", ""),
             "tools": None, "tools_json": None, "documents": None, "strftime_now": STRFTIME}
    try:
        text = render(template, dict(scope))
        # strftime_now() is written {date:format}, which the page fills with the day it sends the prompt. That holds
        # where the template only writes the date; one that reckons with it (yesterday's date from today's day of
        # the month) would come out otherwise, so the two are compared on two days, and it gives up where they differ
        days = CHECK_DAYS if "{date:" in text else []
        fill = lambda day: re.sub(r"\{date:([^}]*)\}", lambda found: time.strftime(found.group(1), day), text)
        if any(fill(day) != render(template, {**scope, DAY: day}) for day in days):
            return None
    except Unsupported:
        return None
    except Exception:
        return None
    if text.count(mark) != 1:
        return None
    # T138: a template that trims what was typed ({{ message['content'] | trim }}) says so as {prompt:trim}, which the
    # page's filled() trims: rendered once more with spaces around the mark, it either keeps them or drops both.
    # (Whether the spaces are there is no test: RakutenAI's writes "USER: " and " ASSISTANT:" around a trimmed one)
    try:
        spaced = render(template, {**scope, "messages": [{"role": "user", "content": f" {mark} "}]})
    except Exception:
        return None
    if spaced == text.replace(mark, f" {mark} "):
        return text.replace(mark, "{prompt}")
    if spaced == text:
        return text.replace(mark, "{prompt:trim}")
    return None  # it drops one of the two, or changes more than that: nothing filled() would do


# ------------------------------------------------------------------------------------------ the weights
def bfloat16(raw):
    # NumPy has no bfloat16, but a bfloat16 is exactly the upper half of a float32: widening is a shift
    wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    wide <<= 16
    return wide.view(np.float32)


def q8_0(raw):
    """GGUF's Q8_0 (T74): blocks of 32 values, each a float16 scale and 32 int8. The same groups of 32 as this
    project's int8, so quantize() gets the very same int8 back: the scale of a block is its largest value / 127."""
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32)
    values = np.ascontiguousarray(blocks[:, 2:]).view(np.int8)
    return (values * scales).reshape(-1)


# bytes per value (Q8_0: 34 bytes for 32 of them), and how to read them
READERS = {"F32": (4, lambda raw: np.frombuffer(raw, dtype=np.float32)),
           "F16": (2, lambda raw: np.frombuffer(raw, dtype=np.float16)), "BF16": (2, bfloat16),
           "Q8_0": (34 / 32, q8_0)}


class Safetensors:
    """The tensors of a .safetensors file behind read(offset, length): a local file, a File of the browser, a URL."""

    def __init__(self, read):
        self.read = read
        (header_size,) = struct.unpack("<Q", bytes(read(0, 8)))
        if not 2 <= header_size <= 100_000_000:
            raise ValueError("This is not a safetensors file.")
        try:
            self.tensors = {name: info for name, info in json.loads(bytes(read(8, header_size))).items() if name != "__metadata__"}
        except ValueError:
            raise ValueError("This is not a safetensors file.") from None
        self.base = 8 + header_size

    def __contains__(self, name):
        return name in self.tensors

    def shape(self, name):
        return tuple(self.tensors[name]["shape"])

    def rows(self, name, start, stop):
        """Rows start..stop of a tensor (all of a vector), as float32 or float16."""
        info = self.tensors[name]
        if info["dtype"] not in READERS:
            raise ValueError(f"{name} is stored as {info['dtype']}: only float32, float16 and bfloat16 are supported.")
        itemsize, reader = READERS[info["dtype"]]
        shape = self.shape(name)
        row = int(np.prod(shape[1:]))
        begin = self.base + info["data_offsets"][0] + int(start * row * itemsize)
        return reader(self.read(begin, int((stop - start) * row * itemsize))).reshape(stop - start, *shape[1:])


class Shards:
    """The tensors of a model split over several .safetensors files (model-00001-of-00002.safetensors, ...), as one
    source: the build's way in (convert_hf.py). Each shard is a Safetensors; a name is looked up in whichever has it."""

    def __init__(self, shards):
        self.owner = {}
        for shard in shards:
            for name in shard.tensors:
                if name in self.owner:
                    raise ValueError(f"{name} is in two shards of this model.")
                self.owner[name] = shard
        self.tensors = {name: shard.tensors[name] for name, shard in self.owner.items()}

    def __contains__(self, name):
        return name in self.owner

    def shape(self, name):
        return self.owner[name].shape(name)

    def rows(self, name, start, stop):
        return self.owner[name].rows(name, start, stop)


def joined_shards(headers):
    """The page's way in for a model split over several files (T105): the JSON headers of the shards, in the order
    their data will be fed, as the header of one file made of their tensor data one after another (base 0). Returns
    that header (text) and, for every shard, how many bytes of data it has: feed each shard from its own base (8 +
    the length of its header) for that many bytes, and Stream sees one file. Nothing of Stream changes."""
    joined, at, lengths = {}, 0, []
    for text in headers:
        try:
            header = json.loads(bytes(text.to_py() if hasattr(text, "to_py") else text).decode()
                                if not isinstance(text, str) else text)
        except ValueError:
            raise ValueError("A shard of this model is not a safetensors file.") from None
        tensors = {name: info for name, info in header.items() if name != "__metadata__"}
        length = max((info["data_offsets"][1] for info in tensors.values()), default=0)
        for name, info in tensors.items():
            if name in joined:
                raise ValueError(f"{name} is in two shards of this model.")
            begin, end = info["data_offsets"]
            joined[name] = {**info, "data_offsets": [at + begin, at + end]}
        lengths.append(length)
        at += length
    return json.dumps(joined), lengths


class Arrays:
    """The same interface for tensors that are in memory already (a PyTorch checkpoint, a test)."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __contains__(self, name):
        return name in self.tensors

    def shape(self, name):
        return tuple(self.tensors[name].shape)

    def rows(self, name, start, stop):
        return self.tensors[name][start:stop]


def architecture(config):
    """Which set of tensors and which forward: "llama" (Qwen2 is a Llama with biases), "gpt2", or "neox"
    (GPT-NeoX: a GPT-2 with RoPE over part of each head, and optionally the two branches in parallel)."""
    return {"gpt2": "gpt2", "gpt_neox": "neox"}.get(config.get("model_type"), "llama")


def head_size(config):
    """The size of an attention head: config.json's head_dim where it says one (Qwen3 0.6B: 128 in a dim of 1024,
    T124), else dim / heads. "head_dim": null says as much as no head_dim at all (cyberagent/CAT-Translate-7b)."""
    return config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]


def rotary_dim(config):
    """How many of each head's values GPT-NeoX rotates (rotary_pct of them, an even number)."""
    head_size = config["hidden_size"] // config["num_attention_heads"]
    return int(head_size * float(config.get("rotary_pct", 1.0))) // 2 * 2


def normalize(config):
    """GPT-2 spells its config.json differently: give it the names the rest of this file uses."""
    rope = config.get("rope_parameters")
    if isinstance(rope, dict):
        # transformers 5 writes rope_theta, rope_scaling and GPT-NeoX's rotary_pct as one rope_parameters. Unread,
        # such a config.json ran at theta 10000, unscaled and rotating whole heads, and wrote nonsense (the review
        # of T106): the old names are what this file reads
        scaling = {key: value for key, value in rope.items() if key not in ("rope_theta", "partial_rotary_factor")}
        config = {**config, "rope_theta": rope.get("rope_theta", config.get("rope_theta", 10000.0))}
        if scaling.get("rope_type", scaling.get("type", "default")) != "default":
            config["rope_scaling"] = scaling
        if "partial_rotary_factor" in rope:
            config["rotary_pct"] = rope["partial_rotary_factor"]
    if config.get("model_type") == "mistral":
        # T125: a Mistral is a Llama by another name (the same tensors, names and forward). v0.1 and some of its
        # descendants attend a sliding window of the last sliding_window positions: a context no longer than the
        # window attends the very same positions, so the context is cut to it
        window, context = config.get("sliding_window"), config.get("max_position_embeddings")
        config = {**config, "model_type": "llama"}
        if isinstance(window, int) and window > 0 and isinstance(context, int):
            config["max_position_embeddings"] = min(context, window)
        return config
    if config.get("model_type") == "gpt_neox":
        # GPT-NeoX has the Llama names already; only the angles are spelled differently
        return {**config, "rope_theta": config.get("rotary_emb_base", config.get("rope_theta", 10000.0)),
                "tie_word_embeddings": config.get("tie_word_embeddings", False)}
    if config.get("model_type") != "gpt2":
        return config
    dim = config.get("n_embd")
    return {**config, "hidden_size": dim, "intermediate_size": config.get("n_inner") or (4 * dim if dim else None),
            "num_hidden_layers": config.get("n_layer"), "num_attention_heads": config.get("n_head"),
            "max_position_embeddings": config.get("n_positions") or config.get("n_ctx"),
            "hidden_act": "gelu", "tie_word_embeddings": config.get("tie_word_embeddings", True)}


def check_config(config):
    """ValueError, in words for the visitor, unless this config.json describes a model the engine can run."""
    def refuse(reason):
        raise ValueError(f"This model cannot be converted: {reason}.")

    # qwen2 is a Llama with a bias on q, k and v: the converter writes those three vectors per layer, the engine
    # adds them after the projections (T64). Everything else about it is the same. qwen3 is a Llama that normalizes
    # every head of q and k (T124): two vectors per layer, the same way.
    if config.get("model_type") not in ("llama", "qwen2", "qwen3", "gpt2", "gpt_neox"):
        refuse(f"it is a {config.get('model_type', 'model of unknown type')}, and only Llama, Mistral, Qwen2, Qwen3, "
               f"GPT-2 and GPT-NeoX models are supported")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "vocab_size",
                "max_position_embeddings"):
        if not isinstance(config.get(key), int) or config[key] <= 0:
            refuse(f"its config.json has no usable {key}")
    dim, n_heads = config["hidden_size"], config["num_attention_heads"]
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    size = head_size(config)
    # a head of another size than dim / n_heads (T124) only where q and o are matrices of their own: a Llama's.
    # GPT-2's c_attn and GPT-NeoX's query_key_value are cut into heads of dim / n_heads
    divides = not dim % n_heads and size == dim // n_heads
    if not isinstance(size, int) or size <= 0 or size % 2 or n_heads % n_kv_heads \
            or not (divides or (config.get("head_dim") and architecture(config) == "llama")):
        refuse("its attention heads do not divide the hidden size the way llama2.c expects")
    scaling = config.get("rope_scaling")
    if scaling and (architecture(config) != "llama" or scaling.get("rope_type", scaling.get("type")) not in ("llama3", "linear")):
        # Llama 3's and the linear one are the kinds the RoPE tables know (llama2_numpy.rope_frequencies)
        refuse(f"it uses RoPE scaling of the {scaling.get('rope_type', scaling.get('type'))} kind")
    if architecture(config) == "neox":
        if config.get("hidden_act", "gelu") not in ("gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh"):
            refuse(f"its activation is {config['hidden_act']}, and only GELU is supported")
        if config.get("num_key_value_heads", config["num_attention_heads"]) != config["num_attention_heads"]:
            refuse("it has grouped-query attention, which GPT-NeoX models do not")
        if rotary_dim(config) < 2:
            refuse("it rotates none of each head")
        return
    if architecture(config) == "gpt2":
        # GPT-2 has one kind of everything; only the activation could be something the GELU kernel is not
        # gelu_fast (T126: rinna/japanese-gpt-1b) is gelu_new's tanh approximation written another way
        if config.get("activation_function", "gelu_new") not in ("gelu_new", "gelu", "gelu_pytorch_tanh", "gelu_fast"):
            refuse(f"its activation is {config['activation_function']}, and only GELU is supported")
        if config.get("num_key_value_heads", config["num_attention_heads"]) != config["num_attention_heads"]:
            refuse("it has grouped-query attention, which GPT-2 models do not")
        return
    if config.get("hidden_act", "silu") != "silu":
        refuse(f"its activation is {config['hidden_act']}, not silu")
    if config.get("mlp_bias") or (config.get("attention_bias") and config.get("model_type") != "qwen2"):
        refuse("its layers have biases")
    if config.get("use_sliding_window"):
        refuse("it uses a sliding window of attention")


def checkpoint_header(config, source, max_seq_len):
    """The 7 ints of the legacy header. A negative vocabulary size signals a classifier of its own (llama2.c)."""
    config = normalize(config)
    # GPT-NeoX calls its classifier embed_out, everyone else lm_head
    classifier = "embed_out.weight" if architecture(config) == "neox" else "lm_head.weight"
    shared_classifier = config.get("tie_word_embeddings", False) or classifier not in source
    if architecture(config) == "gpt2":
        # the learned positions are a table of exactly n_positions rows: the context cannot be cut short
        max_seq_len = config["max_position_embeddings"]
    vocab_size = config["vocab_size"]
    # the KV cache grows with seq_len, so a long context can be cut down for the browser
    return (config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"],
            config["num_attention_heads"], config.get("num_key_value_heads", config["num_attention_heads"]),
            vocab_size if shared_classifier else -vocab_size, min(config["max_position_embeddings"], max_seq_len))


def transformed(values, transform, head_size):
    """What a tensor of the Hugging Face checkpoint becomes in the checkpoint this engine reads.

    None: nothing (the rows go straight through, as they arrive). ("permute", heads): the head interleaving of
    wq and wk. ("transpose",): GPT-2 stores its matrices the other way round (Conv1D). ("part", i, n): one of
    the n stacked matrices of GPT-2's c_attn, transposed with it; ("row", i, n) the same for its bias.
    """
    if transform is None:
        return values
    if transform[0] == "permute":
        return permute_heads(values, transform[1], head_size)
    if transform[0] == "transpose":
        return values.T
    if transform[0] == "neox":
        # query_key_value holds (heads, 3, head_size, dim) or (heads, 3, head_size): take one of the three, and
        # interleave the halves of the part that RoPE rotates (Hugging Face stores it as rotate_half does)
        index, heads, rot = transform[1], transform[2], transform[3]
        taken = values.reshape(heads, 3, values.shape[0] // heads // 3, -1)[:, index]
        if rot:
            rotated = taken[:, :rot].reshape(heads, 2, rot // 2, -1).transpose(0, 2, 1, 3).reshape(heads, rot, -1)
            taken = np.concatenate([rotated, taken[:, rot:]], axis=1)
        return taken.reshape(-1, values.shape[-1]) if values.ndim > 1 else taken.reshape(-1)
    if transform[0] == "part":
        index, parts = transform[1], transform[2]
        width = values.shape[1] // parts
        return values[:, index * width:(index + 1) * width].T
    if transform[0] == "row":
        index, parts = transform[1], transform[2]
        length = values.shape[0] // parts
        return values[index * length:(index + 1) * length]
    # a name nobody wrote must not quietly take a slice of rows (T77)
    raise ValueError(f"there is no transform called {transform[0]!r}")


def source_shape(shape, transform):
    """The shape the Hugging Face tensor must have to become a tensor of this shape."""
    if transform is None or transform[0] == "permute":
        return tuple(shape)
    if transform[0] == "neox":  # one of the three stacked parts, and the rows of all three are one tensor
        return (shape[0] * 3, shape[1]) if len(shape) > 1 else (shape[0] * 3,)
    if transform[0] == "transpose":
        return tuple(reversed(shape))
    if transform[0] == "part":
        return (shape[1], shape[0] * transform[2])
    return (shape[0] * transform[2],)


def permute_heads(w, heads, head_size):
    # Hugging Face stores each head of wq/wk as [first halves, second halves] (rotate_half);
    # llama2.c rotates adjacent pairs, so interleave the two halves again. A bias is a vector of the same rows,
    # and -1 as the last dimension lets one line do both.
    if w.shape[0] != heads * head_size:
        # a head of another size reshapes without complaint and turns rows across heads (the review of T124 found
        # half the rows of a Qwen3 0.6B's wq moved so)
        raise ValueError(f"{w.shape[0]} rows are not {heads} heads of {head_size}.")
    return w.reshape(heads, 2, head_size // 2, -1).transpose(0, 2, 1, 3).reshape(w.shape)


def gpt2_prefix(source):
    """openai-community/gpt2 publishes its tensors as wte.weight and h.0...., other GPT-2 models put
    transformer. in front of them. Both are the same model."""
    return "" if "wte.weight" in source else "transformer."


def has_bias(source):
    """Whether this checkpoint has the q, k and v biases of Qwen2 (o and the FFN never have one)."""
    return "model.layers.0.self_attn.q_proj.bias" in source


def has_qk_norm(source):
    """Whether this checkpoint normalizes every head of q and k before RoPE (Qwen3, T124)."""
    return "model.layers.0.self_attn.q_norm.weight" in source


def checkpoint_form(config, source):
    """The form of the checkpoint converted from this config.json and source (llama2_numpy.FORM): what its file will
    not say. head_dim is 0 where the heads fill dim exactly, the way the engine reads a form without one (T144: not
    where dim // heads is the head's size, which a dim that heads do not divide would pass with narrower heads)."""
    config = normalize(config)
    size = head_size(config)
    return {"bias": has_bias(source), "arch": architecture(config), "qk_norm": has_qk_norm(source),
            "head_dim": 0 if size * config["num_attention_heads"] == config["hidden_size"] else size}


def conversion_plan(header, form=None, prefix="transformer.", rotary=0):
    """For every tensor of layout(): the tensors of the Hugging Face checkpoint it is made of, in order, as
    (name, transform); None instead of a list stands for a RoPE table. And the shapes of layout()."""
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header
    form = form_of(form)
    arch, shapes = form["arch"], [shape for shape, _ in layout(*header, **form)]

    if arch == "neox":
        rot = rotary  # how many of each head RoPE turns, from the config
        def h(name, transform=None):
            return [(f"gpt_neox.layers.{layer}.{name}", transform) for layer in range(n_layers)]

        # only q and k are rotated, so only they are interleaved; v is taken as it is
        fused = lambda i: [(f"gpt_neox.layers.{layer}.attention.query_key_value.weight",
                            ("neox", i, n_heads, rot if i < 2 else 0)) for layer in range(n_layers)]
        fused_bias = lambda i: [(f"gpt_neox.layers.{layer}.attention.query_key_value.bias",
                                 ("neox", i, n_heads, rot if i < 2 else 0)) for layer in range(n_layers)]
        plan = [[("gpt_neox.embed_in.weight", None)], None, None,
                h("input_layernorm.weight"), h("input_layernorm.bias"),
                fused(0), fused(1), fused(2),
                fused_bias(0), fused_bias(1), fused_bias(2),
                h("attention.dense.weight"), h("attention.dense.bias"),
                h("post_attention_layernorm.weight"), h("post_attention_layernorm.bias"),
                h("mlp.dense_h_to_4h.weight"), h("mlp.dense_h_to_4h.bias"),
                h("mlp.dense_4h_to_h.weight"), h("mlp.dense_4h_to_h.bias"),
                [("gpt_neox.final_layer_norm.weight", None)], [("gpt_neox.final_layer_norm.bias", None)]]
        if vocab_size < 0:
            plan.append([("embed_out.weight", None)])
        return plan, shapes

    if arch == "gpt2":
        def h(name, transform=None):
            return [(f"{prefix}h.{layer}.{name}", transform) for layer in range(n_layers)]

        third = lambda i: ("part", i, 3)
        plan = [[(prefix + "wte.weight", None)], [(prefix + "wpe.weight", None)],
                h("ln_1.weight"), h("ln_1.bias"),
                h("attn.c_attn.weight", third(0)), h("attn.c_attn.weight", third(1)), h("attn.c_attn.weight", third(2)),
                h("attn.c_attn.bias", ("row", 0, 3)), h("attn.c_attn.bias", ("row", 1, 3)), h("attn.c_attn.bias", ("row", 2, 3)),
                h("attn.c_proj.weight", ("transpose",)), h("attn.c_proj.bias"),
                h("ln_2.weight"), h("ln_2.bias"),
                h("mlp.c_fc.weight", ("transpose",)), h("mlp.c_fc.bias"),
                h("mlp.c_proj.weight", ("transpose",)), h("mlp.c_proj.bias"),
                [(prefix + "ln_f.weight", None)], [(prefix + "ln_f.bias", None)]]
        if vocab_size < 0:
            plan.append([("lm_head.weight", None)])
        return plan, shapes

    def layers(name, transform=None, what="weight"):
        return [(f"model.layers.{layer}.{name}.{what}", transform) for layer in range(n_layers)]

    plan = [[("model.embed_tokens.weight", None)], layers("input_layernorm"),
            layers("self_attn.q_proj", ("permute", n_heads)), layers("self_attn.k_proj", ("permute", n_kv_heads)),
            layers("self_attn.v_proj"),
            layers("self_attn.o_proj"), layers("post_attention_layernorm"),
            layers("mlp.gate_proj"), layers("mlp.down_proj"), layers("mlp.up_proj"), [("model.norm.weight", None)],
            None, None]
    if vocab_size < 0:
        plan.append([("lm_head.weight", None)])
    if form["bias"]:
        plan += [layers("self_attn.q_proj", ("permute", n_heads), "bias"),
                 layers("self_attn.k_proj", ("permute", n_kv_heads), "bias"), layers("self_attn.v_proj", None, "bias")]
    if form["qk_norm"]:
        # one weight for every head, over the rows of a head: interleaved like the rows it multiplies
        plan += [layers("self_attn.q_norm", ("permute", 1)), layers("self_attn.k_norm", ("permute", 1))]
    return plan, shapes


def rope_table(config, header, which):
    """The cos (which = 0) or sin (1) table of the legacy format, for float32 and float16 checkpoints.

    GPT-NeoX rotates only rotary_pct of each head, and the angles follow that width. The table keeps the shape
    the layout gives it (head_size // 2 columns); the columns past the rotated part are never read.
    """
    size, seq_len = head_size(config), header[6]
    width = rotary_dim(config) if architecture(config) == "neox" else size
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    frequencies = rope_frequencies(width, config.get("rope_theta", 10000.0), config.get("rope_scaling"))
    table = (np.cos if which == 0 else np.sin)(positions * frequencies)
    if width == size:
        return table
    full = np.zeros((seq_len, size // 2), dtype=np.float64)
    full[:, :width // 2] = table
    return full


def convert_weights(source, config, dtype, max_seq_len, out, progress=None, quantize_rows=None):
    """Fill out, a writable buffer of checkpoint_size() bytes, from source (Safetensors or Arrays).

    progress(values done, values in all) is called after every piece. quantize_rows: see Writer.
    """
    for done, total in convert_pieces(source, config, dtype, max_seq_len, out, quantize_rows):
        if progress:
            progress(done, total)
    return checkpoint_header(config, source, max_seq_len)


def convert_pieces(source, config, dtype, max_seq_len, out, quantize_rows=None):
    """convert_weights() as a generator that yields (values done, values in all) after every piece: whoever drives
    it can show the progress, let other work in between, and stop half way (the worker of the page does all three)."""
    config = normalize(config)
    check_config(config)
    header = checkpoint_header(config, source, max_seq_len)
    # the size the heads of wq and wk are interleaved by: another size turns rows across heads without a word
    size = head_size(config)
    form = checkpoint_form(config, source)
    writer = Writer(out, header, dtype, form, quantize_rows=quantize_rows)

    plan, shapes = conversion_plan(header, form, gpt2_prefix(source), rotary_dim(config) if form["arch"] == "neox" else 0)
    total, done = sum(int(np.prod(shape)) for shape in shapes), 0

    for index, (parts, shape) in enumerate(zip(plan, shapes)):
        if parts is None:
            # the RoPE tables (left out of an int8 checkpoint): cos, then sin
            writer.write(index, 0, rope_table(config, header, plan[:index].count(None)))
            done += int(np.prod(shape))
            yield done, total
            continue
        first = 0
        for name, transform in parts:
            found = source.shape(name) if name in source else None
            expected = source_shape(shape[1:] if len(parts) > 1 else shape, transform)
            if found != expected:
                raise ValueError(f"This model cannot be converted: {name} is {found or 'missing'}, not {expected}.")
            rows = found[0] if len(found) > 1 else 1
            row = int(np.prod(found)) // rows
            # a transform needs the whole tensor (a small one); everything else goes piece by piece
            step = rows if transform or len(found) == 1 else max(1, PIECE // row)
            for start in range(0, rows, step):
                stop = min(start + step, rows)
                values = source.rows(name, 0, found[0]) if len(found) == 1 else source.rows(name, start, stop)
                values = transformed(values, transform, size)
                writer.write(index, first, values)
                first += values.size
                done += values.size
                yield done, total
        assert first == int(np.prod(shape)), name


class Stream:
    """The same conversion in the order of the file: feed() takes the bytes of a .safetensors file from its beginning
    to its end, in chunks of any size, and every tensor goes to its place in the checkpoint as soon as its rows are
    there. For a download: reading in the order of the output would mean hundreds of range requests, a second each.

    header: the JSON of the file (its first 8 bytes say how long it is), base: where the tensors begin, start: the
    position in the file of the first byte that feed() will get. out: a buffer of checkpoint_size() bytes, or None
    to have one made (self.out). sink and quantize_rows: see Writer. bfloat16: the widening of bfloat16 on the SIMD
    kernels (llama2_numpy.kernel_widener, T123), the same float32 as this file's bfloat16(). q8_0: the widening of
    GGUF's Q8_0 on the kernels (llama2_numpy.kernel_q8_0, T136), the same float32 as this file's q8_0().
    """

    def __init__(self, header, base, config, dtype, max_seq_len, out=None, start=0, sink=None, quantize_rows=None,
                 bfloat16=None, q8_0=None):
        config = normalize(config)
        self.bfloat16, self.q8_0 = bfloat16, q8_0
        check_config(config)
        self.tensors = {name: info for name, info in header.items() if name != "__metadata__"}
        self.header = checkpoint_header(config, self, max_seq_len)
        self.head_size = head_size(config)
        # what lays out the checkpoint and sizes the forward pass that the header does not say (T115, T124, T144):
        # the options say it, and the worker's footprint() reads it
        self.form = checkpoint_form(config, self)
        if callable(dtype):
            # T115: chosen once the header is known, from the size each quantized dtype would take (the worker's
            # automatic choice: int8 where the forward pass fits a 32-bit memory, six bits where it does not)
            sizes = {name: self.size(name) for name in QUANTIZED}
            dtype = str(dtype(list(self.header), self.form, sizes))
        check_dtype(dtype)
        self.dtype = dtype_name(dtype)
        if out is None and sink is None:
            out = bytearray(self.size(dtype))
        self.out = out  # None when the checkpoint goes to sink
        self.writer = Writer(self.out, self.header, dtype, self.form, sink=sink, quantize_rows=quantize_rows)
        plan, shapes = conversion_plan(self.header, self.form, gpt2_prefix(self),
                                       rotary_dim(config) if self.form["arch"] == "neox" else 0)
        self.total, self.done = sum(int(np.prod(shape)) for shape in shapes), 0
        wanted = {}
        for index, (parts, shape) in enumerate(zip(plan, shapes)):
            if parts is None:
                self.writer.write(index, 0, rope_table(config, self.header, plan[:index].count(None)))
                self.done += int(np.prod(shape))
                continue
            first = 0
            for name, transform in parts:
                found = tuple(self.tensors[name]["shape"]) if name in self.tensors else None
                expected = source_shape(shape[1:] if len(parts) > 1 else shape, transform)
                if found != expected:
                    raise ValueError(f"This model cannot be converted: {name} is {found or 'missing'}, not {expected}.")
                if self.tensors[name]["dtype"] not in READERS:
                    raise ValueError(f"{name} is stored as {self.tensors[name]['dtype']}: only float32, float16 and bfloat16 are supported.")
                # GPT-2's c_attn holds q, k and v in one matrix, so one tensor of the file can feed several
                wanted.setdefault(name, []).append((index, first, transform))
                first += int(np.prod(shape[1:] if len(parts) > 1 else shape))
        # what to do with each stretch of the file, in the order of the file
        self.steps = []
        for name, info in sorted(self.tensors.items(), key=lambda item: item[1]["data_offsets"][0]):
            begin, end = info["data_offsets"]
            # T136: a table to check as it passes, not to convert (gguf_weights)
            target = wanted.get(name) or ("check" if info.get("rope_freqs") else None)
            self.steps.append((base + begin, base + end, name, target))
        self.position, self.step, self.pending, self.first = start, 0, bytearray(), 0
        self.config = config

    def __contains__(self, name):  # what checkpoint_header() asks
        return name in self.tensors

    def size(self, dtype):
        """The bytes of the checkpoint in that dtype."""
        return checkpoint_size(self.header, dtype, self.form)

    def feed(self, data):
        """data: the next bytes of the file. Returns (values done, values in all)."""
        data = memoryview(data.to_py() if hasattr(data, "to_py") else data).cast("B")
        offset = 0
        while offset < len(data) and self.step < len(self.steps):
            begin, end, name, target = self.steps[self.step]
            here = self.position + offset
            if here < begin:  # the JSON header, padding, or a tensor nobody needs
                offset += min(begin - here, len(data) - offset)
                continue
            take = min(end - here, len(data) - offset)
            if target == "check":
                self.pending += data[offset:offset + take]
                if here + take == end:
                    rope_freqs_agree(np.frombuffer(bytes(self.pending), dtype=np.float32), self.config)
            elif target is not None:
                self.pending += data[offset:offset + take]
                self.convert(name, target, last=here + take == end)
            offset += take
            if here + take == end:
                self.step, self.pending, self.first = self.step + 1, bytearray(), 0
        self.position += len(data)
        return self.done, self.total

    def convert(self, name, targets, last):
        info = self.tensors[name]
        itemsize, reader = READERS[info["dtype"]]
        if info["dtype"] == "BF16" and self.bfloat16 is not None:
            reader = self.bfloat16
        if info["dtype"] == "Q8_0" and self.q8_0 is not None:
            reader = self.q8_0
        shape = tuple(info["shape"])
        # T136's third stage: a GPT-2's Conv1D matrix, which the GGUF holds as (out, in), is read in that shape
        stored = tuple(reversed(shape)) if info.get("transposed") else shape
        row = int((int(np.prod(stored[1:])) if len(stored) > 1 else int(stored[0])) * itemsize)
        # the head permutation needs its whole matrix (a small one); everything else goes row by row, as it comes.
        # A GGUF's tensor held in another order than Hugging Face's is put back whole too
        again = info.get("turned") or info.get("split") or info.get("transposed")
        whole = any(transform for _, _, transform in targets) or len(targets) > 1 or bool(again)
        rows = len(self.pending) // row if not whole or last else 0
        if whole and last:
            rows = stored[0] if len(stored) > 1 else 1  # a vector is one row of its own length
        if rows == 0 or (len(self.pending) < PIECE and not last):
            return
        values = reader(bytes(self.pending[:rows * row]))
        del self.pending[:rows * row]
        if len(stored) > 1:
            values = values.reshape(rows, *stored[1:])
        if info.get("transposed"):
            values = values.T  # back to (in, out), which the plan transposes as it does a safetensors' own
        if info.get("turned"):
            # a GGUF of a Llama holds q and k turned already (llama.cpp's convert does what permute_heads does):
            # back to Hugging Face's order, so that the plan below turns them once, like everything else
            values = unturned(values, info["turned"])
        if info.get("split"):
            values = unsplit(values, info["split"])
        for index, first, transform in targets:
            out = transformed(values, transform, self.head_size)
            self.writer.write(index, first + self.first, out)
            self.done += out.size
        self.first += 0 if whole else values.size

    def finish(self):
        if self.step < len(self.steps) or self.done != self.total:
            raise ValueError("The file ended before all of its tensors were read.")
        return self.header


def unturned(w, heads):
    """The inverse of permute_heads: adjacent pairs of each head back to [first halves, second halves]."""
    rows = w.shape[0] // heads
    return w.reshape(heads, rows // 2, 2, -1).transpose(0, 2, 1, 3).reshape(w.shape)


def unsplit(w, heads):
    """GPT-NeoX's query_key_value (or its bias) as llama.cpp stores it, [all of q; all of k; all of v], back to
    Hugging Face's order, q, k and v of the first head, then of the second, ... (T136's third stage)."""
    return w.reshape(3, heads, w.shape[0] // 3 // heads, -1).swapaxes(0, 1).reshape(w.shape)


# ------------------------------------------------------------------------------------------------- GGUF (T74)
# A GGUF file holds what config.json, tokenizer.json and model.safetensors hold, in one. Only what a Q8_0 or F16
# Llama, Qwen2, Qwen3, GPT-2 or GPT-NeoX needs is read; tests/gguf_check.py is the separate reference this is held to.
class Incomplete(Exception):
    """The GGUF header goes on past the bytes given: fetch more and try again."""


GGUF_VALUES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
GGUF_TENSORS = {0: "F32", 1: "F16", 8: "Q8_0"}  # ggml's types; the K-quants and the rest are refused
# llama.cpp's names of the pre-tokenizers, as the engine knows them (llama2_numpy.pretokenize)
GGUF_PRETOKENIZERS = {"gpt-2": "gpt2", "gpt2": "gpt2", "smollm": "gpt2-digits", "qwen2": "qwen", "llama-bpe": "llama3"}
GGUF_LAYER = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj",
              "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
              "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
GGUF_NAMES = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
              "output.weight": "lm_head.weight"}
# T136's third stage: GPT-2 and GPT-NeoX, by the names of their own safetensors (openai-community/gpt2's, without
# "transformer."). For each architecture: the tensors outside the layers, where a layer's go, and the layer's names
GGUF_ARCHITECTURES = {
    "llama": (GGUF_NAMES, "model.layers.{}.", GGUF_LAYER),
    "qwen2": (GGUF_NAMES, "model.layers.{}.", GGUF_LAYER),
    # T203 (T136's fourth stage): a Qwen3 is a Qwen2 without the biases that normalizes each head of q and k (T124).
    # llama.cpp leaves q, k and the two norms in Hugging Face's order, as a Qwen2's; the head's size is key_length
    "qwen3": (GGUF_NAMES, "model.layers.{}.", {**GGUF_LAYER, "attn_q_norm": "self_attn.q_norm",
                                                 "attn_k_norm": "self_attn.k_norm"}),
    "gpt2": ({"token_embd.weight": "wte.weight", "position_embd.weight": "wpe.weight", "output_norm.weight": "ln_f.weight",
              "output_norm.bias": "ln_f.bias", "output.weight": "lm_head.weight"}, "h.{}.",
             {"attn_norm": "ln_1", "attn_qkv": "attn.c_attn", "attn_output": "attn.c_proj", "ffn_norm": "ln_2",
              "ffn_up": "mlp.c_fc", "ffn_down": "mlp.c_proj"}),
    "gptneox": ({"token_embd.weight": "gpt_neox.embed_in.weight", "output_norm.weight": "gpt_neox.final_layer_norm.weight",
                 "output_norm.bias": "gpt_neox.final_layer_norm.bias", "output.weight": "embed_out.weight"},
                "gpt_neox.layers.{}.",
                {"attn_norm": "input_layernorm", "attn_qkv": "attention.query_key_value", "attn_output": "attention.dense",
                 "ffn_norm": "post_attention_layernorm", "ffn_up": "mlp.dense_h_to_4h", "ffn_down": "mlp.dense_4h_to_h"}),
}


def gguf_read(data):
    """(metadata, tensors, base) from the first bytes of a GGUF file: tensors maps each name to its ggml type,
    its shape (outermost first, as NumPy has it) and its offset from base, where the data begins."""
    data = memoryview(data.to_py() if hasattr(data, "to_py") else data).cast("B")
    at = 0

    def take(fmt):
        nonlocal at
        size = struct.calcsize(fmt)
        if at + size > len(data):
            raise Incomplete()
        (value,) = struct.unpack_from(fmt, data, at)
        at += size
        return value

    def string():
        nonlocal at
        size = take("<Q")
        if at + size > len(data):
            raise Incomplete()
        at += size
        return bytes(data[at - size:at]).decode("utf-8", errors="replace")

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            item, count = take("<I"), take("<Q")
            return [value(item) for _ in range(count)]
        if kind not in GGUF_VALUES:
            raise ValueError(f"This GGUF file has a value of type {kind}, which is not in the format.")
        return take(GGUF_VALUES[kind])

    if len(data) >= 4 and bytes(data[:4]) != b"GGUF":
        raise ValueError("This is not a GGUF file.")
    take("<I")
    version = take("<I")
    if version not in (2, 3):
        raise ValueError(f"This GGUF file is of version {version}; only 2 and 3 are supported.")
    count, entries = take("<Q"), take("<Q")
    metadata = {}
    for _ in range(entries):
        key = string()
        metadata[key] = value(take("<I"))
    tensors = {}
    for _ in range(count):
        name = string()
        dims = [take("<Q") for _ in range(take("<I"))]
        tensors[name] = {"type": take("<I"), "shape": list(reversed(dims)), "offset": take("<Q")}
    alignment = metadata.get("general.alignment", 32)
    return metadata, tensors, (at + alignment - 1) // alignment * alignment


def gguf_model(metadata, tensors, base, rope_freqs=False):
    """The safetensors-like header (Hugging Face's names, offsets from base) and the config.json of a GGUF.
    rope_freqs: keep llama.cpp's table of Llama 3's RoPE scaling in the header, to be checked against the original's
    rope_scaling as it streams past (gguf_weights, T136), instead of refusing it."""
    arch = metadata.get("general.architecture")
    if arch not in GGUF_ARCHITECTURES:
        raise ValueError(f"This GGUF holds a {arch}: only Llama, Qwen2, Qwen3, GPT-2 and GPT-NeoX ones are supported.")
    key = lambda name, default=None: metadata.get(f"{arch}.{name}", default)
    common = {"vocab_size": tensors["token_embd.weight"]["shape"][0] if "token_embd.weight" in tensors else None,
              "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id", 1),
              "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id", 2)}
    heads = key("attention.head_count")
    if arch == "gpt2":
        # T136's third stage: config.json's own spelling, which normalize() reads. GPT-2 always shares its classifier
        # with the embedding: llama.cpp writes a copy of it as output.weight, which the conversion leaves unread
        config = {"model_type": "gpt2", "n_embd": key("embedding_length"), "n_inner": key("feed_forward_length"),
                  "n_layer": key("block_count"), "n_head": heads, "n_positions": key("context_length"),
                  "layer_norm_epsilon": key("attention.layer_norm_epsilon"), "tie_word_embeddings": True, **common}
    elif arch == "gptneox":
        dim = key("embedding_length")
        config = {"model_type": "gpt_neox", "hidden_size": dim, "intermediate_size": key("feed_forward_length"),
                  "num_hidden_layers": key("block_count"), "num_attention_heads": heads,
                  "max_position_embeddings": key("context_length"),
                  "rotary_emb_base": float(key("rope.freq_base", 10000.0)),
                  # llama.cpp says the rotated part as a number of values, config.json as a share of the head
                  "rotary_pct": key("rope.dimension_count", 0) / (dim // heads) if dim and heads else None,
                  "use_parallel_residual": bool(key("use_parallel_residual", True)),
                  "layer_norm_eps": key("attention.layer_norm_epsilon"), "hidden_act": "gelu",
                  "tie_word_embeddings": "output.weight" not in tensors, **common}
    else:
        config = {"model_type": arch, "hidden_size": key("embedding_length"), "intermediate_size": key("feed_forward_length"),
                  "num_hidden_layers": key("block_count"), "num_attention_heads": heads,
                  "num_key_value_heads": key("attention.head_count_kv", heads),
                  "max_position_embeddings": key("context_length"), "rope_theta": float(key("rope.freq_base", 10000.0)),
                  "tie_word_embeddings": "output.weight" not in tensors, "hidden_act": "silu",
                  # a head of another size than dim / heads (T124): llama.cpp says it as the length of a key
                  "head_dim": key("attention.key_length"), "rms_norm_eps": key("attention.layer_norm_rms_epsilon"),
                  **common}
        if key("rope.scaling.type", "none") not in ("none", None):
            config["rope_scaling"] = {"type": key("rope.scaling.type"), "factor": key("rope.scaling.factor", 1.0)}
    header = {}
    if "rope_freqs.weight" in tensors:
        # llama.cpp writes Llama 3's RoPE scaling as a table of divisors instead of the rope_scaling of config.json
        if not rope_freqs:
            raise ValueError("This GGUF scales its RoPE with a rope_freqs table, which the engine does not read.")
        info = tensors["rope_freqs.weight"]
        if info["type"] != 0:
            raise ValueError(f"rope_freqs.weight is stored as ggml type {info['type']}, not F32.")
        size = 4 * int(np.prod(info["shape"]))
        header["rope_freqs.weight"] = {"dtype": "F32", "shape": info["shape"], "rope_freqs": True,
                                       "data_offsets": [info["offset"], info["offset"] + size]}
    names, layer, layers = GGUF_ARCHITECTURES[arch]
    turns = {"attn_q": heads, "attn_k": config.get("num_key_value_heads")}
    for name, info in tensors.items():
        if info["type"] not in GGUF_TENSORS:
            raise ValueError(f"{name} is stored as ggml type {info['type']}: only F32, F16 and Q8_0 GGUF files are "
                             f"supported (not the K-quants).")
        parts = name.split(".")
        if name in names:
            target = names[name]
        elif len(parts) == 4 and parts[0] == "blk" and parts[2] in layers:
            target = f"{layer.format(parts[1])}{layers[parts[2]]}.{parts[3]}"
        else:
            continue  # nothing the engine reads
        dtype = GGUF_TENSORS[info["type"]]
        if dtype == "Q8_0" and info["shape"][-1] % 32:
            # ggml itself requires it; a file that breaks it would be read at the wrong offsets and write nonsense
            raise ValueError(f"{name} is Q8_0 with rows of {info['shape'][-1]}, which is not a multiple of 32.")
        size = int(int(np.prod(info["shape"])) * READERS[dtype][0])
        entry = {"dtype": dtype, "shape": info["shape"], "data_offsets": [info["offset"], info["offset"] + size]}
        kind = parts[2] if len(parts) == 4 else None
        if arch == "llama" and kind in turns:
            # llama.cpp turns q and k of a Llama (and their biases) into llama2.c's order; a Qwen2 it leaves alone
            # (it rotates the other way at run time). tests/gguf_check.py found SmolLM2's turned.
            entry["turned"] = turns[kind]
        if arch == "gpt2" and parts[-1] == "weight" and kind in ("attn_qkv", "attn_output", "ffn_up", "ffn_down"):
            # GPT-2's matrices are Conv1D, (in, out): llama.cpp stores them the other way round, as every other
            # model's. Back to Hugging Face's, so that the plan transposes them once, as it does a safetensors' own
            entry["shape"], entry["transposed"] = list(reversed(info["shape"])), True
        if arch == "gptneox" and kind == "attn_qkv":
            # GPT-NeoX's query_key_value holds q, k and v of every head in turn; llama.cpp stores all of q, then k,
            # then v (the matrix and its bias). Back to Hugging Face's order, like the turned q and k of a Llama
            entry["split"] = heads
        header[target] = entry
    return header, config


def gguf_weights(head, config):
    """T136's second stage: the weights of a GGUF with the vocabulary and config.json of the original repository,
    for the GGUF's own vocabulary is of no use there (a sentencepiece one says neither Unigram or BPE nor its
    normalization; llm-jp's scores are all -1000). head: the GGUF's beginning, as far as the tensors' data (Incomplete
    when it is not), config: the text of the original's config.json. Returns the safetensors-like header (JSON text)
    and where the tensors begin, which Conversion() then takes as it takes a safetensors file's."""
    metadata, tensors, base = gguf_read(head)
    header, own = gguf_model(metadata, tensors, base, rope_freqs=True)
    try:
        original = json.loads(config)
    except ValueError:
        raise ValueError("config.json is not JSON.") from None
    if not isinstance(original, dict):
        raise ValueError("config.json is not the configuration of a model.")
    gguf_agrees(normalize(own), normalize(original))
    return json.dumps(header), base


def gguf_agrees(own, config):
    """ValueError unless a GGUF (own: what gguf_model() read of it) holds the model config.json describes. The
    sizes of the tensors the conversion checks anyway (Stream); these are what the sizes do not show: heads and
    key-value heads of the same product, the number of layers (a GGUF of more layers than config.json says went
    through cut to that many: Stream reads the layers the header asks for), a classifier that would silently be the
    embedding (Stream shares it where lm_head is missing), and the numbers that are no tensor. The context is not
    compared: a sliding window cuts it (RakutenAI 2.0 mini: 131072 in the GGUF, 8192 as normalize() cuts it).
    Both are normalize()d. GPT-NeoX's (T136's third stage): also how much of each head turns and whether the two
    branches run in parallel, which the options say (no tensor does)."""
    f32 = lambda value: float(np.float32(value))
    heads = config.get("num_attention_heads")
    pairs = [("architecture", own["model_type"], config.get("model_type")),
             ("number of layers", own["num_hidden_layers"], config.get("num_hidden_layers")),
             ("number of heads", own["num_attention_heads"], heads),
             ("number of key-value heads", own.get("num_key_value_heads", own["num_attention_heads"]),
              config.get("num_key_value_heads", heads)),
             ("RoPE theta", f32(own.get("rope_theta", 10000.0)), f32(config.get("rope_theta", 10000.0)))]
    if own.get("head_dim") and config.get("hidden_size") and heads:
        pairs.append(("size of a head", own["head_dim"], head_size(config)))
    if own.get("rms_norm_eps") is not None and config.get("rms_norm_eps") is not None:
        pairs.append(("RMSNorm epsilon", f32(own["rms_norm_eps"]), f32(config["rms_norm_eps"])))
    if architecture(own) in ("gpt2", "neox"):
        # transformers' default where config.json says none (the engine's LayerNorm takes 1e-5 whatever it says)
        layer_norm_eps = lambda c: c.get("layer_norm_eps", c.get("layer_norm_epsilon", 1e-5))
        pairs.append(("LayerNorm epsilon", f32(layer_norm_eps(own)), f32(layer_norm_eps(config))))
    if architecture(own) == "neox" and architecture(config) == "neox":
        pairs += [("number of rotated values of a head", rotary_dim(own), rotary_dim(config)),
                  ("parallel residual", own.get("use_parallel_residual", True), config.get("use_parallel_residual", True))]
    for what, here, there in pairs:
        if here != there:
            raise ValueError(f"This GGUF does not belong with the original's config.json: its {what} is {here} here "
                             f"and {there} there.")
    if own["tie_word_embeddings"] and not config.get("tie_word_embeddings", False):
        raise ValueError("This GGUF does not belong with the original's config.json: the original has a classifier of "
                         "its own, this GGUF has none.")


def rope_freqs_agree(table, config):
    """ValueError unless llama.cpp's rope_freqs (a divisor of each pair's angle) is what the original's rope_scaling
    makes: the engine makes its RoPE tables from rope_scaling (rope_frequencies), and the table is not used."""
    width = head_size(config)
    theta = config.get("rope_theta", 10000.0)
    expected = rope_frequencies(width, theta) / rope_frequencies(width, theta, config.get("rope_scaling"))
    table = np.asarray(table, dtype=np.float64)
    worst = float(np.max(np.abs(table - expected) / expected)) if table.shape == expected.shape else float("inf")
    if not worst <= 1e-5:
        raise ValueError(f"This GGUF scales its RoPE otherwise than the original's rope_scaling says (by {worst:.1e} "
                         f"at most).")


def gguf_tokenizer(metadata, vocab_size):
    """tokenizer.bin, the engine's options, the tokenizer_config and the special tokens of a GGUF's byte-level BPE
    vocabulary."""
    if metadata.get("tokenizer.ggml.model") != "gpt2":
        raise ValueError(f"This GGUF has a {metadata.get('tokenizer.ggml.model')} vocabulary: only byte-level BPE "
                         f"ones (gpt2) are supported.")
    pre = metadata.get("tokenizer.ggml.pre", "gpt-2")
    if pre not in GGUF_PRETOKENIZERS:
        raise ValueError(f"This GGUF splits text as {pre}, which the engine does not know.")
    tokens, kinds = metadata["tokenizer.ggml.tokens"], metadata.get("tokenizer.ggml.token_type", [])
    ranks = {}
    for rank, merge in enumerate(metadata.get("tokenizer.ggml.merges", [])):
        left, right = merge.split(" ")
        ranks.setdefault(left + right, -float(rank))
    # what tokenizer.json calls special: llama.cpp's control tokens (type 3)
    pieces = [(text, ranks.get(text, UNMATCHABLE), text in ranks and (kinds[id] if id < len(kinds) else 1) != 3)
              for id, text in enumerate(tokens)]
    # Qwen's tokenizer.json normalizes to NFC, which a GGUF does not say: the page's safetensors path does it
    options = {"tokenizer_kind": "bytebpe", "nfkc": False, "nfc": pre == "qwen2", "pretokenizer": GGUF_PRETOKENIZERS[pre],
               "ignore_merges": pre == "llama-bpe"}
    special = lambda key: tokens[metadata[key]] if isinstance(metadata.get(key), int) and metadata[key] < len(tokens) else ""
    config = {"chat_template": metadata.get("tokenizer.chat_template"), "bos_token": special("tokenizer.ggml.bos_token_id"),
              "eos_token": special("tokenizer.ggml.eos_token_id")}
    controls = [text for id, text in enumerate(tokens) if id < len(kinds) and kinds[id] == 3]
    return tokenizer_bin(pieces, vocab_size), options, config, controls


# ---------------------------------------------------------------------------------------- the tokenizer
UNMATCHABLE = -1e9  # control, unknown and byte pieces must never match user text: llama2_numpy.py skips such scores


def tokenizer_bin(pieces, vocab_size):
    """llama2.c's tokenizer.bin from (text, score, matchable) pieces."""
    rows = [(score if matchable else UNMATCHABLE, text.replace("▁", " ").encode("utf-8")) for text, score, matchable in pieces]
    if len(rows) > vocab_size:
        raise ValueError(f"The tokenizer has {len(rows)} pieces, but the model has a vocabulary of {vocab_size}.")
    # A model can have a few more embedding rows than the tokenizer has pieces (padding to a round number). Many
    # more means the tokenizer of another model, which would convert fine and then write nonsense.
    if len(rows) < 0.9 * vocab_size:
        raise ValueError(f"The tokenizer has {len(rows)} pieces, but the model has a vocabulary of {vocab_size}: "
                         f"they do not belong together.")
    rows += [(UNMATCHABLE, b"")] * (vocab_size - len(rows))
    out = [struct.pack("<i", max(len(text) for _, text in rows))]
    out += [struct.pack("<fi", score, len(text)) + text for score, text in rows]
    return b"".join(out)


def tokenizer_kind_of(tokenizer):
    """What kind of model this tokenizer.json holds. The oldest ones (GPT-2's own, version 1.0) have no "type",
    and are told apart by what they carry: merges for a BPE, a list of (piece, score) for a Unigram."""
    model = tokenizer["model"]
    if "type" in model:
        return model["type"]
    return "BPE" if "merges" in model else "Unigram"


def tokenizer_json_pieces(tokenizer):
    kind = tokenizer_kind_of(tokenizer)
    if kind == "BPE":
        yield from tokenizer_json_bpe_pieces(tokenizer)
        return
    if kind != "Unigram":
        raise ValueError(f"This tokenizer.json is a {kind} model: only Unigram and byte-level BPE ones are "
                         f"supported (or a sentencepiece tokenizer.model).")
    special = {token["content"] for token in tokenizer["added_tokens"] if token["special"]}
    for id, (text, score) in enumerate(tokenizer["model"]["vocab"]):
        is_byte = len(text) == 6 and text.startswith("<0x") and text.endswith(">")
        yield text, score, not (is_byte or text in special or id == tokenizer["model"].get("unk_id"))


def tokenizer_json_bpe_pieces(tokenizer):
    """Hugging Face's byte-level BPE (GPT-2, SmolLM2, Qwen). The pieces are written in the byte <-> character
    table, and the score is minus the rank of the merge that makes the piece: the engine merges the best-scoring
    pair, which is then the same as applying the merge with the lowest rank. A piece no merge makes (a single
    character, an added token) never starts a merge, so it is not matchable."""
    model = tokenizer["model"]
    ranks = {}
    for rank, merge in enumerate(model["merges"]):
        left, right = merge if isinstance(merge, list) else merge.split(" ")
        ranks.setdefault(left + right, -float(rank))
    texts = {id: text for text, id in model["vocab"].items()}
    for token in tokenizer["added_tokens"]:
        texts.setdefault(token["id"], token["content"])
    special = {token["content"] for token in tokenizer["added_tokens"] if token["special"]}
    for id in range(max(texts) + 1):
        text = texts.get(id)
        if text is None:
            raise ValueError(f"This tokenizer.json has no piece with id {id}.")
        yield text, ranks.get(text, UNMATCHABLE), text in ranks and text not in special


def tokenizer_json_options(tokenizer):
    """What the engine has to know about this tokenizer: Llama(tokenizer_kind=, nfkc=, nfc=, pretokenizer=)."""
    normalizers = json.dumps(tokenizer.get("normalizer") or {})
    # "Precompiled" is sentencepiece's character map, nmt_nfkc in practice
    nfkc = '"NFKC"' in normalizers or '"Precompiled"' in normalizers
    if tokenizer_kind_of(tokenizer) != "BPE":
        return {"tokenizer_kind": "unigram", "nfkc": nfkc}
    return {"tokenizer_kind": "bytebpe", "nfkc": nfkc, "nfc": '"NFC"' in normalizers,
            "pretokenizer": pretokenizer_name(tokenizer.get("pre_tokenizer")),
            "ignore_merges": bool(tokenizer["model"].get("ignore_merges"))}


# The engine writes these out by hand (llama2_numpy.pretokenize), so only the patterns it knows are accepted.
PRETOKENIZERS = {
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "llama3",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+": "qwen",
}


def pretokenizer_name(spec):
    steps = spec.get("pretokenizers", [spec]) if spec else []
    kinds = [step["type"] for step in steps]
    patterns = [step["pattern"]["Regex"] for step in steps if step["type"] == "Split"]
    if patterns:
        if len(patterns) > 1 or patterns[0] not in PRETOKENIZERS:
            raise ValueError(f"This tokenizer.json splits text in a way the engine does not know: {patterns}")
        return PRETOKENIZERS[patterns[0]]
    if "ByteLevel" not in kinds or not all(kind in ("ByteLevel", "Digits") for kind in kinds):
        raise ValueError(f"This tokenizer.json splits text in a way the engine does not know: {kinds}")
    if not all(step.get("use_regex", True) for step in steps if step["type"] == "ByteLevel"):
        raise ValueError("This tokenizer.json has a ByteLevel pre-tokenizer without its regex, which the engine "
                         "does not know.")
    if any(step["type"] == "ByteLevel" and step.get("add_prefix_space") for step in steps):
        raise ValueError("This tokenizer.json adds a space in front of the text, which the engine does not do.")
    digits = [step for step in steps if step["type"] == "Digits"]
    if digits and not all(step.get("individual_digits") for step in digits):
        raise ValueError("This tokenizer.json groups digits in a way the engine does not know.")
    return "gpt2-digits" if digits else "gpt2"


def protobuf_fields(data):
    """Yield (field number, value) of one protobuf message; nested messages come back as bytes."""
    i = 0

    def varint():
        nonlocal i
        value = shift = 0
        while True:
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    while i < len(data):
        key = varint()
        field, wire_type = key >> 3, key & 7
        if wire_type == 0:
            yield field, varint()
        elif wire_type == 1:
            yield field, data[i:i + 8]
            i += 8
        elif wire_type == 2:
            size = varint()
            yield field, data[i:i + size]
            i += size
        elif wire_type == 5:
            yield field, data[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")


def sentencepiece_pieces(model):
    """(text, score, matchable) of a sentencepiece model (spiece.model, tokenizer.model), given as bytes."""
    NORMAL, USER_DEFINED = 1, 4
    for field, value in protobuf_fields(model):
        if field == 1:  # ModelProto.pieces
            piece = dict(protobuf_fields(value))
            score = struct.unpack("<f", piece[2])[0] if 2 in piece else 0.0
            yield piece.get(1, b"").decode("utf-8"), score, piece.get(3, NORMAL) in (NORMAL, USER_DEFINED)


def sentencepiece_specials(model):
    """The control pieces of a sentencepiece model (<s>, </s>, sarashina's <|user|>): the special tokens a chat
    template writes between the turns. The engine's search of the vocabulary never finds them, so a template read
    from the model (T127) needs them named; without, "</s>" became four tokens of text."""
    CONTROL = 3
    specials = []
    for field, value in protobuf_fields(model):
        if field == 1:
            piece = dict(protobuf_fields(value))
            if piece.get(3) == CONTROL and piece.get(1):
                specials.append(piece[1].decode("utf-8"))
    return specials


def sentencepiece_options(model):
    """Llama(tokenizer_kind=, nfkc=, nmt=, collapse=, unknown=) from the pieces and the trainer and normalizer specs
    of a sentencepiece model.

    nmt: the normalizer is one of the nmt_ kinds, which make tabs, newlines and a few more spaces and drop the other
    control characters. collapse: remove_extra_whitespaces, runs of spaces made one and the ends trimmed. Both are
    said only where on: rinna's models are nmt_nfkc with it (and no byte pieces to spell a newline with), Llama's and
    Mistral's identity without it, tiny-lm's nfkc without it (the review of T126). unknown: the id of the unknown
    piece, for a model without byte pieces (rinna's): a character the vocabulary lacks is that piece, as in
    sentencepiece, and not bytes spelled with whatever pieces happen to be at byte + 3."""
    UNIGRAM, BPE = 1, 2
    UNKNOWN, BYTE = 2, 6
    kind, normalizer, collapse = UNIGRAM, "nmt_nfkc", True  # sentencepiece's own defaults
    unknown, spelled, index = None, False, 0
    for field, value in protobuf_fields(model):
        if field == 1:  # a piece: its type
            piece_type = dict(protobuf_fields(value)).get(3, 1)
            unknown = index if piece_type == UNKNOWN and unknown is None else unknown
            spelled |= piece_type == BYTE
            index += 1
        elif field == 2:  # trainer_spec.model_type
            kind = dict(protobuf_fields(value)).get(3, UNIGRAM)
        elif field == 3:  # normalizer_spec: its name, and remove_extra_whitespaces
            spec = dict(protobuf_fields(value))
            normalizer, collapse = spec.get(1, b"nmt_nfkc").decode("utf-8"), bool(spec.get(4, 1))
    if kind not in (UNIGRAM, BPE):
        raise ValueError("This sentencepiece model is neither unigram nor BPE.")
    options = {"tokenizer_kind": "unigram" if kind == UNIGRAM else "bpe", "nfkc": "nfkc" in normalizer}
    return {**options, **({"nmt": True} if normalizer.startswith("nmt") else {}), **({"collapse": True} if collapse else {}),
            **({"unknown": unknown} if unknown is not None and not spelled else {})}


# ------------------------------------------------------------------------------------------ in the browser
class Conversion:
    """A Hugging Face model converted inside the page, from the visitor's disk or from huggingface.co.

    header: the JSON at the beginning of model.safetensors (text), base: where its tensors begin, config: the text of
    config.json, tokenizer: the bytes of tokenizer.json or of a sentencepiece model. Then feed() the bytes of the
    file in order, beginning at start, and finish(). checkpoint, tokenizer and options are what Llama() takes.
    """

    def __init__(self, header, base, config, tokenizer, tokenizer_name, dtype="int8", max_seq_len=4096, start=0,
                 tokenizer_config=None, sink=None, quantize_rows=None, bfloat16=None, chat_template=None, q8_0=None):
        try:
            self.config = json.loads(config)
        except ValueError:
            raise ValueError("config.json is not JSON.") from None
        if not isinstance(self.config, dict):
            raise ValueError("config.json is not the configuration of a model.")
        # GPT-2 spells its config differently: from here on it has the names the rest of the file uses
        self.config = normalize(self.config)
        check_config(self.config)
        if not callable(dtype):
            check_dtype(dtype)
        try:
            header = json.loads(header)
        except ValueError:
            raise ValueError("This is not a safetensors file.") from None
        vocab_size = self.config["vocab_size"]
        tokenizer = bytes(tokenizer.to_py() if hasattr(tokenizer, "to_py") else tokenizer)
        if tokenizer_name.lower().endswith(".json"):
            try:
                parsed = json.loads(tokenizer)
            except ValueError:
                raise ValueError("tokenizer.json is not JSON.") from None
            self.tokenizer, options = tokenizer_bin(tokenizer_json_pieces(parsed), vocab_size), tokenizer_json_options(parsed)
            specials = [token["content"] for token in parsed.get("added_tokens", []) if token.get("special")]
        else:
            self.tokenizer, options = tokenizer_bin(sentencepiece_pieces(tokenizer), vocab_size), sentencepiece_options(tokenizer)
            specials = sentencepiece_specials(tokenizer)
        self.start(header, base, options, tokenizer_config, dtype, max_seq_len, start, sink, quantize_rows, specials, bfloat16,
                   chat_template, q8_0)

    @classmethod
    def from_gguf(cls, head, dtype="int8", max_seq_len=4096, sink=None, quantize_rows=None, bfloat16=None, q8_0=None):
        """The same from a GGUF file (T74): head is its beginning, as far as the tensors' data (Incomplete when it
        is not). Then feed() the file from self.base on. No config.json and no tokenizer: the GGUF has both."""
        metadata, tensors, base = gguf_read(head)
        header, config = gguf_model(metadata, tensors, base)
        self = cls.__new__(cls)
        self.config = config
        check_config(normalize(config))  # a GPT-2's is in config.json's own spelling (T136's third stage)
        if not callable(dtype):
            check_dtype(dtype)
        self.tokenizer, options, tokenizer_config, specials = gguf_tokenizer(metadata, config["vocab_size"])
        self.base = base
        self.start(header, base, options, tokenizer_config, dtype, max_seq_len, base, sink, quantize_rows, specials, bfloat16,
                   q8_0=q8_0)
        return self

    def start(self, header, base, options, tokenizer_config, dtype, max_seq_len, start, sink=None, quantize_rows=None,
              specials=(), bfloat16=None, chat_template=None, q8_0=None):
        """sink and quantize_rows: see Writer, bfloat16 and q8_0: see Stream. checkpoint is None with a sink: the bytes went there. specials: the
        tokenizer's special tokens, the ones a chat template writes between the turns. chat_template: the text of
        chat_template.jinja, where there is one (T127)."""
        bos = self.config.get("bos_token_id", 1)
        eos = self.config.get("eos_token_id", 2)
        stop = [token for token in [bos, *(eos if isinstance(eos, list) else [eos])] if isinstance(token, int)]
        # a context longer than max_seq_len is cut: the RoPE tables and the scratch of the attention grow with it
        self.stream = Stream(header, int(base), self.config, dtype, int(max_seq_len), start=int(start), sink=sink,
                             quantize_rows=quantize_rows, bfloat16=bfloat16, q8_0=q8_0)
        self.options = {**options, "dtype": self.stream.dtype, "rope_theta": float(self.config.get("rope_theta", 10000.0)),
                        "bos": bos if isinstance(bos, int) else 1, "stop_tokens": stop}
        # the form: bias and arch always (as since T64 and T65), the rest only where it is not the default, so that
        # the options of every model before T124 stay what they were (kept.js's CONVERTER)
        self.options.update({key: value for key, value in self.stream.form.items()
                             if key in ("bias", "arch") or value != FORM[key]})
        # the format of one turn, from the model's own chat_template (T73). src/models.js wins when it has one
        template = one_turn_template(tokenizer_config, chat_template)
        if template:
            self.options["template"] = template
            # the special tokens it writes stand for their token; spelled out they would be a dozen tokens each.
            # The longest first, so that one that begins another never cuts it short
            written = sorted({special for special in specials if special and special in template}, key=len, reverse=True)
            if written:
                self.options["specials"] = written
        eps = self.config.get("rms_norm_eps")
        # a GGUF says it in float32 (1e-5 is 9.99999974e-06 there): six digits are what config.json writes
        eps = float(f"{eps:.6g}") if isinstance(eps, (int, float)) and eps > 0 else RMS_EPS
        if self.stream.form["arch"] == "llama" and eps != RMS_EPS:
            # T124: the epsilon of RMSNorm, where it is not the engine's 1e-5 (Qwen2.5 and Qwen3: 1e-6, which moved
            # Qwen3 0.6B's perplexity by 0.12%). Only where it differs, like qk_norm and head_dim
            self.options["rms_norm_eps"] = float(eps)
        if self.config.get("rope_scaling"):
            # the int8 file has no RoPE tables: the engine makes them, and needs the scaling for that (Llama 3)
            self.options["rope_scaling"] = dict(self.config["rope_scaling"])
        if self.stream.form["arch"] == "neox":
            # GPT-NeoX turns part of every head, and may run its two branches in parallel: the file says neither
            self.options["rotary"] = rotary_dim(self.config)
            self.options["parallel_residual"] = bool(self.config.get("use_parallel_residual", True))
        self.checkpoint = self.stream.out

    def feed(self, data):
        done, total = self.stream.feed(data)
        return done / total

    def finish(self):
        self.stream.finish()
