"""Display helpers: make LaTeX maths readable in a terminal.

The 9B writes maths as LaTeX in about half of its numeric answers (13/23 replies in the numeric check),
which a terminal shows as raw "\\[ \\text{lrate} = ... \\]". Asking for plain-text maths in the system prompt
cut that to 6/23 but numeric accuracy fell from 14/15 and 13/15 to 11/15 in two runs, so the prompt stays
as it was and the chat box converts the common constructs at display time instead. Only the shown text
changes; the stored reply and every check work on the model's original text.
"""
from __future__ import annotations

import re

_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "eta": "η",
    "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
    "omega": "ω", "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "nabla": "∇", "partial": "∂", "infty": "∞",
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "approx": "≈", "neq": "≠", "ne": "≠", "leq": "≤",
    "le": "≤", "geq": "≥", "ge": "≥", "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒",
    "in": "∈", "mid": "|", "sum": "∑", "prod": "∏", "int": "∫", "sqrt": "√", "ldots": "…", "cdots": "⋯", "log": "log",
    "ln": "ln", "exp": "exp", "min": "min", "max": "max", "sin": "sin", "cos": "cos", "quad": " ", "qquad": "  ",
    ",": " ", ";": " ", "!": "", " ": " ",
}
_SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])", re.S)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.S)


def _convert(expr: str) -> str:
    s = expr
    for _ in range(3):                                   # nested \frac / \text etc.
        s = re.sub(r"\\(?:text|textbf|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", lambda m: f"({m[1]})/({m[2]})", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: f"√({m[1]})", s)
    s = re.sub(r"\\left|\\right|\\big|\\Big", "", s)
    s = re.sub(r"\\([A-Za-z]+|[,;! ])", lambda m: _SYMBOLS.get(m[1], m[1]), s)
    s = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: _sup(m[1] if m[1] is not None else m[2]), s)
    s = re.sub(r"_\{([^{}]*)\}|_(\d)", lambda m: _sub(m[1] if m[1] is not None else m[2]), s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def _sup(body: str) -> str:
    b = body.strip()
    return b.translate(_SUP) if b and all(c in "0123456789+-=()n" for c in b) else f"^({b})" if len(b) > 1 else f"^{b}"


def _sub(body: str) -> str:
    b = body.strip()
    return b.translate(_SUB) if b and all(c in "0123456789+-=()" for c in b) else f"_{b}"


def latex_to_text(text: str) -> str:
    """Replace $...$, $$...$$, \\(...\\) and \\[...\\] maths outside code with readable Unicode text."""
    parts = _CODE.split(text)
    for i in range(0, len(parts), 2):                    # even parts are outside code
        parts[i] = _MATH.sub(lambda m: _convert(next(g for g in m.groups() if g is not None)), parts[i])
    return "".join(parts)
