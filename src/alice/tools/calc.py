"""Safe calculator: evaluates arithmetic with an AST whitelist (no eval, no names beyond math)."""
from __future__ import annotations

import ast
import math
import operator
import re

from . import Tool

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {name: getattr(math, name) for name in
          ("sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "asin", "acos", "atan",
           "ceil", "floor", "factorial", "comb", "perm", "gcd", "lcm", "isqrt", "fsum")}
_FUNCS.update(abs=abs, round=round, min=min, max=max, sum=sum)
_CONSTS = {"pi": math.pi, "e": math.e, "inf": math.inf}
MAX_EXPONENT = 10_000
MAX_FACTORIAL = 5_000
MAX_DIGITS = 5_000


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent too large (> {MAX_EXPONENT})")
        result = _BINOPS[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > MAX_DIGITS * 3.33:
            raise ValueError("result too large")
        return result
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and args and args[0] > MAX_FACTORIAL:
            raise ValueError(f"factorial argument too large (> {MAX_FACTORIAL})")
        return _FUNCS[node.func.id](*args)
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


def calculate(expression: str) -> str:
    expr = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if len(expr) > 500:
        return "error: expression too long"
    if re.search(r"\bimport\b|;|\bfor\b|\bdef\b|(?<![<>=!])=(?!=)", expr):
        return ("error: calculate takes ONE arithmetic expression (no imports, statements, variables or "
                "loops). Inline the numbers, e.g. 'exp(2)/(exp(2)+exp(1)+exp(-1))', or use run_code for code.")
    try:
        value = _eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "error: division by zero"
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        return f"error: {e}"
    return f"{expression.strip()} = {format_value(value)}"


def format_value(value) -> str:
    """Exact value, plus easier-to-copy forms: digit grouping for big integers, percent for fractions.
    (The 9B model once copied 101770 as '10,177'.)"""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    if isinstance(value, int):
        return f"{value} ({value:,})" if abs(value) >= 10_000 else str(value)
    if isinstance(value, float):
        text = f"{value:.10g}"
        if 0 < abs(value) < 1:
            text += f" (= {value * 100:.4g}%)"
        return text
    return str(value)


_NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")


def last_result_value(output: str) -> float | None:
    """The numeric value of a calculate() output, e.g. 'a*b = 101770 (101,770)' -> 101770.0."""
    if output.startswith("error:") or " = " not in output:
        return None
    m = _NUM.search(output.rsplit(" = ", 1)[1])
    return float(m.group(0).replace(",", "")) if m else None


def reply_mentions(reply: str, value: float, rel_tol: float = 0.006) -> bool:
    """Does the reply state `value` (allowing rounding, digit grouping, or a percentage)?"""
    for m in _NUM.finditer(reply):
        x = float(m.group(0).replace(",", ""))
        if float(value).is_integer() and abs(value) >= 1:
            if x == value:
                return True
            continue
        tol = abs(value) * rel_tol
        if abs(x - value) <= tol or abs(x / 100 - value) <= tol:
            return True
    return False


TOOL = Tool(
    name="calculate",
    description=("Evaluate an arithmetic expression exactly. Use this for ANY calculation instead of doing "
                 "arithmetic yourself. Supports + - * / // % ** (or ^), parentheses, pi, e and math functions: "
                 "sqrt, log (natural), log10, log2, exp, sin, cos, tan, ceil, floor, factorial, comb, perm, "
                 "gcd, lcm, abs, round, min, max, sum. Example: '0.99*0.001/(0.99*0.001+0.05*0.999)'."),
    parameters={"type": "object",
                "properties": {"expression": {"type": "string", "description": "the expression to evaluate"}},
                "required": ["expression"]},
    run=calculate,
)
