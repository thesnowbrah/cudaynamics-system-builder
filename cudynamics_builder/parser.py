from __future__ import annotations

import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from .model import Equation, EquationKind, METHODS, SystemSpec


class ParseError(ValueError):
    pass


_TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
)
_MATH_NAMES = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,
    "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "atan2": sp.atan2, "sinh": sp.sinh, "cosh": sp.cosh,
    "tanh": sp.tanh, "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log,
    "log10": lambda x: sp.log(x, 10), "log2": lambda x: sp.log(x, 2), "pow": sp.Pow,
    "abs": sp.Abs, "Abs": sp.Abs, "fabs": sp.Abs,
    "floor": sp.floor, "ceil": sp.ceiling, "ceiling": sp.ceiling,
    "min": sp.Min, "Min": sp.Min, "max": sp.Max, "Max": sp.Max,
    "mod": sp.Mod, "fmod": sp.Mod, "sign": sp.sign, "sgn": sp.sign,
    "erf": sp.erf, "erfc": sp.erfc,
    "hypot": lambda x, y: sp.sqrt(x * x + y * y),
    "pi": sp.pi, "E": sp.E,
}


def sympify_expression(text: str) -> sp.Expr:
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    local_dict = {name: sp.Symbol(name) for name in identifiers if name not in _MATH_NAMES}
    local_dict.update(_MATH_NAMES)
    return parse_expr(text, local_dict=local_dict, transformations=_TRANSFORMS, evaluate=False)


def _strip_outer_groups(text: str) -> str:
    """Remove balanced {...} wrappers without touching groups inside an expression."""
    value = text.strip()
    while value.startswith("{") and value.endswith("}"):
        depth = 0
        closes_at_end = False
        for index, char in enumerate(value):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    closes_at_end = index == len(value) - 1
                    break
        if not closes_at_end:
            break
        value = value[1:-1].strip()
    return value


def _clean_latex_fragment(text: str) -> str:
    text = text.strip().strip("$").strip()
    text = re.sub(r"\\(?:displaystyle|textstyle|scriptstyle)\b", "", text)
    text = text.replace("&", "")
    text = re.sub(r"\\(?:quad|qquad|,|;|:|!)", " ", text)
    text = re.sub(r"\\left\s*", "", text)
    text = re.sub(r"\\right\s*", "", text)
    return _strip_outer_groups(text.strip())


def _split_lines(text: str) -> list[str]:
    text = text.replace("\r", "\n").replace("−", "-")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")
    # System delimiters emitted by pix2tex are presentation, not equations.
    text = re.sub(r"\\left\s*\\?[\{\[\(]", "", text)
    text = re.sub(r"\\right\s*(?:\\?[\}\]\)]|\.)", "", text)
    environments = r"cases|aligned|alignedat|gathered|array|matrix|pmatrix|bmatrix|vmatrix"
    text = re.sub(rf"\\begin\s*\{{(?:{environments})\}}(?:\s*\{{[^{{}}]*\}})?", "", text)
    text = re.sub(rf"\\end\s*\{{(?:{environments})\}}", "", text)
    text = re.sub(r"\\\\(?:\s*\[[^\]]*\])?", "\n", text)
    text = text.replace(";", "\n")
    lines = []
    for raw in text.splitlines():
        line = _clean_latex_fragment(raw).strip(" \t,$")
        line = _strip_outer_groups(line)
        if line:
            lines.append(line)
    return lines


def _latex_rhs_to_sympy(rhs: str) -> str:
    rhs = _clean_latex_fragment(rhs).replace("−", "-").replace("·", "*")
    if "\\" not in rhs and "{" not in rhs:
        return rhs
    try:
        from sympy.parsing.latex import parse_latex
        return str(parse_latex(rhs))
    except Exception:
        rhs = rhs.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi")
        rhs = re.sub(r"\\operatorname\s*\{([A-Za-z]+)\}", r"\1", rhs)
        for old, new in (("\\sin", "sin"), ("\\cos", "cos"), ("\\tan", "tan"),
                         ("\\arcsin", "asin"), ("\\arccos", "acos"), ("\\arctan", "atan"),
                         ("\\sinh", "sinh"), ("\\cosh", "cosh"), ("\\tanh", "tanh"),
                         ("\\sec", "sec"), ("\\csc", "csc"), ("\\cot", "cot"),
                         ("\\exp", "exp"), ("\\sqrt", "sqrt"), ("\\ln", "log"),
                         ("\\log", "log"), ("\\min", "min"), ("\\max", "max")):
            rhs = rhs.replace(old, new)
        rhs = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", rhs)
        return rhs.replace("{", "(").replace("}", ")")


def _parse_lhs(lhs: str) -> tuple[str, EquationKind]:
    raw = _clean_latex_fragment(lhs)
    raw = re.sub(r"\\(?:mathrm|operatorname|text)\s*\{d\}", "d", raw)
    raw = re.sub(r"\\partial", "d", raw)
    raw = re.sub(r"\s+", "", raw)
    raw = _strip_outer_groups(raw)
    fraction = re.fullmatch(r"\\(?:dfrac|tfrac|frac)\{(.+)\}\{(.+)\}", raw)
    if fraction:
        numerator = _strip_outer_groups(fraction.group(1)).replace("{", "").replace("}", "")
        denominator = _strip_outer_groups(fraction.group(2)).replace("{", "").replace("}", "")
        match = re.fullmatch(r"d([A-Za-z_][A-Za-z0-9_]*)", numerator)
        if match and denominator in {"dt", "dtime"}:
            return match.group(1), EquationKind.STATE
    patterns = (
        r"^d([A-Za-z_][A-Za-z0-9_]*)/dt$",
        r"^\\dot\{([A-Za-z_][A-Za-z0-9_]*)\}$",
        r"^\\dot([A-Za-z_][A-Za-z0-9_]*)$",
        r"^dot\(?([A-Za-z_][A-Za-z0-9_]*)\)?$",
        r"^([A-Za-z_][A-Za-z0-9_]*)['′]$",
    )
    for pattern in patterns:
        match = re.match(pattern, raw)
        if match:
            return match.group(1), EquationKind.STATE
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)$", raw)
    if match:
        return match.group(1), EquationKind.ALGEBRAIC
    raise ParseError(f"Не удалось распознать левую часть: {lhs!r}")


def safe_identifier(name: str) -> str:
    translit = str.maketrans({"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "σ": "sigma", "ρ": "rho", "μ": "mu", "ω": "omega"})
    name = re.sub(r"[^A-Za-z0-9_]", "_", name.translate(translit))
    if not name or name[0].isdigit():
        name = "v_" + name
    return name


def parse_system(text: str, system_id: str = "new_system", display_name: str = "New system") -> SystemSpec:
    equations: list[Equation] = []
    errors: list[str] = []
    for number, line in enumerate(_split_lines(text), 1):
        line = re.sub(r"^\s*[-•]\s*", "", line)
        if "=" not in line:
            errors.append(f"строка {number}: нет знака '='")
            continue
        lhs, rhs = line.split("=", 1)
        try:
            target, kind = _parse_lhs(lhs)
            expression = str(sympify_expression(_latex_rhs_to_sympy(rhs)))
            equations.append(Equation(safe_identifier(target), expression, kind, line))
        except Exception as exc:
            errors.append(f"строка {number}: {exc}")
    if errors:
        raise ParseError("\n".join(errors))
    if not equations:
        raise ParseError("Система не содержит уравнений")
    targets = [e.target for e in equations]
    duplicates = sorted({x for x in targets if targets.count(x) > 1})
    if duplicates:
        raise ParseError("Повторно заданы переменные: " + ", ".join(duplicates))

    symbols = {str(s) for equation in equations for s in sympify_expression(equation.expression).free_symbols}
    parameters = sorted(symbols - set(targets) - {"t"})
    if "t" in symbols and "t" not in targets:
        equations.append(Equation("t", "1", EquationKind.STATE, "dt/dt = 1 (добавлено автоматически)"))
    spec = SystemSpec(safe_identifier(system_id.lower()), display_name, equations, parameters)
    spec.ensure_defaults()
    return spec


def algebraic_order(spec: SystemSpec) -> list[Equation]:
    remaining = {e.target: e for e in spec.equations if e.kind == EquationKind.ALGEBRAIC}
    available = set(spec.states) | set(spec.parameters)
    ordered: list[Equation] = []
    while remaining:
        ready = [name for name, eq in remaining.items() if {str(s) for s in sympify_expression(eq.expression).free_symbols} <= available]
        if not ready:
            raise ParseError("Циклическая зависимость алгебраических переменных: " + ", ".join(remaining))
        for name in ready:
            ordered.append(remaining.pop(name))
            available.add(name)
    return ordered


def validate_spec(spec: SystemSpec) -> list[str]:
    problems: list[str] = []
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec.system_id):
        problems.append("ID системы должен быть корректным идентификатором C/C++")
    if not spec.states:
        problems.append("Нужно хотя бы одно дифференциальное уравнение")
    if not spec.methods:
        problems.append("Выберите хотя бы один метод интегрирования")
    unsupported_methods = set(spec.methods) - set(METHODS)
    if unsupported_methods:
        problems.append("Неподдерживаемые методы: " + ", ".join(sorted(unsupported_methods)))
    if len(spec.variables) > 128 or len(spec.parameters) + 3 > 128:
        problems.append("CUDAynamics допускает не более 128 переменных и 128 параметров (с учётом h, method и symmetry)")
    reserved_parameters = set(spec.parameters) & {"method", "COUNT", "symmetry"}
    if reserved_parameters:
        problems.append("Зарезервированные имена параметров: " + ", ".join(sorted(reserved_parameters)))
    known = set(spec.variables) | set(spec.parameters)
    for equation in spec.equations:
        try:
            unknown = {str(s) for s in sympify_expression(equation.expression).free_symbols} - known
            if unknown:
                problems.append(f"{equation.target}: неизвестные символы {', '.join(sorted(unknown))}")
        except Exception as exc:
            problems.append(f"{equation.target}: {exc}")
    try:
        algebraic_order(spec)
    except ParseError as exc:
        problems.append(str(exc))
    return problems


def pretty_equations(spec: SystemSpec) -> str:
    rows = []
    for equation in spec.equations:
        lhs = f"d{equation.target}/dt" if equation.kind == EquationKind.STATE else equation.target
        rows.append(f"{lhs} = {sp.sstr(sympify_expression(equation.expression))}")
    return "\n".join(rows)
