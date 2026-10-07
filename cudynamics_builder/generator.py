from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sympy as sp

from .model import EquationKind, SystemSpec
from .parser import algebraic_order, sympify_expression, validate_spec


@dataclass(slots=True)
class GeneratedProject:
    files: dict[str, str]


def _direct_expr(expr: sp.Expr, spec: SystemSpec, variable_refs: dict[str, str]) -> str:
    """Emit an expression in the native systems style: V(...), Vnext(...), locals and P(...)."""
    replacements: dict[sp.Symbol, sp.Symbol] = {}
    tokens: dict[str, str] = {}
    for index, name in enumerate(spec.variables):
        token = f"__direct_v_{index}"
        replacements[sp.Symbol(name)] = sp.Symbol(token)
        tokens[token] = variable_refs[name]
    for index, name in enumerate(spec.parameters):
        token = f"__direct_p_{index}"
        replacements[sp.Symbol(name)] = sp.Symbol(token)
        tokens[token] = f"P({name})"
    text = sp.ccode(expr.xreplace(replacements), standard="C99")
    for token, replacement in tokens.items():
        text = text.replace(token, replacement)
    return text


def _state_rhs(spec: SystemSpec) -> list[sp.Expr]:
    return _expanded_system(spec)[0]


def _current_refs(spec: SystemSpec) -> dict[str, str]:
    return {name: f"V({name})" for name in spec.variables}


def _final_algebraic(spec: SystemSpec, indent: str = "        ") -> list[str]:
    refs = {name: f"Vnext({name})" for name in spec.variables}
    lines: list[str] = []
    for equation in algebraic_order(spec):
        expression = sympify_expression(equation.expression)
        lines.append(f"{indent}Vnext({equation.target}) = {_direct_expr(expression, spec, refs)};")
    return lines


def _expanded_system(spec: SystemSpec) -> tuple[list[sp.Expr], dict[sp.Symbol, sp.Expr]]:
    """Return RHS expressions with all acyclic algebraic variables substituted."""
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    for equation in algebraic_order(spec):
        expression = sympify_expression(equation.expression).subs(substitutions, simultaneous=False)
        substitutions[sp.Symbol(equation.target)] = expression
    rhs = []
    for name in spec.states:
        equation = next(item for item in spec.equations if item.kind == EquationKind.STATE and item.target == name)
        rhs.append(sympify_expression(equation.expression).subs(substitutions, simultaneous=False))
    return rhs, substitutions


def _real_derivative(expression: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """Differentiate state expressions as real-valued functions.

    SymPy otherwise differentiates abs(x) through re(x)/im(x), which cannot be
    emitted as scalar CUDA code. CUDAynamics state variables are real numbers.
    """
    symbols = expression.free_symbols | {variable}
    to_real = {symbol: sp.Symbol(str(symbol), real=True) for symbol in symbols}
    from_real = {real: original for original, real in to_real.items()}
    return sp.diff(expression.xreplace(to_real), to_real[variable]).xreplace(from_real)


def _header(spec: SystemSpec) -> str:
    sid = spec.system_id
    return f'''#pragma once
#include <kernels_common.h>

#define name {sid}

const int THREADS_PER_BLOCK_(name) = 64;

__global__ void gpu_wrapper_(name)(Computation* data, uint64_t variation);

__host__ __device__ void kernelProgram_(name)(Computation* data, uint64_t variation);

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters, PerThread* pt);

#undef name
'''


def _method_euler(spec: SystemSpec) -> str:
    rhs = _state_rhs(spec)
    refs = _current_refs(spec)
    assigns = [f"        Vnext({name}) = V({name}) + H * ({_direct_expr(rhs[i], spec, refs)});" for i, name in enumerate(spec.states)]
    assigns += _final_algebraic(spec)
    return f'''    ifMETHOD(P(method), ExplicitEuler)
    {{
{chr(10).join(assigns)}
    }}'''


def _method_semi_euler(spec: SystemSpec) -> str:
    rhs = _state_rhs(spec)
    refs = _current_refs(spec)
    updates: list[str] = []
    for i, name in enumerate(spec.states):
        updates.append(f"        Vnext({name}) = V({name}) + H * ({_direct_expr(rhs[i], spec, refs)});")
        refs[name] = f"Vnext({name})"
    updates += _final_algebraic(spec)
    return '''    ifMETHOD(P(method), SemiExplicitEuler)
    {
''' + "\n".join(updates) + '''
    }'''


def _method_midpoint(spec: SystemSpec) -> str:
    rhs = _state_rhs(spec)
    current = _current_refs(spec)
    midpoint_refs = dict(current)
    stage = []
    for i, name in enumerate(spec.states):
        stage.append(f"        numb {name}mp = V({name}) + (numb)0.5 * H * ({_direct_expr(rhs[i], spec, current)});")
        midpoint_refs[name] = f"{name}mp"
    final = [f"        Vnext({name}) = V({name}) + H * ({_direct_expr(rhs[i], spec, midpoint_refs)});" for i, name in enumerate(spec.states)]
    final += _final_algebraic(spec)
    return f'''    ifMETHOD(P(method), ExplicitMidpoint)
    {{
{chr(10).join(stage)}

{chr(10).join(final)}
    }}'''


def _method_rk4(spec: SystemSpec) -> str:
    rhs = _state_rhs(spec)
    current = _current_refs(spec)
    lines: list[str] = []
    for stage in range(1, 5):
        if stage == 1:
            refs = current
        else:
            refs = dict(current)
            previous = stage - 1
            factor = "(numb)0.5 * " if stage in (2, 3) else ""
            for name in spec.states:
                declaration = "numb " if stage == 2 else ""
                lines.append(f"        {declaration}{name}mp = V({name}) + {factor}H * k{name}{previous};")
                refs[name] = f"{name}mp"
            lines.append("")
        for index, name in enumerate(spec.states):
            lines.append(f"        numb k{name}{stage} = {_direct_expr(rhs[index], spec, refs)};")
        if stage != 4:
            lines.append("")
    lines.append("")
    for name in spec.states:
        lines.append(f"        Vnext({name}) = V({name}) + H * (k{name}1 + (numb)2.0 * k{name}2 + (numb)2.0 * k{name}3 + k{name}4) / (numb)6.0;")
    lines += _final_algebraic(spec)
    return f'''    ifMETHOD(P(method), ExplicitRungeKutta4)
    {{
{chr(10).join(lines)}
    }}'''


_DP8_M = (
    (0,), (.05555555555556,), (.02083333333333,.0625), (.03125,0,.09375),
    (.3125,0,-1.171875,1.171875), (.0375,0,0,.1875,.15),
    (.04791013711111,0,0,.1122487127778,-.02550567377778,.01284682388889),
    (.01691798978729,0,0,.387848278486,.0359773698515,.1969702142157,-.1727138523405),
    (.06909575335919,0,0,-.6342479767289,-.1611975752246,.1386503094588,.9409286140358,.2116363264819),
    (.183556996839,0,0,-2.468768084316,-.2912868878163,-.02647302023312,2.847838764193,.2813873314699,.1237448998633),
    (-1.215424817396,0,0,16.67260866595,.9157418284168,-6.056605804357,-16.00357359416,14.8493030863,-13.37157573529,5.13418264818),
    (.2588609164383,0,0,-4.774485785489,-.435093013777,-3.049483332072,5.577920039936,6.155831589861,-5.062104586737,2.193926173181,.1346279986593),
    (.8224275996265,0,0,-11.65867325728,-.7576221166909,.7139735881596,12.07577498689,-2.12765911392,1.990166207049,-.234286471544,.1758985777079),
)
_DP8_B = (.04174749114153,0,0,0,0,-.05545232861124,.2393128072012,.7035106694034,-.7597596138145,.6605630309223,.1581874825101,-.2381095387529,.25)


def _method_dp8(spec: SystemSpec) -> str:
    rows = []
    for row in _DP8_M:
        padded = row + (0,) * (12 - len(row))
        rows.append("        {" + ", ".join(f"(numb){x:.15g}" for x in padded) + "}")
    matrix = ",\n".join(rows)
    weights = ", ".join(f"(numb){x:.15g}" for x in _DP8_B)
    rhs = _state_rhs(spec)
    stage_refs = _current_refs(spec)
    declarations = []
    stage_init = []
    stage_sum = []
    evaluations = []
    final = []
    for index, name in enumerate(spec.states):
        declarations.append(f"        numb k{name}[13];")
        stage_init.append(f"            numb {name}Stage = V({name});")
        stage_sum.append(f"                {name}Stage += H * M[s][j] * k{name}[j];")
        stage_refs[name] = f"{name}Stage"
        final += [
            f"        numb {name}Sum = 0;",
            f"        for (int s = 0; s < 13; ++s) {name}Sum += b[s] * k{name}[s];",
            f"        Vnext({name}) = V({name}) + H * {name}Sum;",
        ]
    evaluations = [f"            k{name}[s] = {_direct_expr(rhs[index], spec, stage_refs)};" for index, name in enumerate(spec.states)]
    final += _final_algebraic(spec)
    return f'''    ifMETHOD(P(method), ExplicitDormandPrince8)
    {{
        const numb M[13][12] = {{
{matrix}
        }};
        const numb b[13] = {{ {weights} }};
{chr(10).join(declarations)}
        for (int s = 0; s < 13; ++s)
        {{
{chr(10).join(stage_init)}
            for (int j = 0; j < s; ++j)
            {{
{chr(10).join(stage_sum)}
            }}
{chr(10).join(evaluations)}
        }}
{chr(10).join(final)}
    }}'''


def _method_implicit(spec: SystemSpec, method_name: str, midpoint: bool) -> str:
    """Inline Newton iteration with direct RHS expressions and an analytic Jacobian."""
    rhs = _state_rhs(spec)
    state_symbols = [sp.Symbol(name) for name in spec.states]
    jacobian = [[_real_derivative(rhs[i], state_symbols[j]) for j in range(len(spec.states))] for i in range(len(spec.states))]
    current = _current_refs(spec)
    evaluation_refs = dict(current)
    lines: list[str] = []
    for index, name in enumerate(spec.states):
        lines.append(f"        numb {name}Guess = V({name}) + H * ({_direct_expr(rhs[index], spec, current)});")
        evaluation_refs[name] = f"((V({name}) + {name}Guess) * (numb)0.5)" if midpoint else f"{name}Guess"
    lines += ["", "        for (int iteration = 0; iteration < 8; ++iteration)", "        {"]
    for index, name in enumerate(spec.states):
        expression = _direct_expr(rhs[index], spec, evaluation_refs)
        lines.append(f"            numb residual_{name} = {name}Guess - V({name}) - H * ({expression});")
    norm = " + ".join(f"residual_{name} * residual_{name}" for name in spec.states)
    lines += [f"            numb residualNorm = {norm};", "            if (residualNorm < (numb)1e-18) break;", ""]
    factor = "((numb)0.5 * H)" if midpoint else "H"
    matrix_rows = []
    for i in range(len(spec.states)):
        entries = []
        for j in range(len(spec.states)):
            derivative = _direct_expr(jacobian[i][j], spec, evaluation_refs)
            identity = "(numb)1.0" if i == j else "(numb)0.0"
            entries.append(f"{identity} - {factor} * ({derivative})")
        matrix_rows.append("{ " + ", ".join(entries) + " }")
    residuals = ", ".join(f"-residual_{name}" for name in spec.states)
    size = len(spec.states)
    lines += [
        f"            numb A[{size}][{size}] = {{ {', '.join(matrix_rows)} }};",
        f"            numb delta[{size}] = {{ {residuals} }};",
        f"            for (int k = 0; k < {size}; ++k)",
        "            {",
        "                int pivot = k;",
        f"                for (int i = k + 1; i < {size}; ++i) if (abs(A[i][k]) > abs(A[pivot][k])) pivot = i;",
        f"                if (pivot != k) {{ for (int j = k; j < {size}; ++j) {{ numb q = A[k][j]; A[k][j] = A[pivot][j]; A[pivot][j] = q; }} numb q = delta[k]; delta[k] = delta[pivot]; delta[pivot] = q; }}",
        "                numb divisor = abs(A[k][k]) < (numb)1e-12 ? (A[k][k] < 0 ? (numb)-1e-12 : (numb)1e-12) : A[k][k];",
        "                A[k][k] = divisor;",
        f"                for (int i = k + 1; i < {size}; ++i) {{ numb q = A[i][k] / divisor; for (int j = k; j < {size}; ++j) A[i][j] -= q * A[k][j]; delta[i] -= q * delta[k]; }}",
        "            }",
        f"            for (int i = {size - 1}; i >= 0; --i) {{ for (int j = i + 1; j < {size}; ++j) delta[i] -= A[i][j] * delta[j]; delta[i] /= A[i][i]; }}",
    ]
    for index, name in enumerate(spec.states):
        lines.append(f"            {name}Guess += delta[{index}];")
    lines += ["        }", ""]
    for name in spec.states:
        lines.append(f"        Vnext({name}) = {name}Guess;")
    lines += _final_algebraic(spec)
    return f'''    ifMETHOD(P(method), {method_name})
    {{
{chr(10).join(lines)}
    }}'''


def _method_vscd(spec: SystemSpec) -> str:
    rhs, _ = _expanded_system(spec)
    current = _current_refs(spec)
    stage_refs = dict(current)
    forward: list[str] = []
    reverse: list[str] = []
    # Dummy symbols cannot collide with user variables having similar names.
    candidate_symbol = sp.Dummy("vscdCandidate")
    base_symbol = sp.Dummy("vscdBase")
    h2_symbol = sp.Dummy("h2")

    for index, name in enumerate(spec.states):
        expression = _direct_expr(rhs[index], spec, stage_refs)
        forward.append(f"        numb {name}mp = V({name}) + h1 * ({expression});")
        stage_refs[name] = f"{name}mp"

    reverse_refs = dict(stage_refs)
    for index in range(len(spec.states) - 1, -1, -1):
        name = spec.states[index]
        state_symbol = sp.Symbol(name)
        equation_rhs = rhs[index].subs(state_symbol, candidate_symbol)
        residual = sp.together(candidate_symbol - base_symbol - h2_symbol * equation_rhs)
        numerator = residual.as_numer_denom()[0]
        solution = None
        try:
            polynomial = sp.Poly(numerator, candidate_symbol)
            if polynomial.degree() == 1:
                solution = sp.simplify(-polynomial.nth(0) / polynomial.nth(1))
        except sp.PolynomialError:
            pass

        candidate_name = f"vscdCandidate_{name}"
        if solution is not None:
            concrete = solution.subs({base_symbol: sp.Symbol(f"{name}mp"), h2_symbol: sp.Symbol("h2")})
            reverse += [
                f"        // Exact affine solution of the reverse VSCD coordinate equation for {name}.",
                f"        Vnext({name}) = {_direct_expr(concrete, spec, reverse_refs)};",
            ]
        else:
            residual_expr = residual.subs(
                {base_symbol: sp.Symbol(f"{name}mp"), candidate_symbol: sp.Symbol(candidate_name), h2_symbol: sp.Symbol("h2")}
            )
            derivative_expr = _real_derivative(residual, candidate_symbol).subs(
                {base_symbol: sp.Symbol(f"{name}mp"), candidate_symbol: sp.Symbol(candidate_name), h2_symbol: sp.Symbol("h2")}
            )
            residual_name = f"vscdResidual_{name}"
            derivative_name = f"vscdDerivative_{name}"
            correction_name = f"vscdCorrection_{name}"
            reverse += [
                f"        // Nonlinear reverse VSCD coordinate equation for {name}: analytic scalar Newton.",
                f"        numb {candidate_name} = {name}mp;",
                "        for (int iteration = 0; iteration < 8; ++iteration)",
                "        {",
                f"            const numb {residual_name} = {_direct_expr(residual_expr, spec, reverse_refs)};",
                f"            const numb {derivative_name} = {_direct_expr(derivative_expr, spec, reverse_refs)};",
                f"            const numb {correction_name} = {residual_name} / {derivative_name};",
                f"            {candidate_name} -= {correction_name};",
                f"            if (abs({correction_name}) < (numb)1e-9) break;",
                "        }",
                f"        Vnext({name}) = {candidate_name};",
            ]
        reverse_refs[name] = f"Vnext({name})"
    reverse += _final_algebraic(spec)
    return f'''    ifMETHOD(P(method), VariableSymmetryCD)
    {{
        numb h1 = (numb)0.5 * H - P(symmetry);
        numb h2 = (numb)0.5 * H + P(symmetry);

{chr(10).join(forward)}

{chr(10).join(reverse)}
    }}'''


def _source(spec: SystemSpec) -> str:
    sid = spec.system_id
    methods = ", ".join(spec.methods)
    params = list(spec.parameters)
    if "VariableSymmetryCD" in spec.methods:
        params.append("symmetry")
    params += ["method", "COUNT"]
    pieces = []
    for method in spec.methods:
        pieces.append({
            "ExplicitEuler": lambda: _method_euler(spec),
            "SemiExplicitEuler": lambda: _method_semi_euler(spec),
            "ImplicitEuler": lambda: _method_implicit(spec, "ImplicitEuler", False),
            "ExplicitMidpoint": lambda: _method_midpoint(spec),
            "ImplicitMidpoint": lambda: _method_implicit(spec, "ImplicitMidpoint", True),
            "ExplicitRungeKutta4": lambda: _method_rk4(spec),
            "ExplicitDormandPrince8": lambda: _method_dp8(spec),
            "VariableSymmetryCD": lambda: _method_vscd(spec),
        }[method]())
    return f'''#include "{sid}.h"

#define name {sid}

namespace attributes
{{
    enum variables {{ {", ".join(spec.variables)} }};
    enum parameters {{ {", ".join(params)} }};
    enum methods {{ {methods} }};
}}

__global__ void gpu_wrapper_(name)(Computation* data, uint64_t variation)
{{
    kernelProgram_(name)(data, (blockIdx.x * blockDim.x) + threadIdx.x);
}}

__host__ __device__ void kernelProgram_(name)(Computation* data, uint64_t variation)
{{
    if (variation >= CUDA_marshal.totalVariations) return;      // Shutdown thread if there isn't a variation to compute
    uint64_t stepStart, variationStart = variation * CUDA_marshal.variationSize;         // Start index to store the modelling data for the variation
    LOCAL_BUFFERS;
    LOAD_ATTRIBUTES(false);

    TRANSIENT_SKIP_NEW(finiteDifferenceScheme_(name));

    for (int s = 0; s < CUDA_kernel.steps && !data->isHires; s++)
    {{
        stepStart = variationStart + s * CUDA_kernel.VAR_COUNT;
        finiteDifferenceScheme_(name)(FDS_ARGUMENTS);
        RECORD_STEP;
    }}

    // Analysis
    AnalysisLobby(data, &finiteDifferenceScheme_(name), variation);
}}

__host__ __device__ __forceinline__ void finiteDifferenceScheme_(name)(numb* currentV, numb* nextV, numb* parameters, PerThread* pt)
{{
{(chr(10) + chr(10)).join(pieces)}
}}

#undef name
'''


def _metadata(spec: SystemSpec) -> str:
    lines = [
        f"Name: {spec.display_name}", f"Steps: {spec.steps}", f"Transient: {spec.transient}",
        "// Defining step:",
        "// parameter/variable/discrete",
        "// Then, unless it is discrete, provide its name and define it like an attribute",
        "// If it is discrete, add nothing after",
        f"Step type: parameter h Fixed {spec.step:g} {spec.step * 10:g} {spec.step:g} 100 0.0 0.0",
        "// Compute on CUDAynamics launch: yes/no",
        f"Execute on launch: {'yes' if spec.execute_on_launch else 'no'}",
        "// Ranging types:",
        "// Fixed (single <minimum value>)",
        "// Linear (<step count> values uniformly distributed between <minimum value> and <maximum value>, inclusively)",
        "// Step (values are picked from <minimum value> to <maximum value> with a step <step>, including minimum, including maximum if it ends up as a picked value)",
        "// Random (<step count> values randomly picked from <minimum value> to <maximum value>)",
        "// Normal (normal distribution of <step count> values, defined by <normal mean> and <normal deviation>)",
        "//",
        "// Defining variables/parameters:",
        "// var/param <name> <ranging type> <minimum value> <maximum value> <step> <step count> <normal mean> <normal deviation>",
    ]
    for name in spec.variables:
        d = spec.variable_defaults[name]
        lines.append(f"var {name} {d.range_kind} {d.value:g} {d.maximum:g} {d.step:g} {d.count} {d.mean:g} {d.deviation:g}")
    for name in spec.parameters:
        d = spec.parameter_defaults[name]
        lines.append(f"param {name} {d.range_kind} {d.value:g} {d.maximum:g} {d.step:g} {d.count} {d.mean:g} {d.deviation:g}")
    if "VariableSymmetryCD" in spec.methods:
        lines += [
            f"param symmetry Fixed {spec.symmetry_default:g} 1.0 0.01 100 0.0 0.0",
            "// \"symmetry\" parameter will only be displayed if method equals VariableSymmetryCD",
            "constraint method VariableSymmetryCD",
        ]
    enum_items = " ".join(("1" if i == 0 else "0") + name for i, name in enumerate(spec.methods))
    lines += [
        "// Defining enumerated parameters (useful for methods):",
        "// enum <name> <ranging type> <minimum value> <maximum value> <step> <step count> <normal mean> <normal deviation> <enum names, no spaces>",
        f"enum method {enum_items}",
        "//",
        "// Defining settings for analysis functions:",
        "// analysis <name from \"anfunc_names.cpp\"> settings <values, must exactly match the settings struct>",
        "analysis Minimum-maximum settings 0 1",
    ]
    if len(spec.variables) >= 2:
        lines.append(f"analysis Largest Lyapunov exponent settings {spec.step:g} 30 0 0 1 1 -1")
    return "\n".join(lines) + "\n"


def generate(spec: SystemSpec) -> GeneratedProject:
    errors = validate_spec(spec)
    if errors:
        raise ValueError("\n".join(errors))
    sid = spec.system_id
    return GeneratedProject({f"systems/{sid}/{sid}.h": _header(spec), f"systems/{sid}/{sid}.cu": _source(spec), f"systems/{sid}/{sid}.txt": _metadata(spec)})
