from pathlib import Path

import pytest
import sympy as sp

from cudynamics_builder.generator import generate
from cudynamics_builder.installer import build_plan
from cudynamics_builder.model import EquationKind
from cudynamics_builder.parser import ParseError, algebraic_order, parse_system, validate_spec


LORENZ_WITH_ALGEBRAIC = """dx/dt = sigma*(y-x)
dy/dt = x*(rho-z)-y
dz/dt = x*y-beta*z
energy = x^2+y^2+z^2
"""

PIX2TEX_ROSSLER = r"""\left\{\begin{array}{l} {{\frac{d x}{d t}=-y-z}} \\ {{\frac{d y}{d t}=x+a y}} \\ {{\frac{d z}{d t}=b+z(x-c)}}\end{array}\right."""


def test_parser_classifies_states_algebraic_and_parameters():
    spec = parse_system(LORENZ_WITH_ALGEBRAIC, "generated_lorenz", "Generated Lorenz")
    assert spec.states == ["x", "y", "z"]
    assert spec.algebraic == ["energy"]
    assert set(spec.parameters) == {"sigma", "rho", "beta"}
    assert not validate_spec(spec)


def test_time_is_added_as_state():
    spec = parse_system("dx/dt = sin(t)-a*x", "forced", "Forced")
    assert "t" in spec.states
    assert "a" in spec.parameters


def test_pix2tex_array_output_is_normalized_automatically():
    spec = parse_system(PIX2TEX_ROSSLER, "rossler_test", "Rossler Test")
    assert spec.states == ["x", "y", "z"]
    assert set(spec.parameters) == {"a", "b", "c"}
    assert spec.equations[0].expression == "-y - z"
    assert "a*y" in spec.equations[1].expression
    assert sp.simplify(sp.sympify(spec.equations[2].expression) - sp.sympify("b + z*(x-c)")) == 0


def test_main_math_functions_parse_and_generate_cuda_expressions():
    text = "dx/dt = sin(x)+cos(x)+sec(x)+csc(x)+cot(x)+abs(x)+sign(x)+max(a,x)-min(b,x)+sqrt(c)+exp(-x)+log(d)+log10(d)+log2(d)+floor(e)+ceil(f)+atan2(x,a)+hypot(x,b)"
    spec = parse_system(text, "functions", "Functions")
    assert set(spec.parameters) == {"a", "b", "c", "d", "e", "f"}
    source = generate(spec).files["systems/functions/functions.cu"]
    for token in ("sin(", "cos(", "tan(", "fabs(", "fmax(", "fmin(", "sqrt(", "exp(", "log(", "floor(", "ceil(", "atan2(", "> 0"):
        assert token in source


def test_latex_math_functions_from_pix2tex_are_supported():
    spec = parse_system(r"\frac{d x}{d t}=\sin(x)+\cos(x)+\max(a,x)+\min(b,x)", "latex_functions", "LaTeX functions")
    assert spec.states == ["x"]
    assert set(spec.parameters) == {"a", "b"}


def test_algebraic_cycle_is_rejected():
    spec = parse_system("dx/dt = q\nq = r+x\nr = q-x", "cycle", "Cycle")
    assert any("Циклическая" in error for error in validate_spec(spec))


def test_generation_contains_all_selected_methods_and_algebraic_eval():
    spec = parse_system(LORENZ_WITH_ALGEBRAIC, "generated_lorenz", "Generated Lorenz")
    spec.methods = list(("ExplicitEuler", "SemiExplicitEuler", "ImplicitEuler", "ExplicitMidpoint", "ImplicitMidpoint", "ExplicitRungeKutta4", "ExplicitDormandPrince8", "VariableSymmetryCD"))
    project = generate(spec)
    source = project.files["systems/generated_lorenz/generated_lorenz.cu"]
    header = project.files["systems/generated_lorenz/generated_lorenz.h"]
    for method in spec.methods:
        assert method in source
    assert "Vnext(energy)" in source
    assert "Vnext(x) = V(x) + H *" in source
    assert "numb xmp = V(x) + (numb)0.5 * H" in source
    assert "numb kx1 =" in source
    assert "evaluate_(" not in source and "derivative[" not in source
    assert "numb A[3][3]" in source
    assert "const numb eps" not in source and "shifted[" not in source
    assert "Exact affine solution of the reverse VSCD coordinate equation" in source
    assert "iteration < 6" not in source
    assert "oldY[v] - H * f" not in source
    assert "const int THREADS_PER_BLOCK_(name) = 64;" in header


def test_symbolic_jacobian_expands_algebraic_dependencies_and_abs():
    spec = parse_system("dx/dt = q + abs(x)\nq = x*x", "analytic", "Analytic")
    spec.methods = ["ImplicitEuler"]
    source = generate(spec).files["systems/analytic/analytic.cu"]
    jacobian_line = next(line for line in source.splitlines() if "numb A[1][1]" in line)
    assert "2*" in jacobian_line
    assert "> 0" in jacobian_line and "< 0" in jacobian_line
    assert "Vnext(q)" not in jacobian_line


def test_vscd_uses_analytic_scalar_newton_for_nonlinear_coordinate():
    spec = parse_system("dx/dt = sin(x) + a", "nonlinear_vscd", "Nonlinear VSCD")
    spec.methods = ["VariableSymmetryCD"]
    source = generate(spec).files["systems/nonlinear_vscd/nonlinear_vscd.cu"]
    assert "analytic scalar Newton" in source
    assert "vscdDerivative_x" in source
    assert "cos(vscdCandidate_x)" in source
    assert "evaluate_(name)(work" not in source


def test_metadata_writes_count_ranges_for_variables_and_parameters():
    spec = parse_system("dx/dt = a*x", "ranges", "Ranges")
    spec.variable_defaults["x"].range_kind = "Linear"
    spec.variable_defaults["x"].value = -2
    spec.variable_defaults["x"].maximum = 3
    spec.variable_defaults["x"].count = 51
    spec.parameter_defaults["a"].range_kind = "Linear"
    spec.parameter_defaults["a"].value = 0.1
    spec.parameter_defaults["a"].maximum = 1.1
    spec.parameter_defaults["a"].count = 11
    metadata = generate(spec).files["systems/ranges/ranges.txt"]
    assert "var x Linear -2 3 0.01 51" in metadata
    assert "param a Linear 0.1 1.1 0.01 11" in metadata


def test_real_project_install_plan_is_read_only_and_complete():
    root = Path(__file__).resolve().parents[2] / "cudaynamics-master"
    if not root.exists():
        pytest.skip("CUDAynamics project is not adjacent")
    spec = parse_system("dx/dt = a*x", "builder_test_only", "Builder test")
    plan = build_plan(root, spec, generate(spec))
    assert len(plan.writes) == 7
    assert "addKernel(builder_test_only)" in plan.diff
    assert not (root / "systems" / "builder_test_only").exists()
