"""Static validation for AI-generated support-set selection strategies.

Defense-in-depth, not a guarantee: these AST checks layer on top of the
out-of-process, argv-only, environment-restricted execution in
``strategy_runner.py``. A hostile strategy could still waste CPU or memory
inside its process until the runner timeout kills it; these checks exist to
reject the common accidental and obvious malicious cases before any process
is spawned, and to make human review of the preregistered plan tractable.

Contract enforced here: the source must be exactly one module-level
``def select(candidate_ids, embeddings, k, seed) -> list[str]``. The trusted
template (``strategy_runner.assemble_strategy_file``) already imports the
allowed modules at the top of the file, so the function body must not import
anything itself.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

BUILTIN_STRATEGIES = frozenset({"random", "k_center"})

BUILTIN_DETECTORS = frozenset({"anomalydino", "patchcore", "subspacead"})

GENERATED_DETECTOR_PREFIX = "generated_det_"

ALLOWED_IMPORT_MODULES = frozenset(
    {
        "math",
        "random",
        "hashlib",
        "itertools",
        "collections",
        "statistics",
        "functools",
        "numpy",
    }
)

ALLOWED_DETECTOR_IMPORT_MODULES = frozenset(
    {
        "math",
        "random",
        "hashlib",
        "itertools",
        "collections",
        "statistics",
        "functools",
        "numpy",
        "scipy",
        "sklearn",
        "PIL",
        "cv2",
        "torch",
        "torchvision",
        "transformers",
        "timm",
    }
)

FORBIDDEN_FUNC_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
        "bytearray",
        "hash",
    }
)

FORBIDDEN_NAMES = frozenset({"__builtins__", "__import__"})

ALLOWED_STRATEGY_DIRECT_CALLS = frozenset(
    {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "next",
        "pow",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)

REQUIRED_PARAMETERS = ("candidate_ids", "embeddings", "k", "seed")

REQUIRED_DETECTOR_PARAMETERS = ("image", "support_images", "seed")

MAX_SOURCE_CHARS = 8000
MAX_TOP_LEVEL_STATEMENTS = 60
MAX_DETECTOR_SOURCE_CHARS = 30000


@dataclass
class CodeValidationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


class _StrategyVisitor(ast.NodeVisitor):
    """Flag forbidden constructs anywhere inside the select function body."""

    def __init__(self, *, allow_global_helpers: bool = False) -> None:
        self.issues: list[str] = []
        self.allow_global_helpers = allow_global_helpers

    def visit_Import(self, node: ast.Import) -> None:
        self.issues.append("select 函数体内不允许 import 语句（模板已导入允许的模块）")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.issues.append("select 函数体内不允许 import 语句（模板已导入允许的模块）")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self.issues.append(f"禁止访问内置对象 {node.id!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_FUNC_CALLS:
            self.issues.append(f"禁止调用 {node.func.id!r}")
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id not in ALLOWED_STRATEGY_DIRECT_CALLS
            and not self.allow_global_helpers
        ):
            self.issues.append(
                f"禁止调用未由模板提供的全局函数 {node.func.id!r}；请把计算直接写入 select"
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.issues.append("select 函数体内不允许定义类")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.issues.append("select 函数体内不允许嵌套函数定义")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.issues.append("select 必须是普通同步函数（禁止 async）")
        self.generic_visit(node)


def validate_strategy_source(source: str) -> CodeValidationResult:
    """Validate that ``source`` is exactly one safe, well-formed select function."""

    if len(source) > MAX_SOURCE_CHARS:
        return CodeValidationResult(
            False,
            [f"源码长度 {len(source)} 超过上限 {MAX_SOURCE_CHARS} 字符"],
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return CodeValidationResult(False, [f"语法错误：{exc.msg}（第 {exc.lineno} 行）"])

    issues: list[str] = []
    statements = [
        item
        for item in tree.body
        if not (
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        )
    ]
    if not statements or len(statements) != 1 or not isinstance(statements[0], ast.FunctionDef):
        issues.append("源码必须恰好包含一个模块级 def select 函数，不得有其他语句")
        return CodeValidationResult(False, issues)

    function = statements[0]
    if function.name != "select":
        issues.append(f"函数必须命名为 select，实际为 {function.name!r}")
    issues.extend(_validate_signature(function))
    if function.decorator_list:
        issues.append("select 不允许使用装饰器")
    if len(function.body) > MAX_TOP_LEVEL_STATEMENTS:
        issues.append(
            f"select 函数体语句数 {len(function.body)} 超过上限 {MAX_TOP_LEVEL_STATEMENTS}"
        )

    visitor = _StrategyVisitor()
    for statement in function.body:
        visitor.visit(statement)
    issues.extend(visitor.issues)

    return CodeValidationResult(not issues, sorted(set(issues)))


def _validate_signature_shape(
    function: ast.FunctionDef,
    required: tuple[str, ...],
    label: str,
) -> list[str]:
    issues: list[str] = []
    arguments = function.args
    parameter_names = [item.arg for item in arguments.args]
    if parameter_names != list(required):
        issues.append(f"{label} 参数必须恰好为 {required}，实际为 {parameter_names}")
    if arguments.posonlyargs or arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
        issues.append(f"{label} 不允许位置限定参数、*args、**kwargs 或关键字限定参数")
    if arguments.defaults:
        issues.append(f"{label} 参数不允许默认值")
    return issues


def _validate_signature(function: ast.FunctionDef) -> list[str]:
    issues = _validate_signature_shape(function, REQUIRED_PARAMETERS, "select")
    if function.returns is not None and not _is_list_of_str(function.returns):
        issues.append("select 的返回注解必须是 list[str]（或省略）")
    return issues


def _validate_detector_signature(function: ast.FunctionDef) -> list[str]:
    issues = _validate_signature_shape(
        function, REQUIRED_DETECTOR_PARAMETERS, "anomaly_score"
    )
    if function.returns is not None and not (
        isinstance(function.returns, ast.Name) and function.returns.id == "float"
    ):
        issues.append("anomaly_score 的返回注解必须是 float（或省略）")
    return issues


def _is_list_of_str(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "list"
        and isinstance(node.slice, ast.Name)
        and node.slice.id == "str"
    )


def extract_select_function(text: str) -> str:
    """Extract the single module-level ``def select`` from an LLM reply.

    Strips markdown fences and silently discards every statement outside the
    function (preamble prints, module-level imports, trailing explanations).
    """

    cleaned = _strip_fences(text)
    try:
        tree = ast.parse(cleaned)
    except SyntaxError as exc:
        raise ValueError(f"LLM 返回的代码无法解析：{exc.msg}（第 {exc.lineno} 行）") from exc
    candidates = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "select"
    ]
    if not candidates:
        raise ValueError("LLM 返回中没有模块级 select 函数")
    if len(candidates) > 1:
        raise ValueError("LLM 返回中包含多个 select 定义")
    segment = ast.get_source_segment(cleaned, candidates[0])
    if segment is None:
        raise ValueError("无法定位 select 函数的源码片段")
    return segment


def sanitize_strategy_name(stem: str) -> str:
    """Normalize an LLM-suggested stem to ^[a-z][a-z0-9_]{2,63}$ (max 55 chars)."""

    candidate = re.sub(r"[^a-z0-9_]+", "_", stem.strip().casefold()).strip("_") or "strategy"
    if candidate[0].isdigit():
        candidate = f"s_{candidate}"
    if len(candidate) > 55:
        candidate = candidate[:55].rstrip("_")
    if candidate in BUILTIN_STRATEGIES:
        candidate = f"{candidate}_custom"
    if len(candidate) < 3:
        candidate = f"{candidate}_strategy"
    return candidate


class _DetectorVisitor(_StrategyVisitor):
    """Strategy rules plus download-API and structure bans for detector code."""

    def __init__(self) -> None:
        super().__init__(allow_global_helpers=True)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = ast.unparse(node.func)
        if dotted == "hub.load" or ".hub.load" in dotted:
            self.issues.append("禁止调用 torch.hub.load / hub.load（离线环境禁止下载）")
        super().visit_Call(node)


def validate_detector_source(source: str) -> CodeValidationResult:
    """Validate an LLM-authored detector section.

    The section must contain module-level imports from the detector allowlist,
    top-level functions only (any helpers), exactly one ``anomaly_score`` with
    signature ``(image, support_images, seed)``, no names starting with ``_``
    (the trusted template owns that namespace), and no IO/download anywhere.
    """

    if len(source) > MAX_DETECTOR_SOURCE_CHARS:
        return CodeValidationResult(
            False,
            [f"源码长度 {len(source)} 超过上限 {MAX_DETECTOR_SOURCE_CHARS} 字符"],
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return CodeValidationResult(False, [f"语法错误：{exc.msg}（第 {exc.lineno} 行）"])

    issues: list[str] = []
    functions: list[ast.FunctionDef] = []
    for item in tree.body:
        if (
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
        ):
            continue
        if isinstance(item, ast.Import):
            for alias in item.names:
                if alias.name not in ALLOWED_DETECTOR_IMPORT_MODULES:
                    issues.append(f"禁止导入模块 {alias.name!r}")
            continue
        if isinstance(item, ast.ImportFrom):
            if item.level > 0:
                issues.append("禁止相对导入")
                continue
            top_level = (item.module or "").split(".")[0]
            if top_level not in ALLOWED_DETECTOR_IMPORT_MODULES or any(
                alias.name == "*" for alias in item.names
            ):
                issues.append(f"禁止从模块 {item.module!r} 导入")
            continue
        if isinstance(item, ast.FunctionDef):
            functions.append(item)
            continue
        if isinstance(item, ast.AsyncFunctionDef):
            issues.append("anomaly_score 与辅助函数必须是普通同步函数（禁止 async）")
            continue
        issues.append("模块级只允许 import 语句和函数定义")

    anomaly_functions = [item for item in functions if item.name == "anomaly_score"]
    if len(anomaly_functions) != 1:
        issues.append("必须恰好包含一个模块级 anomaly_score 函数")
    for function in functions:
        if function.name.startswith("_"):
            issues.append(
                f"函数名不允许以下划线开头（模板保留该命名空间）：{function.name}"
            )
        if function.decorator_list:
            issues.append(f"{function.name} 不允许使用装饰器")
    if len(anomaly_functions) == 1:
        issues.extend(_validate_detector_signature(anomaly_functions[0]))

    visitor = _DetectorVisitor()
    for function in functions:
        for statement in function.body:
            visitor.visit(statement)
    issues.extend(visitor.issues)

    return CodeValidationResult(not issues, sorted(set(issues)))


def extract_detector_source(text: str) -> str:
    """Strip markdown fences from an LLM reply; the validator rejects the rest."""

    return _strip_fences(text)


def sanitize_detector_name(stem: str) -> str:
    """Normalize a detector stem to ^[a-z][a-z0-9_]{2,63}$ (max 55 chars)."""

    candidate = re.sub(r"[^a-z0-9_]+", "_", stem.strip().casefold()).strip("_") or "detector"
    if candidate[0].isdigit():
        candidate = f"d_{candidate}"
    if len(candidate) > 55:
        candidate = candidate[:55].rstrip("_")
    if candidate in BUILTIN_DETECTORS:
        candidate = f"{candidate}_custom"
    if len(candidate) < 3:
        candidate = f"{candidate}_detector"
    return candidate


def implementation_detector_name(stem: str, code_digest: str) -> str:
    """Compose the registered name of a generated detector implementation."""

    return f"{GENERATED_DETECTOR_PREFIX}{sanitize_detector_name(stem)}_{code_digest[:8]}"


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped
