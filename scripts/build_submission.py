from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import py_compile
import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


OUTPUT_FILENAME = "TMHS.py"

# Search these directories for strategy classes.
STRATEGY_SEARCH_DIRECTORIES = (
    "backtesting/baselines",
    "strategies",
)

# Classes that should not be offered as concrete submissions.
EXCLUDED_STRATEGY_CLASSES = {
    "BaselineStrategy",
    "MetaStrategy",
    "Strategy",
}

@dataclass(frozen=True, slots=True)
class StrategyPreset:
    strategy: str
    kwargs: dict[str, object]

PRESETS: dict[str, StrategyPreset] = {
    "information_leader_follower": StrategyPreset(
        strategy=(
            "strategies.information_graph_strategies:"
            "LeaderFollowerStrategy"
        ),
        kwargs={
            "window": 85,
            "lag": 2,
            "top_k": 7,
            "percentile": 97.0,
            "signal_lookback": 2,
            "rebalance_interval": 3,
            "self_move_penalty": 0.125,
            "normalise_incoming_weights": True,
            "signal_version": "raw",
        },
    ),
    "mean_reversion_20d": StrategyPreset(
        strategy=(
            "backtesting.baselines.cross_sectional_rank:"
            "CrossSectionalRankStrategy"
        ),
        kwargs={
            "lookback": 20,
            "selection_fraction": 0.20,
            "mode": "reversal",
            "exposure_fraction": 1.0,
            "include_algo": False,
        },
    ),
    "momentum_20d": StrategyPreset(
        strategy=(
            "backtesting.baselines.cross_sectional_rank:"
            "CrossSectionalRankStrategy"
        ),
        kwargs={
            "lookback": 20,
            "selection_fraction": 0.20,
            "mode": "momentum",
            "exposure_fraction": 1.0,
            "include_algo": False,
        },
    ),
    "mean_reversion_5d": StrategyPreset(
        strategy=(
            "backtesting.baselines.cross_sectional_rank:"
            "CrossSectionalRankStrategy"
        ),
        kwargs={
            "lookback": 5,
            "selection_fraction": 0.20,
            "mode": "reversal",
            "exposure_fraction": 1.0,
            "include_algo": False,
        },
    ),
    "previous_day_momentum": StrategyPreset(
        strategy=(
            "backtesting.baselines.previous_return:"
            "PreviousReturnStrategy"
        ),
        kwargs={
            "mode": "momentum",
            "exposure_fraction": 1.0,
        },
    ),
}

@dataclass(frozen=True, slots=True)
class StrategyChoice:
    module_name: str
    class_name: str
    file_path: Path

    @property
    def identifier(self) -> str:
        return f"{self.module_name}:{self.class_name}"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def path_to_module(
    path: Path,
    root: Path,
) -> str:
    relative = path.resolve().relative_to(
        root.resolve()
    )

    parts = list(relative.with_suffix("").parts)

    # package/__init__.py represents package, not package.__init__
    if parts[-1] == "__init__":
        parts.pop()

    return ".".join(parts)


def module_to_path(
    module_name: str,
    root: Path,
) -> Path | None:
    module_path = root.joinpath(
        *module_name.split(".")
    )

    python_file = module_path.with_suffix(".py")

    if python_file.exists():
        return python_file.resolve()

    package_file = module_path / "__init__.py"

    if package_file.exists():
        return package_file.resolve()

    return None


def parse_file(
    path: Path,
) -> ast.Module:
    source = path.read_text(
        encoding="utf-8"
    )

    return ast.parse(
        source,
        filename=str(path),
    )


def discover_strategies(
    root: Path,
) -> list[StrategyChoice]:
    choices: list[StrategyChoice] = []

    for directory_name in STRATEGY_SEARCH_DIRECTORIES:
        directory = root / directory_name

        if not directory.exists():
            continue

        for path in directory.rglob("*.py"):
            if path.name == "__init__.py":
                continue

            try:
                tree = parse_file(path)
            except SyntaxError:
                continue

            module_name = path_to_module(
                path,
                root,
            )

            for node in tree.body:
                if not isinstance(
                    node,
                    ast.ClassDef,
                ):
                    continue

                if node.name in EXCLUDED_STRATEGY_CLASSES:
                    continue

                if not node.name.endswith("Strategy"):
                    continue

                choices.append(
                    StrategyChoice(
                        module_name=module_name,
                        class_name=node.name,
                        file_path=path,
                    )
                )

    return sorted(
        choices,
        key=lambda choice: (
            choice.class_name.lower(),
            choice.module_name.lower(),
        ),
    )


def select_strategy_interactively(
    choices: list[StrategyChoice],
) -> StrategyChoice:
    if not choices:
        raise RuntimeError(
            "No concrete Strategy classes were found."
        )

    print("\nAvailable strategy classes\n")

    for index, choice in enumerate(
        choices,
        start=1,
    ):
        print(
            f"{index:>2}. "
            f"{choice.class_name} "
            f"({choice.module_name})"
        )

    while True:
        raw_choice = input(
            "\nSelect a strategy number: "
        ).strip()

        try:
            selected_index = int(raw_choice) - 1
        except ValueError:
            print("Enter a valid number.")
            continue

        if not (
            0
            <= selected_index
            < len(choices)
        ):
            print("Selection is out of range.")
            continue

        return choices[selected_index]


def parse_strategy_identifier(
    identifier: str,
    root: Path,
) -> StrategyChoice:
    try:
        module_name, class_name = (
            identifier.rsplit(":", maxsplit=1)
        )
    except ValueError as error:
        raise ValueError(
            "--strategy must use the form "
            "'package.module:ClassName'"
        ) from error

    path = module_to_path(
        module_name,
        root,
    )

    if path is None:
        raise FileNotFoundError(
            f"Could not locate local module "
            f"{module_name!r}."
        )

    tree = parse_file(path)

    class_exists = any(
        isinstance(node, ast.ClassDef)
        and node.name == class_name
        for node in tree.body
    )

    if not class_exists:
        raise ValueError(
            f"{class_name!r} was not found in "
            f"{module_name!r}."
        )

    return StrategyChoice(
        module_name=module_name,
        class_name=class_name,
        file_path=path,
    )


def resolve_imported_module(
    current_module: str,
    imported_module: str | None,
    level: int,
    root: Path,
) -> str:
    if level == 0:
        return imported_module or ""

    current_path = module_to_path(
        current_module,
        root,
    )

    if (
        current_path is not None
        and current_path.name == "__init__.py"
    ):
        # current_module already represents the package.
        current_package = current_module
    else:
        # A regular module belongs to its parent package.
        current_package = current_module.rpartition(".")[0]

    if not current_package:
        raise ImportError(
            "Cannot resolve relative import "
            f"from top-level module {current_module!r}."
        )

    relative_name = (
        "." * level
        + (imported_module or "")
    )

    return importlib.util.resolve_name(
        relative_name,
        current_package,
    )


def local_dependencies(
    module_name: str,
    tree: ast.Module,
    root: Path,
) -> list[str]:
    dependencies: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            resolved_module = resolve_imported_module(
                current_module=module_name,
                imported_module=node.module,
                level=node.level,
                root=root,
            )

            if module_to_path(
                resolved_module,
                root,
            ) is not None:
                dependencies.append(
                    resolved_module
                )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                if module_to_path(
                    alias.name,
                    root,
                ) is not None:
                    dependencies.append(
                        alias.name
                    )

    return dependencies


def dependency_order(
    target_module: str,
    root: Path,
) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module_name: str) -> None:
        if module_name in visited:
            return

        if module_name in visiting:
            # Circular imports are valid in some projects.
            # The module is already being processed.
            return

        path = module_to_path(
            module_name,
            root,
        )

        if path is None:
            return

        visiting.add(module_name)

        tree = parse_file(path)

        for dependency in local_dependencies(
            module_name,
            tree,
            root,
        ):
            visit(dependency)

        visiting.remove(module_name)
        visited.add(module_name)
        ordered.append(module_name)

    visit(target_module)

    return ordered


def is_local_import(
    node: ast.Import | ast.ImportFrom,
    current_module: str,
    root: Path,
) -> bool:
    if isinstance(node, ast.ImportFrom):
        module_name = resolve_imported_module(
            current_module=current_module,
            imported_module=node.module,
            level=node.level,
            root=root,
        )
        return (
            module_to_path(
                module_name,
                root,
            )
            is not None
        )

    return all(
        module_to_path(
            alias.name,
            root,
        )
        is not None
        for alias in node.names
    )


def allowed_top_level_node(
    node: ast.stmt,
) -> bool:
    """
    Keep definitions and constants, but remove scripts and other
    executable top-level statements.
    """
    if isinstance(
        node,
        (
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Assign,
            ast.AnnAssign,
            ast.Import,
            ast.ImportFrom,
        ),
    ):
        return True

    # Preserve a module docstring.
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ):
        return True

    return False



def _is_super_init_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False

    function = node.func

    return (
        isinstance(function, ast.Attribute)
        and function.attr == "__init__"
        and isinstance(function.value, ast.Call)
        and isinstance(function.value.func, ast.Name)
        and function.value.func.id == "super"
    )


def _baseline_state_initialiser() -> ast.Assign:
    return ast.Assign(
        targets=[
            ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr="current_positions",
                ctx=ast.Store(),
            )
        ],
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="zeros",
                ctx=ast.Load(),
            ),
            args=[
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="number_of_instruments",
                    ctx=ast.Load(),
                )
            ],
            keywords=[
                ast.keyword(
                    arg="dtype",
                    value=ast.Attribute(
                        value=ast.Name(id="np", ctx=ast.Load()),
                        attr="int64",
                        ctx=ast.Load(),
                    ),
                )
            ],
        ),
    )


class SubmissionSanitiser(ast.NodeTransformer):
    def visit_Expr(
        self,
        node: ast.Expr,
    ) -> ast.stmt | None:
        if _is_super_init_call(node.value):
            replacement = _baseline_state_initialiser()

            return ast.copy_location(
                replacement,
                node,
            )

        visited = self.generic_visit(node)

        if not isinstance(visited, ast.stmt):
            raise TypeError(
                "Expected generic_visit to return an ast.stmt."
            )

        return visited


def sanitise_definition_nodes(
    nodes: list[ast.stmt],
) -> list[ast.stmt]:
    sanitiser = SubmissionSanitiser()
    cleaned: list[ast.stmt] = []

    for node in nodes:
        transformed = sanitiser.visit(node)

        if transformed is None:
            continue

        if isinstance(transformed, list):
            cleaned.extend(transformed)
        else:
            cleaned.append(transformed)

    for node in cleaned:
        ast.fix_missing_locations(node)

    return cleaned


def validate_no_dunder_attribute_access(
    source: str,
) -> None:
    tree = ast.parse(source)

    blocked: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue

        if not (
            node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            continue

        is_allowed_object_setattr = (
            node.attr == "__setattr__"
            and isinstance(node.value, ast.Name)
            and node.value.id == "object"
        )

        if is_allowed_object_setattr:
            continue

        line_number = getattr(
            node,
            "lineno",
            "?",
        )

        blocked.append(
            f"line {line_number}: {node.attr}"
        )

    if blocked:
        raise RuntimeError(
            "Generated submission contains blocked "
            "dunder attribute access:\n"
            + "\n".join(blocked)
        )

def clean_module_nodes(
    module_name: str,
    tree: ast.Module,
    root: Path,
) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """
    Return:
        external import nodes,
        local definition/constant nodes.
    """
    external_imports: list[ast.stmt] = []
    definitions: list[ast.stmt] = []

    for node in tree.body:
        if not allowed_top_level_node(node):
            continue

        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue

            if is_local_import(
                node,
                module_name,
                root,
            ):
                continue

            external_imports.append(node)
            continue

        if isinstance(node, ast.Import):
            local_aliases: list[ast.alias] = []
            external_aliases: list[ast.alias] = []

            for alias in node.names:
                if module_to_path(
                    alias.name,
                    root,
                ) is None:
                    external_aliases.append(alias)
                else:
                    local_aliases.append(alias)

            if local_aliases:
                for alias in local_aliases:
                    if alias.asname is not None:
                        raise RuntimeError(
                            "Local module alias imports are not "
                            "supported by the submission builder: "
                            f"import {alias.name} as {alias.asname}"
                        )

            if external_aliases:
                external_imports.append(
                    ast.Import(
                        names=external_aliases
                    )
                )

            continue

        definitions.append(node)

    definitions = sanitise_definition_nodes(
        definitions
    )

    return external_imports, definitions


def deduplicate_imports(
    import_nodes: list[ast.stmt],
) -> list[ast.stmt]:
    unique: list[ast.stmt] = []
    seen: set[str] = set()

    for node in import_nodes:
        text = ast.unparse(node)

        if text in seen:
            continue

        seen.add(text)
        unique.append(node)

    return unique


def constructor_expression(
    class_name: str,
    kwargs: dict[str, Any],
) -> str:
    if not kwargs:
        return f"{class_name}()"

    arguments = ",\n".join(
        f"    {key}={value!r}"
        for key, value in kwargs.items()
    )

    return (
        f"{class_name}(\n"
        f"{arguments},\n"
        f")"
    )


def build_submission(
    strategy: StrategyChoice,
    constructor_kwargs: dict[str, Any],
    root: Path,
) -> Path:
    modules = dependency_order(
        strategy.module_name,
        root,
    )

    all_external_imports: list[ast.stmt] = []
    module_sections: list[str] = []

    for module_name in modules:
        path = module_to_path(
            module_name,
            root,
        )

        if path is None:
            raise RuntimeError(
                f"Lost local module {module_name!r}."
            )

        tree = parse_file(path)

        imports, definitions = clean_module_nodes(
            module_name,
            tree,
            root,
        )

        all_external_imports.extend(imports)

        if definitions:
            module_source = "\n\n".join(
                ast.unparse(node)
                for node in definitions
            )

            module_sections.append(
                "\n".join(
                    (
                        "# "
                        + "=" * 76,
                        f"# Inlined from {module_name}",
                        "# "
                        + "=" * 76,
                        module_source,
                    )
                )
            )

    external_imports = deduplicate_imports(
        all_external_imports
    )

    imports_source = "\n".join(
        ast.unparse(node)
        for node in external_imports
    )

    strategy_initialiser = constructor_expression(
        strategy.class_name,
        constructor_kwargs,
    )

    wrapper = f'''
# =============================================================================
# Competition entry point
# =============================================================================

_STRATEGY = {strategy_initialiser}


def getMyPosition(
    prices: np.ndarray,
) -> np.ndarray:
    """
    Return the desired integer share positions for all instruments.
    """
    price_history = np.asarray(
        prices,
        dtype=np.float64,
    )

    if price_history.ndim != 2:
        raise ValueError(
            "prices must have shape "
            "(number_of_instruments, number_of_days)"
        )

    positions = _STRATEGY.get_positions(
        price_history
    )

    positions = np.asarray(
        positions,
        dtype=np.int64,
    )

    expected_shape = (
        price_history.shape[0],
    )

    if positions.shape != expected_shape:
        raise ValueError(
            f"strategy returned shape {{positions.shape}}; "
            f"expected {{expected_shape}}"
        )

    return positions
'''

    generated_source = "\n\n".join(
        section
        for section in (
            "from __future__ import annotations",
            imports_source,
            *module_sections,
            wrapper.strip(),
        )
        if section.strip()
    )

    validate_no_dunder_attribute_access(
        generated_source
    )

    output_path = root / OUTPUT_FILENAME

    output_path.write_text(
        generated_source + "\n",
        encoding="utf-8",
    )

    return output_path


def validate_submission(
    output_path: Path,
) -> None:
    py_compile.compile(
        str(output_path),
        doraise=True,
    )

    spec = importlib.util.spec_from_file_location(
        "_tmhs_submission_validation",
        output_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load generated submission."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    module_name = spec.name
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(
            module
        )
    finally:
        sys.modules.pop(
            module_name,
            None,
        )

    get_position = getattr(
        module,
        "getMyPosition",
        None,
    )

    if not callable(get_position):
        raise RuntimeError(
            "TMHS.py does not define callable "
            "getMyPosition."
        )

    # Use enough positive, non-constant history to exercise strategies
    # with long windows and minimum-day requirements.
    rng = np.random.default_rng(0)
    test_returns = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=(51, 220),
    )
    test_prices = 100.0 * np.exp(
        np.cumsum(test_returns, axis=1)
    )

    positions = np.asarray(
        get_position(test_prices)
    )

    if positions.shape != (51,):
        raise RuntimeError(
            "getMyPosition returned shape "
            f"{positions.shape}; expected (51,)."
        )

    if not np.issubdtype(
        positions.dtype,
        np.integer,
    ):
        raise RuntimeError(
            "getMyPosition must return integer positions."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a project Strategy class and its local "
            "dependencies into standalone TMHS.py."
        )
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help=(
            "Strategy in the form "
            "'package.module:ClassName'. "
            "Omit for interactive selection."
        ),
    )

    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Use a predefined strategy configuration.",
    )

    return parser.parse_args()


def main() -> None:
    root = repository_root()

    # Ensure project modules can be resolved.
    sys.path.insert(
        0,
        str(root),
    )

    arguments = parse_arguments()

    if arguments.preset is not None:
        preset = PRESETS[arguments.preset]

        strategy = parse_strategy_identifier(
            preset.strategy,
            root,
        )

        constructor_kwargs = preset.kwargs

    elif arguments.strategy is not None:
        strategy = parse_strategy_identifier(
            arguments.strategy,
            root,
        )

        constructor_kwargs = {}

    else:
        choices = discover_strategies(root)

        strategy = select_strategy_interactively(
            choices
        )

        constructor_kwargs = {}

    print(
        f"\nBuilding {OUTPUT_FILENAME}"
        f"\nStrategy: {strategy.identifier}"
        f"\nArguments: {constructor_kwargs or '{}'}"
    )

    output_path = build_submission(
        strategy=strategy,
        constructor_kwargs=constructor_kwargs,
        root=root,
    )

    validate_submission(
        output_path
    )

    print(
        f"\nBuilt and validated:\n{output_path}"
    )


if __name__ == "__main__":
    main()