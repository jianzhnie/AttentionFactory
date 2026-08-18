---
name: python-code-quality
description: Review, refactor, and generate production-quality Python with consistent naming, typing, structure, documentation, tests, security, and performance practices. Use for Python code reviews, cleanup, modernization, file or API renaming, best-practice audits, maintainability improvements, and requests mentioning code style, clean code, refactoring, optimization, type hints, or coding standards.
---

# Python Code Quality

Improve Python without changing behavior unless the user explicitly requests a
behavioral change. Follow the repository's established conventions before
applying general preferences.

## Workflow

1. Inspect `pyproject.toml`, lint/type/test configuration, package layout,
   public exports, and nearby modules.
2. Check the worktree and preserve unrelated changes.
3. Identify correctness issues before style issues. Establish the current test
   baseline when practical.
4. Define the public API and compatibility boundary before renaming files,
   classes, functions, or parameters.
5. Make the smallest coherent refactor that improves clarity and removes real
   duplication.
6. Add or update focused tests for changed behavior and public imports.
7. Run the repository's formatter, linter, type checker, tests, and pre-commit
   hooks when available. Report skipped checks and environment limitations.

For review-only requests, do not edit files. Lead with findings ordered by
severity and include file and line references.

## Naming

- Use descriptive `snake_case` for modules, packages, functions, parameters,
  and variables.
- Use `PascalCase` for classes and exceptions.
- Use `UPPER_SNAKE_CASE` for module constants.
- Preserve conventional acronym casing in class names, such as `RMSNorm`,
  `ALiBiAttention`, and `HTTPClient`.
- Name constructors and factories `build_*` or `create_*`; use `get_*` for one
  value and `list_*` for collections.
- Prefer names that describe responsibility instead of implementation detail.
- Avoid abbreviations unless they are standard in the domain.
- Keep a single canonical implementation path. Add compatibility aliases only
  when backward compatibility is required; remove them when the user
  explicitly accepts a breaking change.

## Types And APIs

- Type every public function, method, and class attribute whose type is not
  obvious from assignment.
- Use Python 3.10+ union syntax (`X | None`) and `collections.abc` protocols.
- Prefer `dataclass` for passive data and `Protocol` for structural interfaces.
- Return structured results instead of ambiguous tuples when field meaning is
  not obvious.
- Validate inputs at trust boundaries and fail with specific, actionable
  exceptions.
- Avoid widening public APIs during an unrelated refactor.

## Structure

- Keep modules cohesive and organize packages by responsibility.
- Keep functions focused; extract helpers when they remove nesting or repeated
  logic, not merely to reduce line count.
- Prefer early returns over deeply nested branches.
- Use dependency injection for external services and mutable infrastructure.
- Avoid global mutable state, wildcard imports, circular imports, and hidden
  import side effects.
- Remove dead code, stale shims, unused imports, and commented-out code after
  confirming they are not public compatibility surfaces.
- Prefer standard-library and repository-native abstractions over new
  dependencies.

## Correctness, Security, And Performance

- Catch specific exceptions and preserve context with `raise ... from exc`.
- Never silently swallow failures or expose secrets in code and logs.
- Use parameterized database queries and validated structured input.
- Use `pathlib.Path` for filesystem paths and `logging` instead of `print` in
  libraries.
- Measure before optimizing. State the workload and benchmark method.
- Fix algorithmic and I/O costs before micro-optimizing syntax.
- Avoid blocking I/O in async code, N+1 queries, unnecessary materialization,
  and repeated work in hot loops.

## Documentation And Comments

- Add concise module and public API docstrings that explain purpose, inputs,
  outputs, exceptions, and non-obvious constraints.
- Document why a design exists; do not narrate self-explanatory statements.
- Keep terminology and examples synchronized with the actual API.
- Update package maps or migration notes when paths or public names change.

## Tests

- Cover normal behavior, boundary cases, invalid inputs, gradients or state
  transitions when relevant, and public import paths after reorganizing code.
- Use descriptive `test_<behavior>_<condition>` names and parametrization for
  equivalent cases.
- Keep tests deterministic, independent, and focused on observable behavior.
- Do not weaken assertions merely to make a refactor pass.

## Completion Checklist

- Confirm behavior and compatibility match the requested scope.
- Confirm names and file layout are consistent across implementation, exports,
  tests, documentation, and registries.
- Confirm public APIs have useful type hints and docstrings.
- Confirm error messages identify the invalid value or condition.
- Confirm no secrets, bare `except`, dead code, or accidental debug output
  remain.
- Run relevant tests, formatter, linter, type checker, `git diff --check`, and
  `pre-commit run --all-files` when configured.
- Summarize changed behavior, breaking import changes, verification results,
  and residual production risks.
