<!-- SPDX-FileCopyrightText: 2026 OmniNode.ai Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Selector ruling: what proof a `.github/workflows`-only diff requires

**Ticket:** OMN-16745 · **Applies to:** `scripts/ci/detect_test_paths.py`,
`scripts/hooks/prepush_smart_tests.sh`

## Ruling

For a `.github/workflows`-only diff the **necessary and sufficient** proof is the
**CI-contract class** — `tests/ci/` (`CI_CONTRACT_TEST_ROOT`), the workflow-shape,
required-context and gate-wiring tests that read `.github/workflows/**` off disk and
assert its contents — plus, when the diff also touches a test module, that module
itself. The Python unit suite is neither necessary nor sufficient: no test under
`tests/unit/` has an outcome a workflow YAML edit can change, so escalating this class
to the full unit suite is cost without proof.

Because a workflow edit breaks the **enforcement** of tests rather than the tests
themselves, the class may never select *nothing*. It is positively named, always
non-empty, and its suite is asserted to be populated and workflow-aware by a test.

## Why the unit suite is not the proof

`tests/unit/` collects no module whose pass/fail depends on the content of a workflow
file. Running it against a workflow diff produces a green that is true regardless of
what the diff did — a result with zero discriminating power.

The cost is not theoretical. OMN-16346 held a committed, pre-commit-clean workflow fix
through roughly twenty refused pushes across both designated gate hosts — with zero
bypass-token use, zero `PREPUSH_ALLOW_*`, zero `--no-verify` — because the diff
escalated to a whole-suite-equivalent selection and the OMN-16295 / OMN-15059 load
guard correctly refused to run it on an oversubscribed host. Compliance was perfect;
the fix still did not ship. Cost-without-proof is precisely what trains operators and
agents to want a bypass, so removing it is a *hardening* of the gate, not a relaxation.

## Why "skip it" is the wrong inference — the OMN-15541 counterexample

"Workflow YAML cannot break Python tests, therefore run nothing" does not follow.
Workflow files break the **enforcement** of tests, which is worse and invisible.

OMN-15541 is the live proof. `ci.yml` hardcoded `pytest src/omnibase_compat/tests/`
while the selector and `pyproject.toml` named different roots. The result was that
full-suite escalation — the strongest thing the system knows how to do — collected
**zero** of the top-level `tests/` tree. A workflow edit had converted the safety net
itself into a fail-**open** surface, and no Python test failed to say so. Sibling
failure modes are the same shape: a renamed job id silently drops a required status
check (`project_public_repos_use_github_hosted_runners` — "never rename job ids"); a
changed `on:` trigger disables a gate.

That is exactly what `tests/ci/` exists to catch, and exactly why the ruling is a
*substitution* of proof rather than a removal of it.

## Fail-closed properties preserved

| Diff | Outcome |
|------|---------|
| `.github/workflows/**` only | CI-contract class (`tests/ci/`), narrow |
| workflow + a shared module (`src/omnibase_infra/models/**`) | full suite, `shared_module` |
| workflow + test infrastructure (`tests/conftest.py`, `pyproject.toml`, …) | full suite, `test_infrastructure` |
| workflow + a root-level `tests/` module pytest cannot collect | full suite, `changed_test_unnarrowable` |
| workflow + an ordinary source file | CI-contract class **plus** that file's own narrowing, additively |

No environment override, no allowlist mapping workflow paths to zero tests, and no
bypass token is introduced anywhere. `ENABLE_SMART_TESTS` and `PREPUSH_FULL_SUITE` are
untouched and can still only make the hook run *more* tests.

## The second half: root-level test modules are narrowable after all

The stranded shape in this repo is a workflow edit paired with **its own test**. Where
that test sits directly in the `tests/` root, the pre-OMN-16745 selector escalated the
whole diff to the 15-split full suite under `changed_test_unnarrowable`.

The original rule (OMN-15245) conflated two different claims: "has no containing
directory below `tests/`" and "cannot be narrowed". A root-level module *is*
narrowable — to itself. `ModelTestSelection` could not express that (its `TestPath`
pattern admitted only directories), so escalation was the only reachable answer. The
selector now emits the module at **file grain**, which is strictly narrower than the
`tests/` directory the escalation existed to avoid emitting, and strictly covers the
changed module.

The escalation survives, on positive evidence, for the population that genuinely cannot
be narrowed: a root-level module pytest would **not** collect (it matches neither
`python_files` pattern) — e.g. `tests/infrastructure_config.py`. Handing such a module
to pytest collects nothing (exit 5), and any suite in the tree may import it, so its
blast radius really is the whole tree.

`TEST_FILE_PATTERNS` in the selector is held equal to `[tool.pytest.ini_options]
python_files` by a test, so widening pytest's collection rule without widening the
classifier fails a test rather than silently misclassifying a module.

## Enforcement (Operating Rule #5)

The class is not advisory. It is asserted by tests that run on every PR:

- `tests/unit/scripts/ci/test_detect_test_paths.py` — the workflow-only diff selects
  `CI_CONTRACT_TEST_ROOT` deterministically (the *class* is asserted by name, not a
  smaller test count); the class is a populated, workflow-aware suite; every mixed-diff
  row in the table above.
- `tests/unit/scripts/test_prepush_smart_tests_seam.py` — the hook's real
  `filter_prepush_runnable_paths` and `selection_is_whole_suite` functions are
  *executed* (not grepped) to prove the class reaches pytest's argv and does not trip
  the heavy-selection load guard.
