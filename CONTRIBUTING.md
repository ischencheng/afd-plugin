# Contributing to AFD Plugin

Before implementing a contribution, complete the human
[contributor intent gate](.agents/skills/precheck-pr/SKILL.md#contributor-intent-gate).
Before opening a PR, run the [precheck-pr skill](.agents/skills/precheck-pr/SKILL.md)
with your coding agent, or follow its questions and evidence checks yourself.
Reuse explicit human answers already supplied; agents must ask for missing
answers and wait. The gate checks whether the outcome needs a PR and requires
actual failure evidence for bug fixes.

Follow [AGENTS.md](AGENTS.md) for code style and upstream patch discipline.
See [Development](README.md#development) for local checks and the
[run-e2e skill](.agents/skills/run-e2e/SKILL.md) for hardware E2E.

Keep the change focused and report what you actually validated, including
unavailable checks. The contributor gate does not change independent-review
eligibility, maintainer review, or CI requirements.
