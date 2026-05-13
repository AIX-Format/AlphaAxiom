# AGENTS.md — Operating Manual for AI Coding Agents

> 📜 **Stack-wide protocol rules**: read [`AXIOM.md`](https://github.com/Moeabdelaziz007/aix-format/blob/main/AXIOM.md) first. This file complements it with repo-local operating instructions for AlphaAxiom.

## Repository overview

`AlphaAxiom` is the trading product line under the `axiomid.app` authority. Includes the `money-machine/` trading runtime (Python), a Telegram bot, and public dashboards (`aqt.axiomid.app`, `oracle.axiomid.app`). It is a sibling product to the Sovereign Stack rather than a member of it; cross-stack rules from `AXIOM.md` still apply (license policy, naming, sovereignty).

## Conventions

- **License**: MIT (matches the README badge; consistent with public trading product).
- **Branches**: kebab-case (`feat/...`, `fix/...`, `chore/...`).
- **Conventional Commits** preferred.
- **Sensitive paths**: anything under `money-machine/src-python/` that touches live trading must carry a maintainer review.

## What to read before opening a PR

1. [`AXIOM.md`](https://github.com/Moeabdelaziz007/aix-format/blob/main/AXIOM.md) — the stack-wide constitution.
2. The architecture doc closest to the area you are changing (`ARCHITECTURE.md`, `BRAINTRUST_SETUP.md`).
3. The closest neighbouring file in the same directory as the change.

## Relationship to the Sovereign Stack

AlphaAxiom is not in the strict L1 / L2 / L3 chain, but it shares the `axiomid.app` root authority. If your change touches identity or signing primitives, prefer consuming `@axiom/identity` once it is published rather than re-implementing.
