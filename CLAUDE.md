# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> [!IMPORTANT]
> **MULTI-AGENT COORDINATION & ANTIGRAVITY WATCHDOG:**
> - Antigravity is Joel's active peer agent maintaining the continuous watchdog on training runs, GPU telemetry, loss checks, and post-chains.
> - **Never invoke `gemini.exe` headlessly from subshells** (it requires interactive login and fails with auth errors).
> - Whenever starting, pausing, or evaluating runs, **ping and invoke Antigravity** by posting to `out/swarm/COORDINATION_120M.md` and logging handoffs to `F:\Lingual_Data\Agents-Brain\journal.md`. Antigravity will keep the watchdog active.

All agent instructions live in the repo-root `AGENTS.md` (shared by every agent Joel uses). Imported here:

@AGENTS.md

