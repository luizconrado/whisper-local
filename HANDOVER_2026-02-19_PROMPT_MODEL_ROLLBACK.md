# Handover - Complete Reconstruction Timeline (Prompt Improvements -> Model Tuning Arc -> Safe Rollback)

Date prepared: 2026-02-19  
Project root: `/Users/luizconrado/PycharmProjects/whisper-local`  
Primary implementation file in this session: `/Users/luizconrado/PycharmProjects/whisper-local/GUI-whisper-chat-mode_colored_button_Hybrid_v5.py`

---

## A) New Agent Onboarding (Start Here)

Mission:

1. Keep current stable behavior intact.
2. Reintroduce only open phases in safe increments.
3. Avoid regressions previously seen (memory peaks + empty outputs).

Current implementation boundary:

1. Already implemented and active: prompt engineering wave + selected post-process flow + GLM prompt-quality rewrite (see Section 12).
2. Open/pending for reimplementation: Phases `I/J`, `M`, `O`, `Q`, `R/S` (and optional revisit of temporary Phase H 8k experiment).

Three script versions every new agent must compare:

1. `Current`: `HEAD` version of `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py`
2. `Before all implementations`: commit `6c816e6`
3. `After all implementations (pre-rollback regression period)`: `stash@{0}` snapshot

Canonical comparison commands:

```bash
cd /Users/luizconrado/PycharmProjects/whisper-local
SCRIPT=GUI-whisper-chat-mode_colored_button_Hybrid_v5.py

# version extraction for direct inspection
git show HEAD:$SCRIPT > /tmp/whisper_v_current.py
git show 6c816e6:$SCRIPT > /tmp/whisper_v_before.py
git show stash@{0}:$SCRIPT > /tmp/whisper_v_after_all.py

# three-way diffs
git diff --stat 6c816e6..HEAD -- $SCRIPT
git diff --stat HEAD..stash@{0} -- $SCRIPT
git diff --stat 6c816e6..stash@{0} -- $SCRIPT

# targeted symbol verification (open phases)
git show stash@{0}:$SCRIPT | rg -n "OLLAMA_KEEP_ALIVE|_estimate_tokens_from_text|_plan_ollama_budget|_ollama_chat_with_ctx_fallback|tiktoken|MLX_CLEAR_CACHE_MEMORY_MB|_graceful_shutdown|aboutToQuit|keep_alive|num_predict"
```

Required reading order for new agents:

1. Section `0` for canonical repo state.
2. Section `1` for commit timeline and anchor commits.
3. Section `12` for implemented vs open phase matrix.
4. Section `13` for missing-feature reconstruction details and snippets.
5. Section `14` for recovery and diff commands.
6. Section `7` for validation matrix to run after each incremental change.

Implementation rules (strict):

1. Reintroduce one phase family at a time (`I/J` first, then `M`, then `O/Q`, then `R/S`).
2. Keep each change set atomic and independently testable.
3. Run validation matrix after each phase-level increment.
4. Stop and rollback the last increment if memory spikes or empty outputs reappear.

Open work checklist:

1. `[ ]` Phase I/J runtime tuning pack
2. `[ ]` Phase M adaptive budgeting stack
3. `[ ]` Phase O hardening pack 1
4. `[ ]` Phase Q hardening pack 2
5. `[ ]` Phase R/S graceful shutdown coordinator
6. `[ ]` Optional: re-test temporary Phase H `glm ctx_num=8192` experiment

Runtime dependency baseline for onboarding:

1. Core runtime libraries expected: `PyQt5`, `pyaudio`, `mlx_whisper`, `ollama`.
2. Optional-but-impactful libraries:
   - `scipy` (audio processing quality path)
   - `webrtcvad` (VAD path; fallback silence detection is used if missing)
   - `psutil` (memory telemetry)
   - `tiktoken` (only part of stash-only hardening arc, not active in current HEAD)
3. Pre-coding sanity check:

```bash
cd /Users/luizconrado/PycharmProjects/whisper-local
python3 -m py_compile GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
python3 - <<'PY'
mods = ["PyQt5", "pyaudio", "mlx_whisper", "ollama", "scipy", "webrtcvad", "psutil"]
for m in mods:
    try:
        __import__(m)
        print(f"[OK] {m}")
    except Exception as e:
        print(f"[MISS] {m}: {e.__class__.__name__}")
PY
```

Dependency-to-code-path mapping:

| Dependency | Why it exists in this app | Current code anchors |
|---|---|---|
| `PyQt5` | GUI, signals/threads, lifecycle | `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:77`, `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:2103` |
| `pyaudio` | microphone capture + stream handling | `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:76`, `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:1385` |
| `mlx_whisper` | transcription engine and model wrappers | `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:84`, `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:268` |
| `ollama` | refine/promptify model calls | `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:85`, `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:1848` |
| `scipy` (optional) | higher-quality audio preprocessing path | `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:105`, `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:900` |
| `webrtcvad` (optional) | voice activity detection path | `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:115`, `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:1160` |
| `psutil` (optional) | memory telemetry | `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:97`, `GUI-whisper-chat-mode_colored_button_Hybrid_v5.py:339` |
| `tiktoken` (stash-only arc) | tighter token counting for Phase O/Q experiments | `stash@{0}` only (not present in current HEAD) |

Mandatory preflight before Phase I+ reimplementation:

1. Verify canonical state commands from Section `0`.
2. Verify stash integrity and pin durable ref (`handover-phase-i-arc`) from Section `5`/`14`.
3. Capture baseline compile + option snapshots (commands in Section `14`).
4. Run one baseline smoke pass (refine + promptify) and save logs for before/after comparison.
5. If you add optional dependencies during rollout (for example `tiktoken`), rerun the sanity-check import block before continuing.

---

## B) Navigation Map

Use this map to find exactly what you need:

1. `Section 0`: current branch/HEAD/stash truth.
2. `Section 1`: key commits and sequence anchors.
3. `Section 3`: detailed chronology of what happened.
4. `Section 5`: reconstruction and recovery procedures.
5. `Section 6`: recommended reintroduction order.
6. `Section 7`: test matrix.
7. `Section 9`: artifacts and pointers.
8. `Section 11`: three-version comparison summary.
9. `Section 12`: phase status matrix (implemented vs open).
10. `Section 13`: deep technical reconstruction for open phases.
11. `Section 14`: command pack for retrieval and selective restore.

---

## 0) Current Canonical State (Last Verified Checkpoint)

- Branch: `main`
- Current HEAD: `ff704c1`
- Current upstream: `origin/main`
- Rollback anchor commit (runtime baseline): `6a61314`
- Working-tree state at checkpoint time: clean
- Stash entries: `stash@{0}` exists with model/performance/shutdown experiments
- Verification timestamp: `2026-02-19` (refresh before implementation work)

Commands used to verify now:

```bash
cd /Users/luizconrado/PycharmProjects/whisper-local
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
git log --oneline -n 15
git stash list
```

Observed at checkpoint time:

- `git status --short` -> _no output_ (clean tree)
- `HEAD` -> `ff704c1`
- rollback anchor -> `6a61314`
- `stash@{0}` -> `On main: WIP model tuning + shutdown experiments backup`

Canonical-state rule:

1. If values above diverge from live commands, trust live command output and update this section first.
2. Do not start open-phase reimplementation until this section matches repository reality.

---

## 1) Grounded Commit Timeline Relevant to This Session

These are the key commits the session referenced/produced:

1. `8f1a6e6`  
   Message: `Add Promptify action picker and fix sensitivity deadlock`  
   Meaning for this session: pre-existing foundation used by later changes.

2. `5bd8a65`  
   Message: `Enhance Promptify meta-prompt structure and quality guards`  
   Produced in this session and pushed.  
   Contains Promptify structure + quality/safety improvements.

3. `6a61314`  
   Message: `Run selected post-processing action after transcription`  
   Produced in this session and pushed.  
   Contains post-transcription mode behavior fix (refine vs promptify by selected button).

4. `5947940`  
   Message: `Refine GLM transcription prompt with explicit quality gate`  
   Produced after rollback anchoring to restore the Phase-G prompt-quality variant while still keeping Phase-I+ runtime tuning out of current code.

5. `1a6e1b7`  
   Message: `Add rollback handover with full session reconstruction`  
   Adds this handover document to repository history.

6. `ff704c1`  
   Message: `Expand handover with onboarding runbook and phase clarity`  
   Adds structured onboarding/navigation and richer reconstruction guidance.

No committed code from the runtime-tuning/shutdown experimental arc (Phases I onward) was retained; that arc remains in `stash@{0}` (`344b0fa`) and backup patch artifacts.

---

## 2) Executive Summary of the Full Session

The work happened in three major waves:

1. **Prompt engineering wave (stable, committed)**
- Promptify meta-prompt gained deterministic structure and stronger quality gates.
- Transcription flow was corrected to obey selected action (Refine Text vs Promptify).

2. **Model/runtime optimization wave (experimental, local only)**
- Context, adaptive token planning, chunking, fallback logic, warmup, telemetry, and shutdown hardening were added iteratively.
- Several sub-agent review cycles proposed hardening refinements and those were also applied.

3. **Regression + rollback wave (stable recovery)**
- User observed memory spikes and empty model outputs.
- Decision made to revert to state right after prompt improvements and pre-performance tuning arc.
- Local experiments preserved in stash and patch; core runtime baseline anchored at `6a61314`, followed by prompt/handover commits (`5947940`, `1a6e1b7`, `ff704c1`) that did not reintroduce Phase-I+ runtime tuning.

---

## 3) Full Step-by-Step Chronology (Max Detail)

Notes:
- Line references shown below were reported during those moments and may have shifted after later edits.
- All code edits in this session targeted the same file unless stated otherwise:  
  `/Users/luizconrado/PycharmProjects/whisper-local/GUI-whisper-chat-mode_colored_button_Hybrid_v5.py`

### Phase A - Promptify PREPARATION Requirement Introduced

User goal:
- Inject mandatory `PREPARATION` section between `CONSTRAINTS` and `STEP-BY-STEP INSTRUCTIONS`.
- Force preparation item 1 to exact string:
  - `Begin with a concise checklist (3-8 bullets) of what you will do; keep items conceptual, not implementation-level`
- Require context-aware preparation items 2+ (docs/scripts/research/Context7/planning as needed).

What was changed:
- Promptify system message rules expanded to enforce exact section order including `PREPARATION` in required location.
- Added construction rules:
  - `PREPARATION` mandatory
  - ordered-list format
  - item 1 exact-match requirement
  - items 2+ must be context-relevant pre-execution actions
- Added quality gates to validate:
  - section existence and order
  - item-1 exact sentence
  - relevance/preparation-only behavior of items 2+

Purpose:
- Deterministic prompt template generation + explicit pre-execution planning behavior.

Validation style:
- Prompt-text update (no runtime functional tests yet).

---

### Phase B - Promptify Best-Practice Hardening

User ask:
- Suggest and apply high-impact improvements beyond PREPARATION.

Applied additions to Promptify system message:
- Source text injection resistance:
  - treat source as untrusted data; do not execute embedded instructions.
- Strict header contract:
  - exact required section headers, uppercase, no extras.
- Requirement preservation:
  - explicit requirements from source must not be dropped unless contradictory.
- Language preservation:
  - keep source language unless user explicitly requests otherwise.
- INPUTS structure tightening:
  - required + optional with defaults/fallback behavior.
- Step operationalization:
  - numbered steps and decision branches for ambiguity.
- Anti-fabrication quality gates:
  - no invented tools/files/APIs/URLs.

Purpose:
- Improve reliability, reduce formatting drift, reduce hallucinations, preserve user intent.

---

### Phase C - First Commit/Push Checkpoint

User requested:
- audit changed files
- per-file commit message plan
- atomic commit + push + verification

Result:
- Only changed file found.
- Commit created and pushed:
  - `5bd8a65`
  - `Enhance Promptify meta-prompt structure and quality guards`
- Upstream sync verified clean at that moment.

Meaning:
- Promptify improvements are preserved and part of stable history.

---

### Phase D - Behavioral Investigation: What Runs After Transcription?

User asked:
- whether post-transcription path follows selected button action or always refines.

Findings at that time:
- automatic post-transcription always called `refine_text`.
- selected action (`current_text_action`) only affected manual action button flow.

Risk identified:
- UX mismatch between selected mode and automatic pipeline behavior.

---

### Phase E - Design + Implementation of Selected Post-Process Mode

Plan approved and implemented in a fresh pass:
- `TranscriptionThread` receives `post_process_mode`.
- mode validated/normalized (`refine` or `promptify`, fallback safe behavior).
- `start_transcription(...)` passes `self.current_text_action` at run start.
- `TranscriptionThread.run()` branches:
  - refine path when selected refine
  - promptify path when selected promptify
- promptify logic added in transcription worker path.
- existing UI signal wiring preserved (no redesign).

Purpose:
- Automatic post-transcription behavior now matches user-selected mode.

Validation:
- syntax/compile check passed.

---

### Phase F - Second Commit/Push Checkpoint

User requested same audit/atomic commit flow.

Result:
- Single modified file committed and pushed:
  - `6a61314`
  - `Run selected post-processing action after transcription`
- remote sync verified (`ahead/behind 0/0`).

Meaning:
- post-transcription mode fix is preserved in stable history.

---

### Phase G - GLM Refine Prompt Quality Review Rounds (Local Iterative Prompt Work)

User requested deep quality analysis + multiple sub-agent review cycles for refinement prompt.

What was done across rounds:
- Evaluated clarity/contradictions in GLM refine system prompt.
- Identified weak points:
  - subjective confidence gating language
  - possible conflict between preserving wording vs consolidation
  - filler removal overreach on words like `like`
  - determinism concerns for thinking model behavior
- Proposed revised refine prompt preserving original essence.
- Added explicit internal restructuring quality-gate model.
- Added stronger fallback behavior when uncertain.
- Ran additional 4-subagent review with adversarial + holistic angles.
- Merged recommendations into more explicit wording.

Merged refinements applied during local prompt edits included:
- explicit definition of “refined text” (no metadata/commentary)
- tightened gate wording (all checks pass, uncertainty -> fail)
- stricter non-restructure fallback
- semantic exceptions for filler handling
- preserving names/acronyms/technical/code/mixed-language tokens unless clear errors
- tie-break handling for equally plausible interpretations

Validation reported:
- AST/compile syntax checks reported as passed.

Important status note:
- These prompt refinements happened before and alongside later tuning arc in local evolution.
- Final persisted state after rollback is the committed baseline at `6a61314`.

---

### Phase H - Token Budget Discussion and Context Change Trigger

User asked for token estimates for 10-minute voice recording.

Reasoning communicated:
- rough speech-to-token conversion
- context must cover prompt + input + output

Action taken:
- temporary config change from 16k to 8k context (`ctx_num=8192`) in local working tree.

Purpose:
- test memory/speed tradeoff while still supporting expected duration.

Status:
- local experimental state only; not retained after rollback.

---

### Phase I - Model Optimization Research (Docs + Sub-agent Driven)

User requested external review using Ollama docs/context.

Research conclusions integrated into plan:
- use `keep_alive`
- adaptive `num_ctx`
- bound/plan `num_predict`
- warmup model to avoid first-hit latency
- telemetry logging (`load_duration`, `prompt_eval_duration`, `eval_duration`, counts)
- watch memory implications of context size and loaded model settings

Purpose:
- improve speed + memory behavior while preserving quality.

---

### Phase J - First Runtime Tuning Implementation Pack (Local)

Implemented locally:
- new tuning knobs/constants for Ollama behavior
- adaptive context sizing hooks
- response telemetry logging
- async warmup behavior
- applied tuned options on all ollama chat call paths
- kept `think=True` as requested

Purpose:
- reduce cold start latency and improve observability + budget control.

Status:
- local only; later changed further; not committed.

---

### Phase K - Clarification: One-Shot vs Multi-Turn

User concern:
- whether tuning converted app to ongoing chat memory behavior.

Result:
- clarified no conversation-history carryover was introduced.
- flow remained one-shot (system + user prompt each call).

---

### Phase L - Gap Identified: num_predict Was Not Fully Input-Adaptive Yet

User challenged static caps correctly.

Issue:
- early tuning used mode-based max caps; not truly derived from input size each request.

Decision:
- implement genuine adaptive output budgeting with context-aware bounds.

---

### Phase M - Adaptive Budgeting Overhaul (Local)

Implemented locally:
- restored context ceiling to 16k (`ctx_num=16384`) per user request
- added helper/planner stack:
  - `_estimate_tokens_from_text`
  - `_estimate_tokens_from_messages`
  - `_desired_output_tokens`
  - `_plan_ollama_budget`
- enforced:
  - `num_predict <= (num_ctx - prompt_tokens - safety)`
- added optional auto-bump logic toward 16k
- added refine chunking fallback when budget insufficient
- applied adaptive planning to promptify path too
- added quick token estimate display after transcription

Purpose:
- avoid forced summarization and preserve rewrite fidelity.

Status:
- local only; later hardened; not committed.

---

### Phase N - Sub-agent Review on Adaptive Budgeting

Review consensus:
- directionally correct
- main risk: auto-bump may fail on models constrained to smaller context
- estimator/chunking/caching robustness concerns identified

High-value fixes requested:
- context oversize fallback retry
- better multilingual token estimation
- stronger chunk splitter strategy
- improve behavior around cache clearing cadence

---

### Phase O - Hardening Pack 1 (Local)

Implemented locally:
- added guard to prevent bump beyond configured model context unless explicitly enabled
- added retry wrapper for context-window failures with recomputed budget
- replaced coarse token estimate with script-aware multilingual heuristic
- improved chunk splitting hierarchy:
  - paragraphs -> lines -> punctuation (including CJK) -> fallback segmenting
- reduced cache clearing frequency (periodic instead of every chunk)

Purpose:
- reduce edge-case failures and regressions from aggressive budgeting logic.

Status:
- local only.

---

### Phase P - Sub-agent Re-Review + Follow-up Patch List

Reviewer split:
- one reviewer: safe
- another reviewer: no-go without further hardening

Requested precise follow-ups:
- enforce splitter budget in edge cases
- optional tokenizer-backed counting (`tiktoken`) with fallback
- memory-threshold gate for cache clear
- richer fallback logging for original + retry failure context

---

### Phase Q - Hardening Pack 2 (Local)

Applied exactly in one pass:
- optional `tiktoken` token counting fallback path
- added `MLX_CLEAR_CACHE_MEMORY_MB` threshold gate
- strengthened fallback splitter budget enforcement
- improved context-fallback diagnostics/logging

Validation reported:
- syntax/compile checks passed.

Status:
- local only.

---

### Phase R - Shutdown Robustness Planning (Docs + 3 Subagents)

User requested robust graceful shutdown plan.

Findings from review:
- previous close path lacked full worker orchestration
- only current worker canceled, not all active workers
- recording thread join and centralized cleanup were incomplete
- model unload path was not centrally orchestrated

Proposed plan:
- idempotent shutdown coordinator
- hook all exit paths (`closeEvent`, `aboutToQuit`, signals)
- track/cancel/wait all workers
- join recording/background threads
- ordered cleanup of audio + model resources
- bounded timeouts + fail-safe behavior

---

### Phase S - Graceful Shutdown Implementation (Local)

Implemented locally after fresh pass:
- centralized `_graceful_shutdown(...)` coordinator
- idempotent shutdown flags/locks
- `aboutToQuit` + `closeEvent` wiring
- worker registry and cooperative cancel/wait loops
- recording/background thread tracking and join support
- cleanup flow for PyAudio and model-related resources
- best-effort model unload requests (`keep_alive=0`) with timeout-bound behavior
- signal handling path (`SIGINT`/`SIGTERM`) to request app quit
- guards to avoid starting new actions during shutdown

Validation:
- repeated compile checks reported as passed.

Status:
- local only; not committed.

---

### Phase T - Regression Reported by User

Observed symptoms after tuning arc:
- huge memory peaks
- model returning empty output

User goal:
- safely return to state after prompt improvements and before model tuning arc.

---

### Phase U - Rollback Planning and Execution (Performed)

Rollback target selected:
- `6a61314` (stable, committed, pushed; before model tuning/shutdown experiments)

Why this target:
- contains desired prompt/flow improvements
- excludes risky experimental tuning/shutdown changes
- synced with remote

Exact steps executed:

1) backup patch of local experiments:
```bash
git diff -- GUI-whisper-chat-mode_colored_button_Hybrid_v5.py > /tmp/whisper-local-pre-rollback-20260219-020752.patch
```

2) stash experimental changes:
```bash
git stash push -m "WIP model tuning + shutdown experiments backup" -- GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
```

3) verify stable state:
```bash
git status --short
git rev-parse --short HEAD
python3 -m py_compile GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
```

Recorded outcome:
- clean working tree after stash
- HEAD at `6a61314`
- stash preserved
- backup patch saved in `/tmp`

---

## 4) What Is Preserved vs Not Preserved in Current HEAD

### Preserved in current stable history

1. Promptify meta-prompt structure and quality guard improvements (commit `5bd8a65`)
- mandatory PREPARATION section placement
- fixed PREPARATION item-1 sentence enforcement
- context-specific pre-execution prep behavior
- strengthened quality gates (format, preservation, anti-fabrication, etc.)

2. Post-transcription selected mode execution (commit `6a61314`)
- transcription follow-up honors selected button mode (refine/promptify)

### Not present in current HEAD (only in stash/patch)

- 8k context switch and later context rebalance experiments
- adaptive output planner and associated helper stack
- auto-bump logic + context fallback wrappers
- advanced chunking/token estimation hardening
- tokenizer optional integration path
- memory-threshold cache clearing logic
- centralized graceful shutdown orchestration and model unload enhancements

---

## 5) How Another Agent Can Reconstruct/Compare Exactly

Context note:

- `6a61314` is the **rollback/runtime anchor** used for stable-vs-experimental comparisons.
- Current repository tip is newer (`ff704c1`) and includes post-rollback documentation/prompt commits.

### Confirm current state and stash anchor

```bash
cd /Users/luizconrado/PycharmProjects/whisper-local
git status --short
git rev-parse --short HEAD
git log --oneline -n 6
git stash list
git show --no-patch --pretty=fuller stash@{0}
```

### Make the stash reference durable before any stash operations

`stash@{0}` is positional and can move. Pin it to a hash/tag first:

```bash
cd /Users/luizconrado/PycharmProjects/whisper-local
STASH_HASH=$(git rev-parse stash@{0})
echo "$STASH_HASH"  # expected: 344b0fa0073e20049d9f14f995e7d7bafc30e281
git tag -fa handover-phase-i-arc "$STASH_HASH" -m "Pinned stash snapshot for Phase I+ reconstruction"
```

Fallback if `stash@{0}` no longer exists:

```bash
# Use pinned tag or known commit hash directly
git show handover-phase-i-arc:GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
# or
git show 344b0fa0073e20049d9f14f995e7d7bafc30e281:GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
```

### Compare current HEAD vs stash snapshot

```bash
git diff --stat HEAD..stash@{0}
git diff HEAD..stash@{0} -- GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
```

### Compare rollback anchor vs stash snapshot (runtime-arc lens)

```bash
git diff --stat 6a61314..stash@{0}
git diff 6a61314..stash@{0} -- GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
```

### Search important symbols in stash version

```bash
git show stash@{0}:GUI-whisper-chat-mode_colored_button_Hybrid_v5.py | rg -n "_plan_ollama_budget|_estimate_tokens_from_text|_split_text_by_token_budget|_ollama_chat_with_ctx_fallback|MLX_CLEAR_CACHE_MEMORY_MB|tiktoken|_graceful_shutdown|aboutToQuit|keep_alive"
```

### Recover whole experiment set on a safe branch

```bash
git checkout -b codex/recover-model-optimization
# Optional: if you want exact runtime-anchor context before apply:
# git checkout 6a61314
git stash apply stash@{0}
git status --short
```

### Recover selectively (recommended)

```bash
git checkout -p stash@{0} -- GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
```

---

## 6) Suggested Reintroduction Strategy (Safe, Incremental, Testable)

Goal: reintroduce value without re-triggering memory/output regressions.

Recommended order:

1. telemetry-only changes
- no behavior changes, only observability

2. minimal adaptive budgeting without auto-bump/chunking
- conservative bounds only

3. add context fallback retry
- strict retry condition on context error signatures

4. add chunking behind explicit feature flag
- validate punctuation-poor transcripts before default enable

5. optional tokenizer-backed counting path
- keep heuristic fallback for portability

6. shutdown coordinator as isolated feature
- separate test plan from model tuning changes

7. introduce one feature per commit with targeted smoke matrix
- quick bisect path if regressions reappear

Mandatory phase gate:

1. Do not start phase `N+1` until phase `N` has:
- a dedicated commit hash,
- Section `7` validation evidence captured,
- and explicit pass/fail notes for that phase criteria in Section `13.6`.

---

## 7) Validation Matrix to Run After Each Reintroduced Step

Run these after every phase-level increment and capture evidence in your PR/commit notes:

1. Record -> Stop -> Transcribe -> Refine  
Expected: output non-empty; no mode mismatch; no unhandled exception in logs.
2. Record -> Stop -> Transcribe -> Promptify (selected mode)  
Expected: promptify path is used automatically when selected; output non-empty.
3. Long transcript (well punctuated)  
Expected: no context-window crash; fidelity retained.
4. Long transcript (poor punctuation)  
Expected: fallback/chunking (if enabled) is stable and merged output remains readable.
5. Multi-language text fragment  
Expected: no language corruption from estimator/planner changes.
6. Repeated runs for memory trend (`>=20` mixed runs)  
Expected: no runaway growth; no empty outputs.
7. App close while idle  
Expected: exits cleanly with no hanging worker/thread.
8. App close while recording  
Expected: recording thread cancels/joins; app exits.
9. App close while processing  
Expected: workers cancel/join within timeout; app exits.
10. Verify no empty final outputs on normal cases  
Expected: 0 empty responses in normal-path runs.

Evidence capture minimum:

1. Command outputs from Section `14` comparison/retrieval checks.
2. Runtime logs showing success/failure for each item above.
3. `python3 -m py_compile GUI-whisper-chat-mode_colored_button_Hybrid_v5.py` success after each increment.

Definitions and measurable gates:

1. Empty output = final model text where `len(text.strip()) == 0`.
2. Memory trend baseline = capture RSS peak from 5 baseline runs before phase changes.
3. Fail memory gate if:
- median RSS peak after phase increases by more than `35%` versus baseline, or
- any single run exceeds `2.0x` baseline median.

Evidence storage template (recommended):

```bash
PHASE_TAG=phase-i-j
STAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR=/tmp/whisper-phase-validation/$PHASE_TAG/$STAMP
mkdir -p "$OUT_DIR"

# capture git + stash context
git status --short > "$OUT_DIR/git-status.txt"
git rev-parse --short HEAD > "$OUT_DIR/head.txt"
git stash list > "$OUT_DIR/stash-list.txt"

# capture current option-related anchors
rg -n "ollama.chat\\(|keep_alive|num_predict|num_ctx|_plan_ollama_budget|_ollama_chat_with_ctx_fallback" \
  GUI-whisper-chat-mode_colored_button_Hybrid_v5.py > "$OUT_DIR/option-anchors.txt"

# compile check
python3 -m py_compile GUI-whisper-chat-mode_colored_button_Hybrid_v5.py > "$OUT_DIR/py-compile.txt" 2>&1
```

---

## 8) Decision Rationale Ledger (Why Decisions Were Made)

- PREPARATION section enforcement: improve determinism and execution readiness of generated meta-prompts.
- Additional Promptify guardrails: reduce prompt drift, instruction hijacking, and requirement loss.
- Post-transcription mode fix: align UI intent with automatic workflow.
- Aggressive tuning arc: sought speed/memory gains while preserving quality.
- Adaptive budgeting: attempted fidelity-first rewrite for long inputs.
- Hardening cycles: addressed real edge-case risks from sub-agent reviews.
- Shutdown architecture: aimed to prevent thread/resource leaks on close paths.
- Rollback: chosen to restore reliability quickly after observed regression.

---

## 9) Known Artifacts and Pointers

- Current canonical HEAD commit: `ff704c1`
- Handover base reconstruction commit: `1a6e1b7`
- Post-rollback GLM prompt refinement commit: `5947940`
- Handover onboarding/phase-clarity extension commit: `ff704c1`
- Stable runtime anchor commit: `6a61314`
- Promptify enhancement commit: `5bd8a65`
- Pre-session foundational commit: `8f1a6e6`
- Stash of experimental arc: `stash@{0}`
- Stash commit hash (durable): `344b0fa0073e20049d9f14f995e7d7bafc30e281`
- Backup patch artifact: `/tmp/whisper-local-pre-rollback-20260219-020752.patch`
- Main implementation file: `/Users/luizconrado/PycharmProjects/whisper-local/GUI-whisper-chat-mode_colored_button_Hybrid_v5.py`

---

## 10) Final Handover Statement

If a new agent starts from the current repo and reads only this document plus git history, they can:

- identify exactly what is currently stable and why,
- recover all tuning/shutdown experiments safely,
- compare stable vs experimental behavior at file and symbol level,
- and reintroduce changes in controlled increments with clear rollback points.

This intentionally preserves your key requirement: proceed calmly, step-by-step, with optimization only where it is proven safe.

---

## 11) Three-Version Script Comparison (Revalidated After New Commits)

Comparison target file:  
`/Users/luizconrado/PycharmProjects/whisper-local/GUI-whisper-chat-mode_colored_button_Hybrid_v5.py`

Three reference snapshots used:

1. **Before all session implementations**: `6c816e6`  
   Commit message: `Add Hybrid v5 transcriber with Ollama 0.6 API compatibility`  
   Line count snapshot: `2453`

2. **Current active version**: `HEAD` = `ff704c1` (includes follow-up commits `5947940`, `1a6e1b7`, `ff704c1`)  
   Line count now: `2799`

3. **After all implementations (pre-rollback, regression period)**: `stash@{0}` (`344b0fa`)  
   Stash message: `WIP model tuning + shutdown experiments backup`  
   Line count snapshot: `3722`

Diff size indicators:

- `6c816e6 -> HEAD`: `389 insertions, 43 deletions`
- `HEAD -> stash@{0}`: `999 insertions, 76 deletions`
- `6c816e6 -> stash@{0}`: `1376 insertions, 107 deletions`

---

## 12) Session Task Matrix (Current Status vs 3 Snapshots)

Status keys used:

- `Implemented`: active in current `HEAD`
- `Partial`: historically done in session, but not active now as code behavior/value
- `Not Implemented (Current)`: absent in current `HEAD`; available only in stash/patch/history

Important provenance note:

1. Phases `I/J`, `M`, `O`, `Q`, and `R/S` do **not** have per-phase committed SHAs in current branch history.
2. Their reconstruction source is `stash@{0}` (`344b0fa0073e20049d9f14f995e7d7bafc30e281`) plus patch artifacts.
3. Reintroduction must follow strict order (`I/J` -> `M` -> `O/Q` -> `R/S`) with one commit per phase family.

| Phase / Task | Before (`6c816e6`) | Current (`HEAD`) | After-all (`stash@{0}`) | Current Status | Evidence |
|---|---|---|---|---|---|
| **Phase A** Promptify PREPARATION section + strict placement/order | Missing | Present | Present | Implemented | Handover `Phase A`; code has `PREPARATION` contract and exact item-1 rule. |
| **Phase B** Promptify hardening (injection resistance, requirement/language preservation, stricter quality gates) | Missing | Present | Present | Implemented | Current Promptify rules include untrusted-source rule and anti-fabrication quality gates. |
| **Phase C** Prompt improvements checkpoint commit `5bd8a65` | N/A | In history | In history ancestry | Implemented | Commit exists in branch history. |
| **Phases D/E/F** Auto post-transcription follows selected mode (`refine`/`promptify`) | Missing | Present | Present | Implemented | `post_process_mode` in `TranscriptionThread` and passed by UI worker creation. |
| **8f1a6e6 foundation** UI action picker + Promptify flow + sensitivity callback deadlock fix | Missing | Present | Present | Implemented | `QToolButton` action picker and lock-safe callback notify path exist in current. |
| **Phase G** GLM refine prompt iterative rewrite (quality gate version) | Missing | Present | Present | Implemented | Current GLM system prompt includes `Restructuring quality gate (internal decision)`. |
| **Phase H** temporary `glm ctx_num=8192` experiment | Missing | Not active (`ctx_num=16384`) | Historically present during arc | Partial | Session did this temporarily; current code no longer keeps `glm` at 8k. |
| **Phases I/J** runtime tuning pack (`keep_alive`, adaptive hooks, `num_predict` planning, telemetry/warmup expansion) | Missing | Missing | Present | Not Implemented (Current) | Stash-only symbols: `OLLAMA_KEEP_ALIVE`, `_plan_ollama_budget`, metrics logger, warmup calls. |
| **Phase M** adaptive budgeting helper stack (`_estimate_tokens*`, `_plan_ollama_budget`, chunk fallback) | Missing | Missing | Present | Not Implemented (Current) | Stash-only function set; absent in current file. |
| **Phase O** hardening pack 1 (context fallback retry, multilingual estimation, splitter improvements) | Missing | Missing | Present | Not Implemented (Current) | Stash-only helpers (`_ollama_chat_with_ctx_fallback`, multilingual token estimate). |
| **Phase Q** hardening pack 2 (`tiktoken`, memory-threshold cache clear, richer diagnostics) | Missing | Missing | Present | Not Implemented (Current) | Stash-only imports/config: `tiktoken`, `MLX_CLEAR_CACHE_MEMORY_MB`. |
| **Phases R/S** graceful shutdown coordinator (`aboutToQuit`, global cancel/wait/join, model unload flow) | Missing | Missing | Present | Not Implemented (Current) | Stash-only `_graceful_shutdown` and `aboutToQuit` wiring. |
| **Phase U** rollback and stabilization path | N/A | Reflected by preserved baseline and stash artifact flow | N/A | Implemented (historical action) | Handover rollback steps + stash/patch artifact preservation. |

### Boundary Clarification: Phase H vs Phases I/J

1. **Phase H scope** was limited to token-budget discussion plus temporary context experiments (including a temporary `glm` 8k setting during the arc).
2. **Phase I begins** when runtime-tuning machinery appears (for example symbols such as `OLLAMA_KEEP_ALIVE`, `_plan_ollama_budget`, `_estimate_tokens_from_text`, `_ollama_chat_with_ctx_fallback`, telemetry fields, and dynamic `num_predict` planning).
3. In current HEAD, `phi4` remains `ctx_num=8192` while `glm` is `ctx_num=16384`; therefore Phase H is marked `Partial` (historically applied, not currently active as temporary 8k `glm` behavior).

### Current "Already Implemented" Set (effective now)

1. Phase A
2. Phase B
3. Phase C (history checkpoint)
4. Phases D/E/F
5. `8f1a6e6` foundation changes
6. Phase G
7. Phase U (historical rollback workflow and artifacting)

### Current "Open / Not Yet Active" Set

1. Phase H temporary 8k `glm` setting (not active now)
2. Phases I/J
3. Phase M
4. Phase O
5. Phase Q
6. Phases R/S

---

## 13) Open-Task Reconstruction Pack (What After-All Had That Current Does Not)

This section is intentionally detailed so future agents can reintroduce features from Phase I onward safely and in isolation.

### 13.0 Open-Phase Symbol-to-Anchor Map (Current file insertion points)

Use this as a fast navigation index before reintroducing code from stash:

| Open Phase | Stash-only symbols/features | Current file anchor(s) to patch |
|---|---|---|
| I/J runtime tuning | `OLLAMA_KEEP_ALIVE`, warmup, telemetry fields, dynamic chat options | `TranscriptionThread.refine_text` (`...Hybrid_v5.py:1819`), `TranscriptionThread.promptify_text` (`...Hybrid_v5.py:1873`), `RefinementThread.refine_text` (`...Hybrid_v5.py:1964`), `PromptifyThread.promptify_text` (`...Hybrid_v5.py:2060`) |
| M adaptive budgeting | `_estimate_tokens_from_text`, `_estimate_tokens_from_messages`, `_desired_output_tokens`, `_plan_ollama_budget`, `_split_text_by_token_budget` | helper-function region before worker classes; then same Ollama call anchors above |
| O/Q fallback + hardening | `_ollama_chat_with_ctx_fallback`, optional `tiktoken`, `MLX_CLEAR_CACHE_MEMORY_MB`, richer diagnostics | helper-function region + all `ollama.chat` call sites (`...Hybrid_v5.py:1848`, `...Hybrid_v5.py:1896`, `...Hybrid_v5.py:1992`, `...Hybrid_v5.py:2082`) |
| R/S graceful shutdown | `aboutToQuit` wiring, `_graceful_shutdown`, global cancel/wait/join, unload flow | `AudioTranscriberApp` init (`...Hybrid_v5.py:2103`) and `AudioTranscriberApp.closeEvent` (`...Hybrid_v5.py:2768`) with `AppState` worker registry (`...Hybrid_v5.py:1306`) |

### 13.1 Runtime Tuning Baseline (Phase I/J) — Stash-only Features

What was added in the after-all snapshot:

1. Environment-driven Ollama tuning constants (`keep_alive`, context safety, prediction caps, warmup toggles).
2. Model usage tracking + running-model discovery helpers for unload decisions.
3. Expanded warmup behavior for Ollama models and telemetry logging.

Representative stash snippet:

```python
# Ollama runtime tuning (configurable via environment).
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "20m")
OLLAMA_MIN_CTX = _env_int("OLLAMA_MIN_CTX", 2048)
OLLAMA_CTX_SAFETY_TOKENS = _env_int("OLLAMA_CTX_SAFETY_TOKENS", 768)
OLLAMA_NUM_PREDICT_MIN = _env_int("OLLAMA_NUM_PREDICT_MIN", 256)
OLLAMA_NUM_PREDICT_MAX_REFINE = _env_int("OLLAMA_NUM_PREDICT_MAX_REFINE", 8192)
OLLAMA_NUM_PREDICT_MAX_PROMPTIFY = _env_int("OLLAMA_NUM_PREDICT_MAX_PROMPTIFY", 8192)
...
OLLAMA_WARMUP_ENABLED = _env_bool("OLLAMA_WARMUP_ENABLED", True)
```

```python
class OllamaWarmup:
    @classmethod
    def warm_model(cls, model_name: str) -> None:
        ...
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': 'warmup'}],
            stream=False,
            keep_alive=OLLAMA_KEEP_ALIVE,
            options={'num_ctx': OLLAMA_MIN_CTX, 'temperature': 0.0, 'seed': 1, 'num_predict': 8}
        )
```

Why it was added:

1. Reduce cold-start latency.
2. Bound context/output behavior per call.
3. Improve observability for tuning decisions.

Why it is currently absent:

1. Entire runtime tuning arc was intentionally not retained in stable HEAD after regression wave.

#### 13.1.a Environment Variable Inventory (from `stash@{0}`)

| Variable | Default in stash snapshot | Purpose |
|---|---:|---|
| `OLLAMA_KEEP_ALIVE` | `20m` | Keep model loaded between calls to reduce cold starts. |
| `OLLAMA_MIN_CTX` | `2048` | Lower bound for context window planning. |
| `OLLAMA_CTX_SAFETY_TOKENS` | `768` | Reserved safety buffer in context budgeting. |
| `OLLAMA_NUM_PREDICT_MIN` | `256` | Lower bound for output token budget. |
| `OLLAMA_NUM_PREDICT_MAX_REFINE` | `8192` | Upper cap for refine-mode output. |
| `OLLAMA_NUM_PREDICT_MAX_PROMPTIFY` | `8192` | Upper cap for promptify-mode output. |
| `OLLAMA_OUTPUT_RATIO_REFINE` | `1.10` | Desired output/input ratio for refine mode. |
| `OLLAMA_OUTPUT_RATIO_PROMPTIFY` | `1.00` | Desired output/input ratio for promptify mode. |
| `OLLAMA_REFINE_MIN_FIDELITY_RATIO` | `0.85` | Minimum rewrite fidelity threshold before chunking. |
| `OLLAMA_MIN_CHUNK_INPUT_TOKENS` | `256` | Smallest chunk size for token-based splitting. |
| `OLLAMA_AUTO_BUMP_ENABLED` | `True` | Enables context auto-bump logic. |
| `OLLAMA_AUTO_BUMP_CTX` | `16384` | Auto-bump target context. |
| `OLLAMA_AUTO_BUMP_TRIGGER_TOKENS` | `5500` | Trigger threshold for bump attempt. |
| `OLLAMA_AUTO_BUMP_BEYOND_CONFIG` | `False` | Allows bump beyond model config ceiling when enabled. |
| `OLLAMA_WARMUP_ENABLED` | `True` | Enables Ollama warmup call path. |
| `MLX_CLEAR_CACHE_EVERY_N_CHUNKS` | `2` | Periodic cache clear cadence in chunk loop. |
| `MLX_CLEAR_CACHE_MEMORY_MB` | `0` | Memory-threshold gate for cache clear (`0` disables thresholding). |

#### 13.1.b Recommended conservative rollout profile for first reintroduction pass

Use this profile initially to reduce regression risk while validating Phase I/J plumbing:

```bash
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_WARMUP_ENABLED=true
export OLLAMA_AUTO_BUMP_ENABLED=false
export OLLAMA_AUTO_BUMP_BEYOND_CONFIG=false
export OLLAMA_NUM_PREDICT_MIN=256
export OLLAMA_NUM_PREDICT_MAX_REFINE=4096
export OLLAMA_NUM_PREDICT_MAX_PROMPTIFY=4096
```

After stable validation, tune upward intentionally and re-run Section `7`.

### 13.2 Adaptive Budgeting Core (Phase M) — Stash-only Functions

What existed in after-all snapshot:

1. Multilingual token estimation path.
2. Message token envelope estimation.
3. Desired output token planner per mode.
4. Context/output planner returning `num_ctx`, `num_predict`, and chunk requirement hints.
5. Token-budget text chunking pipeline.

Representative stash snippet:

```python
def _estimate_tokens_from_text(text: str) -> int:
    if _TT_ENCODING is not None:
        return max(1, len(_TT_ENCODING.encode(normalized)))
    ...
```

```python
def _plan_ollama_budget(max_ctx: int, messages: List[Dict[str, Any]], mode: str, source_tokens: int) -> Dict[str, int]:
    prompt_tokens = _estimate_tokens_from_messages(messages)
    desired_predict = _desired_output_tokens(mode, source_tokens)
    ...
    return {"num_ctx": ..., "num_predict": ..., "should_chunk": ...}
```

```python
def _split_text_by_token_budget(text: str, target_tokens: int) -> List[str]:
    # paragraphs -> lines -> sentence punctuation (western + CJK) -> fallback split
```

Why it was added:

1. Prevent output truncation/forced summarization on long transcripts.
2. Keep rewrite fidelity while fitting context budget.

Current gap:

1. None of these helpers are active in current HEAD.

### 13.3 Hardening Pack 1 + 2 (Phases O/Q) — Stash-only Reinforcements

What existed in after-all snapshot:

1. Context-window retry wrapper (`_ollama_chat_with_ctx_fallback`) that retries with configured context when auto-bump is rejected.
2. Detailed response telemetry (`total/load/prompt_eval/eval` timing and counts).
3. Optional `tiktoken` path for tighter token counting.
4. Memory-threshold-gated MLX cache clearing (`MLX_CLEAR_CACHE_MEMORY_MB`).
5. Better diagnostics around fallback failures.

Representative stash snippet:

```python
try:
    import tiktoken
    _TT_ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
```

```python
def _ollama_chat_with_ctx_fallback(...):
    try:
        response = ollama.chat(**active_kwargs)
    except Exception as exc:
        if attempted_ctx <= fallback_ctx or not _is_ollama_context_error(exc):
            raise
        ...
```

```python
MLX_CLEAR_CACHE_MEMORY_MB = _env_int("MLX_CLEAR_CACHE_MEMORY_MB", 0)
...
if MLX_CLEAR_CACHE_MEMORY_MB and PSUTIL_AVAILABLE:
    should_clear_cache &= rss_mb >= MLX_CLEAR_CACHE_MEMORY_MB
```

Current gap:

1. These hardening layers are not in current HEAD.

### 13.4 Integration Points (Where Tuning Entered Runtime Paths)

In the after-all snapshot, refine/promptify calls were wrapped by planning + fallback, with `keep_alive` and dynamic `num_predict`.

Representative stash snippet (transcription-thread refine path):

```python
source_tokens = _estimate_tokens_from_text(text)
plan = _plan_ollama_budget(config.ctx_num, messages, mode="refine", source_tokens=source_tokens)
chat_kwargs = {
    'model': self.model_name,
    'messages': chunk_messages,
    'stream': False,
    'keep_alive': OLLAMA_KEEP_ALIVE,
    'options': {'num_ctx': chunk_plan['num_ctx'], 'temperature': config.temperature, 'seed': config.seed, 'num_predict': chunk_plan['num_predict']}
}
response, used_num_ctx, used_num_predict = _ollama_chat_with_ctx_fallback(...)
```

Representative stash snippet (promptify path):

```python
plan = _plan_ollama_budget(model_config.ctx_num, messages, mode="promptify", source_tokens=source_tokens)
chat_kwargs = {..., 'keep_alive': OLLAMA_KEEP_ALIVE, 'options': {..., 'num_predict': plan['num_predict']}}
```

Current gap:

1. Current HEAD still uses direct chat calls with fixed option set, without this planning/fallback layer.

Current-to-target mini integration sketch (for each `ollama.chat` call site):

```python
# Current HEAD pattern (simplified)
chat_kwargs = {
    "model": model_name,
    "messages": messages,
    "stream": False,
    "options": {"num_ctx": cfg.ctx_num, "temperature": cfg.temperature, "seed": cfg.seed},
}
response = ollama.chat(**chat_kwargs)

# Phase I/M/O target pattern (simplified)
source_tokens = _estimate_tokens_from_text(source_text)
plan = _plan_ollama_budget(cfg.ctx_num, messages, mode=mode, source_tokens=source_tokens)
chat_kwargs = {
    "model": model_name,
    "messages": messages,
    "stream": False,
    "keep_alive": OLLAMA_KEEP_ALIVE,
    "options": {
        "num_ctx": plan["num_ctx"],
        "temperature": cfg.temperature,
        "seed": cfg.seed,
        "num_predict": plan["num_predict"],
    },
}
response, used_num_ctx, used_num_predict = _ollama_chat_with_ctx_fallback(...)
```

### 13.5 Graceful Shutdown Orchestration (Phases R/S) — Stash-only

What existed in after-all snapshot:

1. App-level shutdown flags and locks (`_shutdown_started`, `_shutdown_lock`).
2. Worker registry and cancel/wait orchestrators.
3. Background thread tracking and joins.
4. `aboutToQuit` hook and unified `_graceful_shutdown(...)`.
5. PyAudio termination retries + best-effort model unload flow.

Representative stash snippet:

```python
app = QApplication.instance()
if app:
    app.aboutToQuit.connect(self._on_about_to_quit)
...
def _graceful_shutdown(self, reason: str, timeout_ms: int = 12000) -> bool:
    self._cancel_all_workers()
    self._wait_for_recording_thread(...)
    self._wait_for_workers(...)
    self._terminate_pyaudio_singleton()
    self._release_model_resources()
```

Current gap:

1. Current HEAD retains simpler close handling and does not include this centralized coordinator.
2. Current HEAD has no `aboutToQuit`-coordinated shutdown path and no signal-driven shutdown coordinator from the stash-only arc.

### 13.6 Proposed Phase-Specific Acceptance Criteria (to reduce reimplementation ambiguity)

These are operational pass/fail gates to use when reintroducing missing phases.

#### Phase I/J (runtime tuning + warmup + telemetry)

Pass if all are true:

1. No empty final outputs in `>=20` mixed-length runs (short, medium, long).
2. Telemetry fields are consistently logged per call (`load_duration`, `prompt_eval_duration`, `eval_duration`, counts).
3. Warmup path is non-fatal: failures are logged and do not block normal transcription/refinement.
4. No behavior regression in selected post-process mode (`refine` vs `promptify`).

#### Phase M (adaptive budgeting)

Pass if all are true:

1. For long inputs, planner computes bounded `num_ctx`/`num_predict` without raising context-window errors.
2. Refine outputs preserve content fidelity vs source (no systematic truncation/summarization).
3. Chunking activates only when budget requires it and returns merged readable output.

#### Phase O/Q (fallback/hardening)

Pass if all are true:

1. Context-retry wrapper triggers only on context-like failures and succeeds/fails deterministically.
2. `tiktoken` path is optional; absence falls back cleanly to heuristic estimation.
3. Cache-clearing threshold logic behaves as configured and does not cause instability.
4. Diagnostics include enough context to debug both primary and retry failures.

#### Phases R/S (graceful shutdown)

Pass if all are true:

1. App exits cleanly while idle, recording, and processing (no deadlocks/hangs).
2. Workers are canceled and joined within bounded timeout.
3. Recording/background threads do not remain alive after shutdown.
4. PyAudio/model resource cleanup occurs without fatal exceptions.
5. Shutdown flow is idempotent (`aboutToQuit` + `closeEvent` do not double-fail).

### 13.7 Logging Proof Points (what reviewers should be able to grep)

Use these as concrete audit targets after each phase increment:

1. Phase I/J: log entries containing `load_duration`, `prompt_eval_duration`, `eval_duration`, and the effective `num_ctx`/`num_predict`.
2. Phase M: log planner decisions (`source_tokens`, planned `num_ctx`, planned `num_predict`, `should_chunk`).
3. Phase O/Q: log fallback attempts (`context retry`, initial failure reason, retry context), and whether `tiktoken` path or heuristic path was used.
4. Phase O/Q: when cache controls are active, log `MLX_CLEAR_CACHE_MEMORY_MB` decision inputs and whether cache clear executed.
5. Phase R/S: log shutdown lifecycle markers (`shutdown_started`, worker cancel summary, join outcomes, `shutdown_completed`).

---

## 14) Retrieval Commands for Future Reimplementation (Phase I -> End)

Use these to inspect the exact after-all snapshot that contained the tuning/shutdown arc:

```bash
cd /Users/luizconrado/PycharmProjects/whisper-local

# identify stash snapshot used in this handover
git stash list
git show --no-patch --pretty=fuller stash@{0}
git rev-parse stash@{0}

# optional: pin to stable tag before any stash operations
git tag -fa handover-phase-i-arc "$(git rev-parse stash@{0})" -m "Pinned stash snapshot for Phase I+ reconstruction"

# inspect full file at after-all state
git show stash@{0}:GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
# or durable reference
git show handover-phase-i-arc:GUI-whisper-chat-mode_colored_button_Hybrid_v5.py

# inspect only open-task symbols (I onward)
git show stash@{0}:GUI-whisper-chat-mode_colored_button_Hybrid_v5.py | \
  rg -n "OLLAMA_KEEP_ALIVE|_estimate_tokens_from_text|_plan_ollama_budget|_split_text_by_token_budget|_ollama_chat_with_ctx_fallback|tiktoken|MLX_CLEAR_CACHE_MEMORY_MB|_graceful_shutdown|aboutToQuit|keep_alive|num_predict|load_duration|prompt_eval_duration|eval_duration"

# compare current HEAD vs after-all snapshot
git diff HEAD..stash@{0} -- GUI-whisper-chat-mode_colored_button_Hybrid_v5.py

# snapshot current-head call patterns before editing (baseline proof)
rg -n "ollama.chat\\(|num_ctx|temperature|seed|num_predict|keep_alive|aboutToQuit|closeEvent" \
  GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
```

Selective recovery approach (recommended):

```bash
# create a safe branch before reintroduction
git checkout -b codex/reintroduce-phase-i-plus

# selectively recover hunks from stash
git checkout -p stash@{0} -- GUI-whisper-chat-mode_colored_button_Hybrid_v5.py
```

Pre-Phase-I baseline smoke checklist:

1. Run one short recording with selected action `Refine Text` and confirm non-empty output.
2. Run one short recording with selected action `Promptify Text` and confirm non-empty output.
3. Save logs/output notes under `/tmp/whisper-phase-validation/baseline/<timestamp>/`.
4. Only then start Phase I/J edits.

This section is intended to let future agents reproduce the missing arc with traceability and controlled incremental reintroduction.
