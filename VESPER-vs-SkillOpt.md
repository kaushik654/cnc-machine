# VESPER vs Microsoft SkillOpt — Comparison

**Prepared for architectural / novelty review · Knox AI, SRI-Bangalore**

Sources reviewed: `github.com/microsoft/SkillOpt` (repo + `docs/index.md`, `docs/reference/config.md`, `docs/sleep/README.md`), the project page `microsoft.github.io/SkillOpt`, and the paper reference *SkillOpt: Executive Strategy for Self-Evolving Agent Skills*, arXiv 2605.23904 (2026). Release timeline: v0.1.0 — 2 Jun 2026, v0.2.0 (SkillOpt-Sleep) — 2 Jul 2026.

Accompanying diagrams:
- `vesper_vs_skillopt.png` — side-by-side loop comparison
- `vesper_vs_skillopt_overlap.png` — pillar-by-pillar prior-art overlap map
- `skill_update_comparison.png` — **dynamic skill updating compared dimension by dimension (the headline diagram)**

---

## 1. One-paragraph summary of each

**SkillOpt** treats a single natural-language **skill document (Markdown) as the trainable state of a frozen LLM agent**. It never touches model weights. An optimizer LLM reads scored rollouts from a target LLM and proposes bounded add/delete/replace edits to that document; a candidate edit is accepted only if it strictly improves a **held-out validation score**. The deployed artifact is one `best_skill.md` (300–2,000 tokens) that is loaded into the frozen agent's context with zero inference-time overhead. The entire discipline is borrowed from deep learning: epochs, batch size, a textual "learning rate" (max edits per step), an LR schedule, gradient-clipping-like edit selection, a rejected-edit buffer, an epoch-boundary slow update, and cross-epoch optimizer memory ("meta skill").

**VESPER** is an on-device device-control agent with **two coupled loops**. The execution loop (V1–V4) grounds the live screen, retrieves skills that are *applicable to the current screen*, previews the plan for human approval, then executes and **visually verifies every single step** against a stored expected-result image — recovering mid-task when reality deviates. The creation loop (M1–M4) fires when retrieval finds nothing: it measures the capability gap as an embedding delta, hypothesizes and composes a new skill from certified primitives using formal operators, validates it in a simulated sandbox with progressive difficulty, grades and publishes it into a versioned skill-dependency DAG, and keeps improving it from verified execution outcomes.

**The one-line difference:** SkillOpt optimizes *how well a frozen model follows one document*, scored by a benchmark number, offline, in the cloud. VESPER decides *what to do on a physical device*, scored by the screen, per step, at runtime, on-device — and creates the skill when none exists.

---

## 2. ⭐ THE HEADLINE DIFFERENCE — DYNAMIC SKILL UPDATING

> **This is the section to lead with.** Both systems update skills automatically, so "we update skills dynamically" is *not* our novelty — SkillOpt does that, published and benchmarked. What differs is **what triggers the update, what evidence drives it, what unit changes, how it is validated, and how fast it lands.** Diagram: `skill_update_comparison.png`.

### 2.1 The shape of the update loop

**SkillOpt — scheduled batch update.** A training step (or a nightly Sleep cycle) runs a batch of tasks, an optimizer LLM reflects on the scored trajectories, produces at most *N* bounded text patches (default 4), and a held-out gate accepts the candidate only on strict improvement. Between two scheduled steps, **nothing can change** — a live failure has no way to start an update.

**VESPER — event-driven, two-speed update, plus a creation path.**
- **Fast path (inside the task, seconds):** a step's screen fails visual verification → the failing screen becomes the retrieval query → a corrected step is spliced into the running plan.
- **Slow path (background):** verified traces accumulate in episodic memory → M4 updates parameters, graph topology, and learning strategy → the change lands as a new **version of a node**.
- **Creation path:** retrieval finds nothing applicable → the gap is measured as an embedding delta → M1 hypothesises, M2 composes, M3 certifies → a **brand-new node** enters the graph.

### 2.2 Fourteen concrete differences in how a skill gets updated

| # | Aspect of the update | SkillOpt | VESPER |
|---|---|---|---|
| **1** | **Trigger** | A schedule — training step, epoch boundary, or nightly cycle | An **event** — visual deviation, retrieval miss, user correction, or measured degradation |
| **2** | **Evidence driving the change** | An aggregate benchmark score over a batch: "the document got better/worse" | **Localized physical evidence**: which step, the actual screen, the expected screen, the deviation |
| **3** | **Attribution of failure** | Indirect — the score cannot tell you which instruction was wrong | Direct — the failing step index and its screen pair are the record |
| **4** | **Unit that changes** | The one shared Markdown document | One step inside one skill node, a new node, or the graph topology |
| **5** | **Blast radius** | Global: every task shares that file, so edits interfere — this is *why* they need a bounded edit budget and a rejected-edit buffer | Node-scoped: fixing one skill leaves others untouched, unless a shared primitive is deliberately updated (then all dependants benefit at once) |
| **6** | **Can the update create a NEW capability?** | **No.** It rewords what the seed document already covers; the capability set is fixed at authoring | **Yes.** Gap → hypothesis → composition from certified primitives → new certified node |
| **7** | **Validation of the update** | One gate: held-out split, strict improvement — statistical, offline | **Three gates**: sandbox certification (M3), human preview (V3), per-step visual verification at runtime (V4) — simulated, human, and physical |
| **8** | **Latency to take effect** | One training step or one night — and only if the gate passes; the user lives with the failure until then | **Seconds** for in-task repair; background consolidation for the durable change |
| **9** | **Rollback semantics** | Reject the candidate, keep the previous `best_skill.md` — one file, binary revert | **Versioned DAG nodes**: roll one skill back to v(n−1) while everything else keeps running; hot-swap; per-node history |
| **10** | **Where experience is stored** | Rejected-edit buffer + optimizer-side meta-skill memory; explicitly *not* shipped with the artifact — it only shapes future edits | On-device tripartite memory; episodic traces are **retrievable at runtime** and re-rank retrieval — memory changes *selection*, not just future edits |
| **11** | **Who performs the update** | A separate, usually stronger optimizer LLM, in the cloud, on an API budget | On-device modules (M1 genesis, M2 operators, M4 controller) — no stronger teacher model required |
| **12** | **Cost model of updating** | API calls × batch size × epochs; the nightly cycle runs on the user's own API budget | One embedding + one similarity per verified step; sandbox runs locally |
| **13** | **Behaviour when the update fails** | Edit rejected and buffered as negative feedback; the skill is unchanged, so the user's failure persists until a future cycle | Recovery budget exhausts → safe fallback **within the session**; the gap is logged for creation; the failure is contained |
| **14** | **UI / version drift** | No notion of it — a text document cannot know the UI moved; drift only shows up later as a lower score | Drift surfaces **instantly** as a visual deviation, triggers re-grounding, and schedules a refresh of that skill's visual states |

### 2.3 The four points to say out loud in review

1. **Trigger:** *"Theirs is time-triggered. Ours is evidence-triggered — the screen decides when a skill needs to change."*
2. **Granularity:** *"They patch one document. We version a node inside a dependency graph, so a fix is scoped and reversible per capability."*
3. **Creation:** *"Their update loop can only improve what exists. Ours can produce a capability the agent never shipped with."*
4. **Latency:** *"Their improvement arrives next epoch or next night. Ours repairs the task the user is running right now, and consolidates afterwards."*

### 2.4 Honest counterweight (do not skip this in review)

SkillOpt's update machinery is, in isolation, **more disciplined than ours**: a textual learning rate with a schedule, an epoch-boundary slow update, cross-epoch optimizer memory, and a rejected-edit buffer — each ablated and shown to matter. Our M4 currently names three layers without specifying the stabilizers. Two things worth adopting:

- A **rejected-attempt buffer per skill** ("this repair was tried on this screen and failed") so recovery does not re-explore dead ends.
- An explicit **bounded-change budget** on M4, analogous to their textual learning rate, to keep structural adaptation from thrashing the graph.

Adopting both strengthens the design and costs nothing in claim scope — they are implementation stabilizers, not claimed novelty.

---

## 3. Simple comparison table (the one for the slide)

| | **SkillOpt (Microsoft)** | **VESPER (ours)** |
|---|---|---|
| **What a "skill" is** | One Markdown document (prompt-like text) | Multi-step procedure; each step = {instruction, target image, expected-result image} |
| **What gets optimized** | The text of that one document | The *set* of skills — new ones created, existing ones improved |
| **Ground truth / reward** | Benchmark accuracy on a held-out split | The **live device screen** vs the step's expected visual state |
| **When learning happens** | Offline training epochs, or a nightly "sleep" cycle | Runtime (recovery + gap trigger) **and** background meta-learning |
| **Can it create a missing capability?** | No — it only edits an existing document | Yes — gap detection → hypothesis → composition → certification |
| **Structure of skills** | Flat, monolithic file | Versioned DAG of dependencies, composed from atomic primitives |
| **Runtime verification** | None (verification = the score at task end) | Per-step visual verification, cos(live, expected) ≥ τ |
| **Runtime recovery** | None — offline optimization only | Rollback + alternative path + re-grounded re-retrieval, bounded budget |
| **Screen / UI awareness** | None | Core: grounding, affordances, OCR validation, screenshot comparison |
| **Modalities** | Text in / text out (plus generated code for spreadsheet tasks) | UI actions · MCP tools · A2A agents, from one abstraction |
| **Human oversight** | Sleep stages a nightly proposal for adoption | Per-execution plan preview with risk flags and confidence, before acting |
| **Where it runs** | Cloud LLM APIs (GPT / Claude / Qwen / MiniMax) | 100% on-device |
| **Inference-time cost** | Zero extra model calls | One embedding + one similarity per step |
| **Maturity** | Published, open-source, benchmarked (52/52 best-or-tied) | Invention disclosure, no benchmarks yet |

---

## 4. Deep technical comparison

| Dimension | SkillOpt | VESPER |
|---|---|---|
| Optimization target | `skill.md` text | Skill graph contents + individual skill parameters/structure |
| Analogy | Deep-learning optimizer in text space | Closed-loop control + capability synthesis |
| "Forward pass" | Rollout: frozen target LLM runs benchmark tasks | Execution: agent runs steps on the real device |
| "Gradient" | Optimizer LLM reflects on success/failure minibatches → edit patches | Visual deviation signal (cos < τ) + outcome traces |
| "Clipping" | `learning_rate` = max edit patches per step (default 4), `lr_scheduler` cosine/linear/constant | Bounded recovery budget; performance guards on meta-learning |
| Acceptance rule | Strict improvement on a held-out selection split (`use_gate: true`) | M3 certification grade from sandbox curriculum; V3 human approval; V4 per-step τ |
| Negative feedback | Rejected-edit buffer feeds the optimizer | Failed traces → episodic memory → M4; corrections logged from V3 |
| Long-horizon memory | Epoch-boundary slow update + optimizer-side "meta skill" | Tripartite memory (semantic / episodic / procedural) + versioned skill graph |
| Regression safety | Gate rejects the edit; `best_skill.md` unchanged | Versioned DAG node rollback on measured degradation; hot-swap |
| Retrieval at runtime | None — one document is loaded into context. (`recall_k` retrieves similar *past tasks* into the nightly consolidation, not skills at runtime) | Hybrid dense+lexical retrieval over step-granular index, re-ranked by execution history, filtered by live-screen applicability |
| Composition | Text edits (add / delete / replace) | Formal operators — sequential, parallel, conditional, iterative — with constraint-satisfaction validity |
| Evaluation environment | Benchmark splits (SearchQA, ALFWorld, DocVQA, OfficeQA, LiveMathematicianBench, SpreadsheetBench) | Simulated device sandbox (M3) + the real device (V4) |
| Failure mode when nothing works | Edit rejected; skill unchanged | Gap signature → M1 creates a new skill; or safe fallback |
| Data leaving the device | Yes — cloud API calls, including the nightly cycle | None |
| Reported gains | +23.5 pts avg (GPT-5.5, direct chat); +3.1 to +4.5 pts per nightly Sleep cycle on SearchQA | Not yet measured — disclosure stage |

---

## 5. Where SkillOpt is genuinely better (state this honestly to reviewers)

1. **Empirical proof.** SkillOpt is best or tied-best on all 52 evaluated (model × benchmark × harness) cells, with a published paper, reproducible configs, and an open-source implementation. VESPER currently has worked traces, not measurements. This is the single biggest gap and it is not a design gap — it is an evidence gap.
2. **Optimizer discipline is far more developed.** Textual learning rate with a schedule, epoch-boundary slow update, rejected-edit buffer, and optimizer-side meta-skill memory are ablated and shown to matter (e.g., removing the rejected-edit buffer costs several points on SpreadsheetBench). Our M4 "three layers" is, by comparison, a conceptual frame with no tuning machinery specified.
3. **The validation gate is simpler and stronger than a sandbox.** "Accept only if a held-out score strictly improves" is cheap, unambiguous, and hard to game. Our M3 simulated-device sandbox is more ambitious and much harder to build faithfully — SkillOpt gets a comparable safety guarantee with far less engineering.
4. **Zero deployment overhead and clean transfer.** The exported artifact costs nothing at inference and transfers across model scales and across Codex↔Claude Code harnesses. VESPER adds per-step embedding and comparison cost on a battery-powered device.
5. **Engineering maturity.** PyPI package, WebUI dashboard, plugins for Claude Code / Codex / Copilot, deterministic no-API-key repro path.
6. **Honest scope reporting.** They state plainly that gains are flat within noise on saturated benchmarks. We should match that standard in our own claims.

**What we should borrow:** the rejected-edit buffer idea (log *what not to try again* per skill), the bounded-edit budget as an explicit knob on M4, and a strict-improvement held-out gate as a cheap complement to the M3 sandbox.

---

## 6. Where VESPER is fundamentally different (not merely better-engineered)

1. **The oracle is physical, not statistical.** SkillOpt's signal is an aggregate score over a task split. VESPER's signal is a per-step comparison of the actual screen against a stored expected screen. A score tells you a skill is *generally* better; a screen tells you *this step, right now, on this device* did or did not work.
2. **It creates capabilities, not just better wording.** SkillOpt cannot answer a request no document covers. VESPER measures the gap as an embedding delta and builds the skill from certified primitives.
3. **It recovers mid-task.** SkillOpt has no runtime loop at all. VESPER detects deviation at the step where it happens, rolls back, and re-retrieves using the failing screen as the query.
4. **Skills are structured and composable.** A DAG of versioned nodes over atomic primitives, versus one flat file — this is what makes cross-skill splicing and node-level rollback possible.
5. **It is multimodal in the device sense.** Screens, UI elements, MCP tools, and agent delegation from one modality-independent abstraction. SkillOpt is text-space by definition.
6. **It is on-device.** Privacy, offline operation, and no API budget — including for the learning cycle, which for SkillOpt-Sleep explicitly runs on the user's own API budget.

---

## 7. Filing implications — read this part carefully

**SkillOpt is citable prior art against our filing.** The paper and code were public in June–July 2026, before our filing date. It will be found by any competent examiner searching "self-evolving agent skills", "skill validation gate", or "agent skill optimization."

Where it bites, and what to do:

| Our claim | Risk | Recommended narrowing |
|---|---|---|
| **F3 — validation before deployment** | **High.** SkillOpt's held-out gate is exactly "validate before you accept a skill change," and it is published, ablated and shipped. | Claim the *device-simulation* specifics: virtual environment generation, safety-boundary enforcement isolating the candidate from real user state, and a **certification grade that drives runtime policy** (e.g., low grade forces user preview). Do not claim "validating a skill before deployment" in the abstract. |
| **F4 — self-improvement with guards** | **High.** LR schedule, slow update, meta-skill memory and the nightly consolidation cycle cover "bounded, scheduled, memory-assisted self-improvement." | Claim Layer-2 specifically: **modifying skill-graph topology** and **rolling back versioned skill nodes** on measured degradation. A single-document revert is not graph-node versioning. |
| **F1 — gap-driven genesis** | **Low.** No counterpart — SkillOpt never detects a missing capability. | Keep as drafted; emphasize the embedding-delta computation and the retrieval-miss trigger. |
| **F2 — formal composition** | **Low.** Text edits are not composition operators. | Keep; emphasize constraint-satisfaction validity and emergent capability. |
| **V1 / V2 / V4 — visual grounding, live-screen applicability, per-step verification** | **None.** SkillOpt has no screen at all. | These are the strongest surviving pillars. Anchor the independent claims here. |
| **V3 — human approval** | **Partial.** Sleep stages a nightly proposal for human adoption. | Distinguish clearly: ours is a **per-execution, pre-action** preview of a concrete action plan with risk highlighting — not an offline artifact-adoption review. |
| **On-device** | **None.** SkillOpt is cloud-API-based throughout. | Worth stating explicitly in the independent claim; it is a genuine structural limitation of the prior art. |

**Recommended posture:** anchor the independent claim on the *coupling* — a capability gap detected by **live-screen-grounded retrieval** triggering skill genesis, with the created skill certified and then exercised under **per-step visual verification**, all on-device. SkillOpt covers "improve a text skill against a score." It does not cover "create a device skill because the screen said you can't do this, then prove each step against the screen."

---

## 8. If asked "should we just use SkillOpt instead?"

They solve different problems and are, in fact, complementary:

- SkillOpt would be a reasonable engine for **improving the wording of a skill document** so a frozen LLM follows it better. If our skill instructions are ever LLM-interpreted, SkillOpt's optimizer discipline could tune them.
- SkillOpt cannot select a skill for the current screen, verify a tap, recover from a dialog, create a missing skill, or run without a cloud API. Those are precisely the VESPER problems.

A defensible framing for the review: *"SkillOpt is the strongest published work on making one skill document better. VESPER is about deciding which skill applies to the screen in front of the user, proving each step against that screen, and building the skill when none exists. The overlap is real in F3/F4 and we have narrowed those claims accordingly."*

---

*Knox AI · SRI-Bangalore · comparison prepared against SkillOpt v0.2.0 (July 2026).*
