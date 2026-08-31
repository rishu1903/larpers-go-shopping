# TechJam 2026 — Conversational Shopping Copilot

AI-powered conversational product search and recommendation system for **TechJam 2026 Problem Statement 4: Shopping Copilot — AI Conversational Search and Recommendations**.

The system operates over the frozen **50,000-product Amazon Reviews 2023 Clothing, Shoes & Jewelry catalogue** and maintains conversational state across a maximum of 10 user turns.

Our approach combines:

- field-aware BM25 lexical retrieval;
- lightweight catalogue-trained semantic retrieval;
- deterministic candidate reranking;
- multi-turn preference accumulation;
- intent and override handling;
- candidate-aware clarification;
- adaptive long-tail exploration;
- hard budget constraints;
- safe aggregate-profile personalization;
- catalogue-derived robustness evaluation.

The original participant-kit README is preserved at:

`docs/participant-kit/ORIGINAL_README.md`

---

## Current Performance

Latest official **200-session public evaluator** result:

| Metric | Score |
|---|---:|
| Hit Rate@10 | **1.0000** |
| MRR | **0.815145** |
| MTTC | **2.035** |
| Efficiency | **0.8965** |
| Technical Score | **0.923844** |

Official starter baseline:

| Metric | Baseline |
|---|---:|
| Hit Rate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| Efficiency | 0.119 |
| Technical Score | 0.10671 |

The current agent therefore reaches all 200 public targets while substantially improving ranking quality and recommendation speed relative to the starter implementation.

---

# Architecture

```mermaid
flowchart TD
    U[User Message] --> S[Session State]

    P[Aggregate User Profile] --> S

    S --> I[Intent Router]
    S --> H[Hard Constraint Parser]

    I --> Q{Retrieval Route}

    Q -->|Buying / strong lexical evidence| L[Field-aware BM25]
    Q -->|Exploration / sparse browsing| L
    Q -->|Exploration / sparse browsing| D[LSA Semantic Retrieval]

    L --> C[Candidate Pool]
    D --> C

    H --> F[Hard Constraint Filter]
    C --> F

    F --> R[Constraint-aware Reranker]

    R --> TOP[Top Recommendations]
    R --> A[Candidate-aware Clarification]

    A --> S
    TOP --> U
```

The primary architecture deliberately remains **lexical-first**.

Catalogue-derived robustness testing showed that BM25 remains substantially stronger as a standalone retriever, while semantic retrieval provides complementary recovery on queries with weaker lexical overlap.

---

# Conversational Flow

```mermaid
stateDiagram-v2
    [*] --> Initial

    Initial --> Buying: explicit requirement
    Initial --> Browsing: exploring language

    Browsing --> Buying: preference becomes explicit

    Buying --> Buying: additional constraint

    Buying --> Override: user changes requirement
    Browsing --> Override: user changes requirement

    Override --> Buying: stale evidence removed

    Buying --> Exploration: no additional preference
    Browsing --> Exploration: no additional preference

    Exploration --> Buying: new explicit preference

    Buying --> [*]: turn limit / sufficient result
    Exploration --> [*]: turn limit / sufficient result
```

---

# Retrieval Strategy

```mermaid
flowchart LR
    M[Conversation Evidence] --> B[BM25]

    M --> G{Dense route needed?}

    G -->|No| B
    G -->|Yes| S[Semantic LSA]

    B --> U[Candidate Union]
    S --> U

    U --> HC[Hard Constraint Filter]
    HC --> RR[Reranker]
    RR --> T10[Final Top 10]
```

Dense retrieval is **not enabled universally**.

Previous ablations showed that always-on semantic retrieval reduced public MRR. The semantic route is therefore used as a recovery mechanism during exploration or when browsing retrieval is sparse.

---

# Implementation Evolution

```mermaid
flowchart TD
    V0[Starter BM25<br/>Score 0.10671]

    V1[V1 Multi-turn Evidence<br/>Score 0.738582]

    V11[V1.1 Category Preservation<br/>Score 0.752190]

    V2[V2 Constraint-aware Reranker<br/>Score 0.852379]

    V3[V3 Popularity Tie-break<br/>Score 0.920764]

    V4[V4 Adaptive Exploration<br/>Score 0.923814]

    V5[V5 Hybrid Semantic Retrieval<br/>Score 0.923844]

    V6[V6 Candidate-aware Questions]

    V7[V7 Buying / Browsing Intent Routing]

    V8[V8 Context-safe Budget Constraints]

    V9[V9 Safe Profile Personalization]

    V10[V10.2 Domain-gated Robustness Benchmark]

    V11E[V11 End-to-end Shadow Ranking]

    V0 --> V1 --> V11 --> V2 --> V3 --> V4 --> V5 --> V6 --> V7 --> V8 --> V9 --> V10 --> V11E
```

---

# Version History

## V1 — Conversational State

The starter agent was stateless and searched only the latest customer message.

V1 introduced accumulated conversational evidence so later turns benefited from earlier preferences.

Public score:

`0.106710 → 0.738582`

---

## V1.1 — Category Preservation

Separated persistent product-category information from mutable user preferences.

This was important for intent override scenarios: when the user changes a preference, stale preference evidence can be removed without accidentally deleting the product category.

Public score:

`0.738582 → 0.752190`

---

## V2 — Constraint-aware Reranking

Added a deterministic reranker over the BM25 candidate pool.

Signals include:

- category token coverage;
- exact category phrase matches;
- accumulated evidence coverage;
- exact evidence phrases;
- BM25 ranking.

Public score:

`0.752190 → 0.852379`

---

## V3 — Popularity Tie-breaking

Many catalogue products can have effectively identical textual relevance.

V3 uses `rating_number` as a deterministic secondary signal inside equal relevance tiers.

Public result:

- Hit Rate@10: `0.995`
- MRR: `0.811881`
- Technical Score: `0.920764`

---

## V4 — Adaptive Exploration

One public target was a highly ambiguous long-tail product that could not be distinguished from many lexical matches.

V4 introduced:

- clarification exhaustion detection;
- wider candidate retrieval after the user has no more information;
- long-tail exploration ordering;
- seen-product filtering.

Public result:

- Hit Rate@10: **1.000**
- Technical Score: `0.923814`

---

## V5 — Hybrid Semantic Retrieval

Added a local semantic retrieval path trained entirely on the provided 50k catalogue.

Implementation:

- TF-IDF;
- 1–2 gram features;
- Truncated SVD;
- 96-dimensional LSA representation;
- normalized document embeddings;
- `float16` catalogue matrix.

Semantic search is used adaptively rather than universally because always-on dense retrieval reduced public MRR.

Current semantic assets:

- `assets/catalog_lsa.npy`
- `assets/semantic_asins.json`
- `assets/semantic_pipeline.joblib`

Public score reached:

**0.923844**

---

## V6 — Candidate-aware Clarification

The first three turns retain broad `other` clarification because this aligns well with the deterministic competition simulator.

For unresolved later turns, clarification is selected using candidate uncertainty.

Possible dimensions include:

- material;
- color;
- size;
- style;
- use case;
- feature.

Questions are chosen using coverage and normalized entropy across the current candidate set.

---

## V7 — Intent Routing

Added explicit conversational intent state:

- `BUYING`
- `BROWSING`

Browsing language can enable broader retrieval when the lexical pool is weak.

Explicit narrowing such as:

- `I prefer`
- `I would prefer`
- `I'd like`
- `I need`
- `my budget`
- `what matters most`

moves the session into buying mode.

This preserved the public benchmark while making retrieval behavior more intentional.

---

## V8 — Hard Budget Constraints

Added structured price filtering.

Examples correctly understood:

- `under $80`
- `budget under 80`
- `price below 100`
- `I can spend up to 120`
- `$50 to $100`

The parser intentionally rejects unrelated measurements such as:

- `fits up to 8-inch wrist circumference`
- `water resistant up to 30m`

This avoids interpreting arbitrary catalogue measurements as budget constraints.

---

## V9 — Safe Profile Personalization

Uses anonymized aggregate profile tags to influence **which clarification dimension** may matter.

The profile never invents a concrete preference.

For example:

`material matters historically`

does **not** imply:

`user wants cotton`

Profile information is only allowed to break near-ties between candidate-driven clarification choices.

Priority:

1. current explicit preference;
2. current candidate uncertainty;
3. aggregate historical profile.

---

# V10.2 — Robustness Benchmark

The public evaluator contains 200 labeled sessions, while 800 sessions remain private.

To reduce public-set overfitting, we added a self-supervised catalogue-derived robustness benchmark that does **not use public labels**.

The benchmark:

1. identifies explicit product concepts in catalogue metadata;
2. removes the original concept keyword from the query;
3. generates semantically equivalent paraphrases;
4. restricts concepts to sensible product domains;
5. treats every qualifying same-category product as relevant;
6. compares lexical and semantic candidate generation.

Examples:

`waterproof`

→ `something designed to keep water from getting through`

`breathable`

→ `something that helps reduce heat buildup during wear`

`non-slip`

→ `something with secure traction on slick surfaces`

## V10.2 Results

130 cases, balanced across 13 concept families:

| Retrieval route | Hit@10 | Hit@100 | Macro Recall@100 |
|---|---:|---:|---:|
| Lexical | **58.46%** | **90.00%** | **67.78%** |
| Semantic | 16.15% | 59.23% | 18.86% |
| Hybrid candidate union | **60.00%** | **94.62%** | **73.43%** |

Complementarity:

- 71 cases found by both routes;
- 46 lexical-only cases;
- 6 semantic rescue cases;
- 7 missed by both.

This supports the current design:

> Use lexical retrieval as the precision-first primary route and semantic retrieval as a complementary recovery mechanism.

---

# V11 — End-to-End Shadow Ranking

V10.2 measures whether relevant products enter the candidate pool.

V11 exercises the actual production interface:

```text
Agent.reset(...)
Agent.respond(...)
```

against the same catalogue-derived paraphrase benchmark and measures the final top-10 recommendations shown to the user.

```mermaid
sequenceDiagram
    participant U as Synthetic User
    participant A as Production Agent
    participant R as Retrieval + Reranker

    U->>A: Buying-style paraphrased requirement
    A->>R: Normal precision-first retrieval
    R-->>A: Ranked candidates
    A-->>U: Turn-1 top 10

    U->>A: No additional preference
    A->>R: Exploration / semantic recovery
    R-->>A: Unseen alternatives
    A-->>U: Turn-2 top 10
```

## V11 Results

Across the 130 domain-gated shadow cases:

| Metric | Result |
|---|---:|
| Turn-1 Hit@10 | **77.69%** |
| Turn-1 MRR@10 | **0.4988** |
| New turn-2 rescues | **14** |
| Cumulative hit by turn 2 | **88.46%** |
| Rescue rate among initial misses | **48.28%** |
| Missed after both turns | **15 / 130** |

The production agent deliberately avoids repeating already-seen products during exploration. Therefore a turn-1 hit remains a successful session even when the target is not repeated on turn 2.

V11 shows that exploration is valuable: almost half of the cases that missed on the first recommendation set were recovered by the second.

However, semantic promotion remains an opportunity. Of the six V10.2 cases in which semantic retrieval found a relevant product that lexical retrieval completely missed at depth 100, only one was promoted into the final top 10 during exploration.

This motivates the next experiment:

> Improve semantic candidate fusion during exploration without changing the protected turn-1 lexical ranking path.

## V12 — Semantic-Aware Exploration Fusion

V11 showed that semantic retrieval could recover products missed by lexical search, but many semantic-only candidates were not promoted into the final recommendation set.

V12 adds a bounded semantic reciprocal-rank signal during **exploration only**. The normal buying path remains unchanged.

Semantic promotion is deliberately conservative:

- only semantic-only candidates receive the bonus;
- hybrid/lexical candidates receive no additional semantic boost;
- candidates must remain compatible with the active product category;
- semantic retrieval rank is used instead of raw cosine similarity;
- the feature activates only during exploration.

A label-free ablation tested weights from `0.0` to `1.5`. A weight of `1.0` was selected because it was the smallest value that improved session coverage while preserving ranking quality.

| Metric | V11 | V12 |
|---|---:|---:|
| Shadow Turn-1 Hit@10 | 77.69% | 77.69% |
| Shadow cumulative Hit@10 | 88.46% | **89.23%** |
| Turn-2 rescues | 14 | **15** |
| Remaining misses | 15 | **14** |
| Semantic-only rescues | 1 / 6 | **2 / 6** |
| Public Hit@10 | 1.000 | **1.000** |
| Public MRR | 0.815145 | **0.815145** |
| Public TechnicalScore | 0.923844 | **0.923844** |

The result suggests semantic retrieval is useful as a targeted recovery mechanism, but should remain subordinate to the stronger lexical relevance path.

---

# V13 — Hard Constraint Parser Generalization Hardening

`data/public_set.jsonl` contains no free-text customer messages at all -- every budget phrase the parser was ever validated against locally is synthesized by `evaluator/local_evaluator.py`'s own template functions. A code and data audit found that the parser and its session-state coupling had, in places, been tuned to that same narrow template distribution rather than to budget language in general:

- the intent-override detector was a hardcoded exact match on two substrings (`"actually"` and `"ignore my earlier preference"`), character-for-character the evaluator's own override sentence;
- negated bounds such as `not over $100` were parsed as the opposite positive constraint (`min_price=100`);
- there was no way to explicitly cancel a budget (`"actually, no budget limit"` was a silent no-op);
- the bound vocabulary was closed -- any phrasing outside ~15 templates silently dropped the constraint.

None of this showed up in the public benchmark, because the public benchmark only exercises the same template phrasings the code was tuned against. V13 fixes the four gaps above with small, additive changes to `src/hard_constraints.py` and `src/state.py` -- no restructuring of the existing money-context → approximate-language → range/bound staged pipeline, and no new hard-filter attribute types (brand/size/color/etc. remain soft reranking signals by design).

Examples now handled correctly:

- `Not above $50, please.` / `Not under $80.` → no longer misparsed as an inverted bound.
- `Actually, no budget limit anymore.` → clears an existing budget instead of leaving it stale.
- `On second thought, forget my earlier preference -- here's what I actually need: waterproof shoes.` → triggers the override despite not matching the evaluator's literal sentence.
- `Show me something cheaper than $60.` / `I'm capped at $90.` → new generic vocabulary.
- `Budget: $.75` → bare-decimal amounts now parse.

A new held-out parser eval suite (`scripts/budget_constraint_eval.py`), phrased independently of the evaluator's templates, was built first and used to measure the change:

| Metric | Before | After |
|---|---:|---:|
| Suite pass rate | 60% (15/25) | **96% (24/25)** |
| Detection precision | 0.625 | **0.909** |
| Detection recall | 0.5 | **1.000** |
| Detection F1 | 0.556 | **0.952** |
| Negation accuracy | 0.0 | **1.000** |
| Numeric accuracy | 0.6 | **1.000** |
| Multi-turn state accuracy | 0.667 | **1.000** |
| Public Hit@10 | 1.000 | **1.000** |
| Public MRR | 0.815145 | **0.815145** |
| Public TechnicalScore | 0.923844 | **0.923844** |

The public benchmark is byte-identical before and after -- every change is additive (a new guard, a new sentinel, new vocabulary) rather than a modification to any currently-matching pattern, so nothing that previously passed can regress.

Known, documented limitations not addressed in this pass:

- negation handling is adjacency-based only -- semantically displaced negation such as `I don't want to spend more than $90` is not detected;
- the money-context gate scans the whole message rather than text near the numeric match, so an unrelated money-context word can still license an unrelated `up to N` phrase (e.g. `The cost is unclear, but it holds up to 10 items.`) -- tracked in the eval suite's `ACCEPTED_LIMITATIONS` rather than silently unmeasured.

---

# Repository Structure

```text
.
├── assets/
│   ├── catalog_lsa.npy
│   ├── semantic_asins.json
│   └── semantic_pipeline.joblib
│
├── data/
│   ├── catalog.jsonl
│   └── public_set.jsonl
│
├── evaluator/
│   └── local_evaluator.py
│
├── experiments/
│   ├── v8_hard_constraints.json
│   ├── v8_1_money_safe_constraints.json
│   ├── v9_safe_personalization.json
│   ├── v10_2_concept_smoke.json
│   ├── v10_2_concept_robustness.json
│   ├── v11_end_to_end_shadow.json
│   └── v13_budget_constraint_hardening_after.json
│
├── scripts/
│   ├── build_semantic_index.py
│   ├── budget_constraint_eval.py
│   ├── concept_robustness_eval.py
│   └── end_to_end_shadow_eval.py
│
├── src/
│   ├── dialogue.py
│   ├── hard_constraints.py
│   ├── intent.py
│   ├── profile.py
│   ├── questions.py
│   ├── reranker.py
│   ├── retrieval.py
│   ├── semantic.py
│   └── state.py
│
├── starter/
│   └── agent.py
│
└── tests/
```

---

# Setup

Python 3.10+ is recommended.

Create the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full automated test suite:

```powershell
python -m unittest -v
```

---

# Official Public Evaluation

Run:

```powershell
python -m evaluator.local_evaluator --output experiments\latest_public_eval.json
```

Do not modify:

- evaluator logic;
- public labels;
- target ASINs;
- official evaluation configuration.

---

# Robustness Evaluation

Run the domain-gated retrieval benchmark:

```powershell
python -m scripts.concept_robustness_eval --cases 130 --output experiments\v10_2_concept_robustness.json
```

Run the end-to-end top-10 benchmark:

```powershell
python -m scripts.end_to_end_shadow_eval --output experiments\v11_end_to_end_shadow.json
```

Run the budget constraint parser eval suite:

```powershell
python -m scripts.budget_constraint_eval --output experiments\v13_budget_constraint_hardening.json
```

---

# Design Principles

### Precision before breadth

Dense retrieval is not automatically better than lexical search.

The system keeps BM25 as its default high-precision retrieval route.

### Semantic retrieval as recovery

Semantic search is activated when there is evidence that lexical retrieval may be insufficient.

### Explicit session intent wins

Current user requirements always outrank historical profile information.

### No fabricated preferences

Aggregate profile tags influence clarification strategy, never concrete product values.

### Deterministic and reproducible

The current production pipeline does not require an external paid LLM API.

This minimizes:

- latency;
- cost;
- nondeterminism;
- external dependencies.

### Public-set restraint

Production changes are not accepted solely because they improve the 200 public sessions.

Catalogue-derived shadow tests provide an additional evaluation axis for private-set robustness.

---

# Team Workflow

Recommended Git workflow:

```text
main
│
├── feature/<feature-name>
├── fix/<bug-name>
└── experiment/<experiment-name>
```

For each change:

1. create a branch;
2. implement the change;
3. run `python -m unittest -v`;
4. run the relevant evaluation;
5. compare against the protected benchmark;
6. commit only if the change is justified;
7. open a pull request into `main`.

---

# Protected Public Benchmark

Treat this as the current regression baseline:

```text
Hit Rate@10      1.000000
MRR              0.815145
MTTC             2.035
Efficiency       0.8965
Technical Score  0.923844
```

A production change that reduces Hit Rate@10 should normally be rejected unless there is compelling evidence of better private-set generalization.

---

# Competition Constraints

Key constraints from the participant kit:

- frozen 50,000-product catalogue;
- maximum 10 turns per session;
- exact `parent_asin` recommendation matching;
- 200 labeled public sessions;
- 800 private evaluation sessions;
- catalogue is read-only;
- participant-visible metadata only;
- no modification of evaluator/public labels.

---

# Current Status

Completed:

- multi-turn conversational memory;
- override handling;
- field-aware lexical retrieval;
- constraint-aware reranking;
- popularity tie-breaking;
- adaptive exploration;
- semantic candidate recovery;
- buying/browsing intent routing;
- candidate-aware clarification;
- context-safe budget filtering;
- safe profile personalization;
- catalogue-derived robustness benchmark.

Completed:
- V11 end-to-end shadow top-10 analysis;
- V12 semantic-aware exploration candidate fusion;
- V13 hard constraint parser generalization hardening (negation handling, explicit budget removal, evaluator-independent override detection, expanded numeric/vocabulary coverage).

Next:
- clause-level negation scope beyond adjacency (e.g. "I don't want to spend more than $X");
- proximity-based money-context detection to close the remaining false-positive case tracked in the budget constraint eval suite's ACCEPTED_LIMITATIONS.

Next production change will be chosen based on evidence from the relevant robustness suite rather than assumed in advance.