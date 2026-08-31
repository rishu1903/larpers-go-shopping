TechJam 2026 — Shopping Copilot

A local-first conversational shopping search system for TechJam 2026 Problem Statement 4: Shopping Copilot — AI Conversational Search and Recommendations.

The system is designed for a frozen 50,000-product catalogue and a maximum of 10 dialogue turns per session. The production path prioritizes deterministic, reproducible retrieval and ranking, with semantic search used only as bounded recovery.

Current Production Architecture

User message
    ↓
Session state + structured slots
    ↓
Buying / browsing intent routing
    ↓
Field-aware BM25 retrieval
    ↓
Conditional LSA semantic recovery
    ↓
Hard budget filtering
    ↓
Constraint-aware deterministic reranking
    ↓
Observed negative-constraint filtering
    ↓
Candidate-aware clarification / recommendations
    ↓
Failure-aware protected recovery on later misses

Design principle

The system does not let every new signal reorder the entire candidate list.

The strongest proven path remains lexical-first. Semantic retrieval, profile information, structured context, failure recovery, and negative constraints are all deliberately bounded so that they add capability without replacing behavior that already works well.

Current Results

The current protected public benchmark result is:

Metric

Result

Samples

200

Hit Rate@10

1.000000

MRR

0.820423

MTTC

2.045

Efficiency

0.8955

Recommended Technical Score

0.925227

Scenario-level results:

Scenario

Hit@10

MRR

MTTC

Boundary

1.000

1.000000

2.600

Browsing

1.000

0.768333

1.775

Buying

1.000

0.822307

1.650

Intent Override

1.000

0.894444

3.633

The latest cleanup preserves this public behavior while adding conservative support for explicit negative constraints.

V15 — Intent-Aware Negative Constraints

A key private-set robustness issue was that negated attributes could previously become positive retrieval evidence.

Example:

User:
"T-Shirts. without polyester"

Old behavior:
"polyester" entered positive retrieval / slot evidence.

New behavior:
material = polyester
is recorded as an exclusion,
removed from positive retrieval evidence,
and used only to remove products whose catalogue
explicitly confirms polyester.

The production rule is deliberately asymmetric:

confirmed forbidden attribute
    → remove product

missing catalogue evidence
    → keep product

This avoids over-filtering products simply because the frozen catalogue is incomplete.

Safe canonical aliases

The controlled vocabulary currently normalizes exact equivalents including:

hood → hooded

pocket → pockets

insulation → insulated

nonslip
non-slip
non slip
slip-resistant
slip resistant
anti-slip
anti slip
    → non_slip

water-resistant
water resistant
    → water_resistant

stretchy → stretch

jog / jogging → running

trail / trails → hiking

Loose semantic assumptions such as warm → insulated are intentionally not treated as hard exclusions.

Retrieval

Lexical path

BM25 remains the primary high-precision retrieval route.

The query is field-aware: category evidence and preference evidence are scoped separately instead of flattening every signal into one undifferentiated string.

Semantic path

Semantic retrieval uses a local TF-IDF + TruncatedSVD (LSA) representation.

Assets:

assets/catalog_lsa.npy
assets/semantic_asins.json
assets/semantic_pipeline.joblib

Semantic search is complementary rather than universal. It is most useful during exploration and recovery when lexical retrieval may be too narrow.

Structured Conversation State

The agent maintains structured state for:

category
budget
material
color
size
style
brand
feature
use case

State also tracks lifecycle information for:

intent overrides
clarification history
recommended products
failed recommendation sets
intent epochs
failure depth

Current-session evidence always takes priority over historical profile information.

Context Distillation

Structured slot state is used to rebuild active retrieval context safely after overrides.

Important safeguards:

cleared slots are not restored;

unknown free text is preserved;

weak inference is not injected as hard evidence;

budget remains structured instead of becoming free-text retrieval evidence;

disabling context distillation preserves the older retrieval path.

Negative Constraint Pipeline

User message
    ↓
extract_exclusions(...)
    ↓
remove negated phrase from positive retrieval text
    ↓
normal retrieval + reranking
    ↓
build ProductProfile only for exclusion evaluation
    ↓
filter_observed_exclusions(...)

ProductProfile uses provenance-aware product evidence from trusted catalogue fields such as:

title
features
description
categories (for use case)
selected native detail keys

Store names are not treated as product attributes.

Generic details dictionaries are not flattened indiscriminately.

Failure-Aware Protected Recovery

After explicit recommendation failure, retrieval breadth can increase.

The deeper candidate pool is not allowed to replace the normal ranking wholesale.

Instead:

trusted baseline recommendations
        +
bounded recovery candidate

Failure memory is scoped to the current intent epoch so an explicit override resets the active miss streak.

Clarification Strategy

The system distinguishes broad browsing from stronger buying intent.

Clarification is candidate-aware and avoids repeatedly asking about dimensions already established in the current session.

Historical aggregate profile information can break near-ties between clarification dimensions, but it never fabricates a concrete preference.

Robustness Evaluation

The project does not accept changes solely because they improve the 200 public sessions.

The evaluation stack is:

unit / regression tests
        ↓
catalogue-derived label-free robustness checks
        ↓
official public evaluator

The catalogue-derived benchmark uses concept families such as:

waterproof
breathable
non-slip
insulated
lightweight
hooded
pockets
stretch
adjustable
running
hiking
winter

This helps detect overfitting and retrieval failures that may appear in the 800-session private evaluation.

Repository Structure

.
├── assets/
│   ├── catalog_lsa.npy
│   ├── semantic_asins.json
│   └── semantic_pipeline.joblib
│
├── data/
│   ├── catalog.jsonl        # local / ignored
│   └── public_set.jsonl
│
├── docs/
│   └── participant-kit/
│       └── ORIGINAL_README.md
│
├── evaluator/
│   └── local_evaluator.py
│
├── experiments/
│   └── historical accepted evaluation outputs
│
├── scripts/
│   ├── build_semantic_index.py
│   ├── concept_robustness_eval.py
│   ├── end_to_end_shadow_eval.py
│   ├── v12_fusion_ablation.py
│   ├── v13_failure_orchestration_eval.py
│   ├── v13_protected_recovery_eval.py
│   ├── v14_context_distillation_eval.py
│   └── v14_context_state_eval.py
│
├── src/
│   ├── context.py
│   ├── dialogue.py
│   ├── fusion.py
│   ├── hard_constraints.py
│   ├── intent.py
│   ├── orchestration.py
│   ├── product_profile.py
│   ├── profile.py
│   ├── questions.py
│   ├── relevance_gate.py
│   ├── reranker.py
│   ├── retrieval.py
│   ├── semantic.py
│   ├── shopping_intent.py
│   ├── slots.py
│   └── state.py
│
├── starter/
│   └── agent.py
│
├── tests/
├── DATA_ATTRIBUTION.md
├── requirements.txt
└── README.md

Temporary V15 audit, shadow-ranking, smoke-result, and patch artifacts are intentionally excluded from the cleaned repository.

Setup

Python 3.10+.

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Run the full test suite:

python -m unittest -v

The cleaned V15 code currently has:

166 tests

Public Evaluation

python -m evaluator.local_evaluator --output experiments\v15_public_eval.json

Protected expected metrics:

sample_count                 200
hit_rate_at_10               1.0
mrr                          0.820423
mttc                         2.045
efficiency                   0.8955
recommended_technical_score  0.925227

Robustness Evaluation

Catalogue-derived concept robustness:

python -m scripts.concept_robustness_eval `
  --cases 130 `
  --output experiments\concept_robustness.json

End-to-end shadow evaluation:

python -m scripts.end_to_end_shadow_eval `
  --output experiments\end_to_end_shadow.json

Competition Constraints

The implementation is designed around the participant-kit constraints:

frozen 50,000-product catalogue;

maximum 10 turns;

exact parent_asin recommendation matching;

200 public sessions;

800 separate private sessions;

read-only catalogue;

participant-visible metadata only;

no evaluator or target-label modification;

lightweight local execution;

network access may not be available in final scoring.

The production path therefore does not require a paid external LLM API.

Development Rules

A production change should satisfy all three:

Architectural justification — it solves a real shopping-search problem.

Direct behavioral test — the specific behavior is tested independently.

Regression protection — existing public and robustness behavior must not regress.

A small public-score increase alone is not sufficient reason to ship a change.

Current Priorities

The strongest next areas are:

better product-type / query understanding;

category-specific clarification with higher information gain;

improved private-set robustness without changing the proven ranking path;

latency and memory profiling for final packaging.

Broad positive relevance reranking was tested and rejected because it hurt ranking quality. Structured relevance is therefore currently used only where it has a clear, conservative role: observed negative constraints.