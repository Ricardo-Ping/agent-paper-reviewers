# Cross-Dialect SQL Translation with Adaptive Validation

## Abstract
We present an LLM-based SQL dialect translation system that targets real production migration scenarios.
The system combines dialect-aware schema encoding, constrained decoding, and execution-guided repair to reduce syntax and semantic failures.
Across cross-dialect benchmarks, our system improves exact-match accuracy and significantly lowers runtime execution errors.

## Contributions
1. We propose a dialect-aware SQL translation architecture with explicit schema grounding.
2. We introduce execution-guided iterative repair to improve reliability on hard SQL patterns.
3. We provide a comprehensive evaluation across multiple dialect pairs with latency and robustness analysis.

## Method
Our method contains a retriever-augmented planner, constrained SQL generator, and post-generation validator.
The validator executes translated SQL on sampled schemas and repairs invalid queries with focused prompts.

## Experiments
We benchmark against strong SQL translation baselines and evaluate exact match, execution success rate, and latency.
We run ablations for schema grounding, constrained decoding, and iterative repair.

## Limitations
Our benchmarks do not yet cover all long-tail enterprise dialect extensions.
