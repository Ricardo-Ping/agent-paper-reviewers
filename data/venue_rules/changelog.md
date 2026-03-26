# Venue Policy Changelog

- 2026-03-25: initialized V1 profiles for 9 venues (NeurIPS, ICLR, ICML, ACL-ARR, EMNLP, KDD, AAAI, CVPR, ECCV).
- 2026-03-25: enabled dynamic OpenReview policy fallback marker (`policy_needs_manual_check`) for venues with non-static rebuttal rules.
- 2026-03-25: refined SIGMOD/VLDB/ICDE required checks with database-system-specific criteria (workload diversity, scalability, efficiency tradeoff, baseline fairness, system reproducibility, and top-venue related-work coverage).
- 2026-03-25: added venue-aware `decision_policy` thresholds for report decisioning (high-competition vs medium-competition defaults with overridable per-venue settings).
