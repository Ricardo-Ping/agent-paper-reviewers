from __future__ import annotations

from copy import deepcopy
from typing import Any


def student_pack_analysis_template() -> dict[str, Any]:
    """Template contract for agent-authored analysis payload."""
    return {
        "paper_title": "Your Paper Title",
        "venue": "ICLR 2026",
        "language": "en",
        "decision": {
            "recommendation": "Hold for Revision",
            "one_line_reason": "Top P0/P1 risks are not yet closed.",
            "top_issues": [
                {
                    "id": "P0-1",
                    "priority": "P0",
                    "title": "Statistical significance missing",
                    "problem": "Main results do not report multi-seed variance and p-values.",
                    "impact": "High reject risk for soundness and reproducibility.",
                    "evidence_anchor": "Section 6, Table 2",
                    "fix_steps": [
                        "Run at least 5 seeds for key results.",
                        "Report mean+/-std and pairwise significance tests.",
                    ],
                    "time_estimate": "2 days",
                    "rebuttal_hint": "Point to revised table and statistical test setup.",
                }
            ],
        },
        "action_items": [
            {
                "id": "A-001",
                "priority": "P0",
                "title": "Add statistical validation block",
                "problem": "No significance analysis on major claims.",
                "evidence_anchor": "Section 6 / Table 2",
                "steps": [
                    "Run 5 seeds on benchmark A/B.",
                    "Add p-value report against strongest baseline.",
                ],
                "time_estimate": "2 days",
                "gpu_estimate": "12 GPU-hours",
                "rebuttal_link": "R1 statistical concern",
            }
        ],
        "rebuttal_items": [
            {
                "review_id": "R1",
                "concern": "Statistical significance is not sufficiently supported.",
                "risk_id": "P0-1",
                "status": "in_progress",
                "response": (
                    "Thank you for raising this concern. We added multi-seed evaluation "
                    "(n=5), report mean+/-std in revised Table 2, and include pairwise "
                    "significance tests versus the strongest baseline."
                ),
                "new_evidence": [
                    "Revised Table 2 with mean+/-std over 5 seeds.",
                    "Appendix A.3 significance test details (p-values).",
                ],
            }
        ],
    }


def render_student_pack_markdown(
    analysis_payload: dict[str, Any],
    language: str = "en",
) -> dict[str, str]:
    """Render 3 user-facing markdown files from agent analysis payload."""
    payload = deepcopy(analysis_payload)
    lang = (language or str(payload.get("language", "en"))).strip().lower()
    if lang not in {"en", "zh"}:
        lang = "en"

    return {
        "001-submission-decision.md": _render_submission_decision(payload, lang),
        "002-action-items.md": _render_action_items(payload, lang),
        "003-rebuttal-draft.md": _render_rebuttal(payload, lang),
    }


def _render_submission_decision(payload: dict[str, Any], lang: str) -> str:
    decision = payload.get("decision", {})
    if not isinstance(decision, dict):
        decision = {}
    recommendation = str(decision.get("recommendation", "Hold for Revision")).strip()
    reason = str(
        decision.get("one_line_reason", "Key risks should be closed before submission.")
    ).strip()
    issues = _collect_top_issues(payload)

    header = "# Submission Decision (001)"
    intro = [
        f"- Recommendation: **{recommendation or 'Hold for Revision'}**",
        f"- One-line reason: {reason or 'Key risks remain unresolved.'}",
        "",
        "## Top Issues (Priority-Ordered)",
    ]
    if lang == "zh":
        header = "# [ZH] Submission Decision (001)"
        intro = [
            f"- Conclusion: **{recommendation or 'Hold for Revision'}**",
            f"- One-line reason: {reason or 'Key risks remain unresolved.'}",
            "",
            "## [ZH] Top Issues (Priority-Ordered)",
        ]

    lines = [header, ""]
    lines.extend(intro)
    if not issues:
        lines.append("- No issue item found. Ask the agent to fill `decision.top_issues`.")
    for idx, issue in enumerate(issues[:5], start=1):
        lines.extend(_render_issue_block(issue, idx))
    return "\n".join(lines).strip() + "\n"


def _render_action_items(payload: dict[str, Any], lang: str) -> str:
    action_items = payload.get("action_items", [])
    if not isinstance(action_items, list):
        action_items = []
    if not action_items:
        action_items = _derive_actions_from_issues(_collect_top_issues(payload))

    header = "# Action Items (002)"
    intro = "Ordered by P0 -> P1 -> P2; each item includes anchor and executable steps."
    if lang == "zh":
        header = "# [ZH] Action Items (002)"
        intro = "Ordered by P0 -> P1 -> P2 with anchors and executable steps."

    lines = [header, "", intro, ""]
    if not action_items:
        lines.append("- No action items found. Ask the agent to output `action_items`.")
        return "\n".join(lines).strip() + "\n"

    sorted_items = sorted(action_items, key=lambda x: _priority_rank(str(x.get("priority", "P2"))))
    for idx, item in enumerate(sorted_items, start=1):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", f"A-{idx:03d}")).strip()
        title = str(item.get("title", "Action item")).strip()
        prio = str(item.get("priority", "P2")).strip().upper()
        problem = str(item.get("problem", "")).strip()
        anchor = str(item.get("evidence_anchor", "N/A")).strip()
        time_est = str(item.get("time_estimate", "N/A")).strip()
        gpu_est = str(item.get("gpu_estimate", "N/A")).strip()
        rebuttal_link = str(item.get("rebuttal_link", "N/A")).strip()
        steps = item.get("steps", [])
        if not isinstance(steps, list):
            steps = []

        lines.extend(
            [
                f"## {idx}. {item_id} [{prio}] {title}",
                f"- Problem: {problem or 'TBD'}",
                f"- Evidence anchor: {anchor}",
                f"- Estimated time: {time_est}; GPU budget: {gpu_est}",
                f"- Rebuttal link: {rebuttal_link}",
                "- Steps:",
            ]
        )
        if steps:
            for step_idx, step in enumerate(steps, start=1):
                lines.append(f"  {step_idx}. {str(step).strip()}")
        else:
            lines.append("  1. TBD actionable step.")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_rebuttal(payload: dict[str, Any], lang: str) -> str:
    items = payload.get("rebuttal_items", [])
    if not isinstance(items, list):
        items = []

    header = "# Rebuttal Draft (003)"
    if lang == "zh":
        header = "# [ZH] Rebuttal Draft (003)"

    lines = [
        header,
        "",
        "This draft is agent-generated and should be manually verified before submission.",
        "",
    ]
    if not items:
        lines.append("- No `rebuttal_items` found. Ask the agent for per-reviewer draft responses.")
        return "\n".join(lines).strip() + "\n"

    for row in items:
        if not isinstance(row, dict):
            continue
        review_id = str(row.get("review_id", "R?")).strip()
        concern = str(row.get("concern", "N/A")).strip()
        risk_id = str(row.get("risk_id", "N/A")).strip()
        status = str(row.get("status", "pending")).strip()
        response = str(row.get("response", "")).strip()
        evidence = row.get("new_evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        lines.extend(
            [
                f"## {review_id}: {concern}",
                f"- Risk mapping: {risk_id} (status: {status})",
                "",
                "> " + (response or "TBD response."),
                "",
                "- New evidence:",
            ]
        )
        if evidence:
            for ev in evidence:
                lines.append(f"  - {str(ev).strip()}")
        else:
            lines.append("  - TBD evidence item.")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _collect_top_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    decision = payload.get("decision", {})
    if isinstance(decision, dict) and isinstance(decision.get("top_issues"), list):
        return [item for item in decision["top_issues"] if isinstance(item, dict)]
    for key in ("top_issues", "risks", "issues"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _derive_actions_from_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, issue in enumerate(issues[:8], start=1):
        issue_id = str(issue.get("id", f"P?-{idx}")).strip()
        title = str(issue.get("title", issue.get("problem", "Action item"))).strip()
        fix_steps = issue.get("fix_steps", [])
        if not isinstance(fix_steps, list):
            fix_steps = []
        out.append(
            {
                "id": f"A-{idx:03d}",
                "priority": str(issue.get("priority", "P2")).strip().upper(),
                "title": title or f"Address {issue_id}",
                "problem": str(issue.get("problem", "")).strip(),
                "evidence_anchor": str(issue.get("evidence_anchor", "N/A")).strip(),
                "steps": [str(x).strip() for x in fix_steps if str(x).strip()],
                "time_estimate": str(issue.get("time_estimate", "N/A")).strip(),
                "gpu_estimate": str(issue.get("gpu_estimate", "N/A")).strip(),
                "rebuttal_link": str(issue.get("rebuttal_hint", issue_id)).strip(),
            }
        )
    return out


def _render_issue_block(issue: dict[str, Any], idx: int) -> list[str]:
    issue_id = str(issue.get("id", f"ISSUE-{idx}")).strip()
    prio = str(issue.get("priority", "P2")).strip().upper()
    title = str(issue.get("title", issue.get("problem", "Issue"))).strip()
    problem = str(issue.get("problem", "")).strip()
    impact = str(issue.get("impact", "")).strip()
    anchor = str(issue.get("evidence_anchor", "N/A")).strip()
    time_est = str(issue.get("time_estimate", "N/A")).strip()
    rebuttal_hint = str(issue.get("rebuttal_hint", "N/A")).strip()
    fix_steps = issue.get("fix_steps", [])
    if not isinstance(fix_steps, list):
        fix_steps = []

    lines = [
        f"### {idx}. {issue_id} [{prio}] {title}",
        f"- Problem: {problem or 'TBD'}",
        f"- Impact: {impact or 'TBD'}",
        f"- Evidence anchor: {anchor}",
        f"- Estimated time: {time_est}",
        f"- Rebuttal anchor: {rebuttal_hint}",
        "- Suggested fix:",
    ]
    if fix_steps:
        for step_idx, step in enumerate(fix_steps, start=1):
            lines.append(f"  {step_idx}. {str(step).strip()}")
    else:
        lines.append("  1. Add one concrete executable fix step.")
    lines.append("")
    return lines


def _priority_rank(priority: str) -> int:
    p = priority.strip().upper()
    if p == "P0":
        return 0
    if p == "P1":
        return 1
    if p == "P2":
        return 2
    return 9

