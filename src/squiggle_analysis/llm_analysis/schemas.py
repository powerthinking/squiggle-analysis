"""Output schema definitions for LLM analysis responses."""

from typing import Any

# Schema definition for LLM analysis output
# This documents the expected structure; validation happens at runtime
ANALYSIS_SCHEMA = {
    "headline_summary": ["string (3 bullet points summarizing the key takeaways)"],
    "overall_assessment": {
        "verdict": "strong | moderate | weak | inconclusive",
        "confidence": "float 0.0-1.0",
        "why": ["string (3 reasons supporting the verdict)"],
    },
    "key_findings": [
        {
            "title": "string (brief finding title)",
            "evidence": ["string (specific values/quotes from report)"],
            "interpretation": "string (what this finding means)",
            "impact": "high | medium | low",
        }
    ],
    "inconsistencies_or_red_flags": [
        {
            "issue": "string (description of the problem)",
            "why_it_matters": "string (impact on interpretation)",
            "how_to_verify": ["string (steps to check/resolve)"],
        }
    ],
    "hypotheses": [
        {
            "hypothesis": "string (testable claim)",
            "supporting_evidence": ["string"],
            "counter_evidence": ["string"],
            "confidence": "float 0.0-1.0",
            "tests": ["string (experiments to validate)"],
        }
    ],
    "recommended_next_actions": [
        {
            "action": "string (what to do)",
            "goal": "string (why do it)",
            "steps": ["string (how to do it)"],
            "effort": "low | medium | high",
            "expected_impact": "low | medium | high",
        }
    ],
    "suggested_report_improvements": [
        {
            "change": "string (what to add/modify in reporting)",
            "benefit": "string (why this helps)",
            "implementation_hint": "string (how to implement)",
        }
    ],
    "questions_to_answer_next": ["string (follow-up questions for future analysis)"],
}


def validate_analysis_response(response: dict[str, Any]) -> list[str]:
    """Validate LLM response against expected schema.

    Args:
        response: Parsed JSON response from LLM

    Returns:
        List of validation errors (empty if valid)
    """
    errors: list[str] = []

    # Check required top-level keys
    required_keys = [
        "headline_summary",
        "overall_assessment",
        "key_findings",
        "recommended_next_actions",
    ]

    for key in required_keys:
        if key not in response:
            errors.append(f"Missing required key: {key}")

    # Validate headline_summary
    if "headline_summary" in response:
        if not isinstance(response["headline_summary"], list):
            errors.append("headline_summary must be a list")
        elif len(response["headline_summary"]) < 1:
            errors.append("headline_summary must have at least 1 item")

    # Validate overall_assessment
    if "overall_assessment" in response:
        assessment = response["overall_assessment"]
        if not isinstance(assessment, dict):
            errors.append("overall_assessment must be a dict")
        else:
            if "verdict" not in assessment:
                errors.append("overall_assessment missing 'verdict'")
            elif assessment["verdict"] not in ["strong", "moderate", "weak", "inconclusive"]:
                errors.append(
                    f"overall_assessment.verdict must be one of: "
                    f"strong, moderate, weak, inconclusive (got: {assessment['verdict']})"
                )

            if "confidence" in assessment:
                try:
                    conf = float(assessment["confidence"])
                    if not 0.0 <= conf <= 1.0:
                        errors.append("overall_assessment.confidence must be 0.0-1.0")
                except (TypeError, ValueError):
                    errors.append("overall_assessment.confidence must be a number")

    # Validate key_findings
    if "key_findings" in response:
        if not isinstance(response["key_findings"], list):
            errors.append("key_findings must be a list")
        else:
            for i, finding in enumerate(response["key_findings"]):
                if not isinstance(finding, dict):
                    errors.append(f"key_findings[{i}] must be a dict")
                    continue
                if "title" not in finding:
                    errors.append(f"key_findings[{i}] missing 'title'")
                if "impact" in finding and finding["impact"] not in ["high", "medium", "low"]:
                    errors.append(
                        f"key_findings[{i}].impact must be high/medium/low "
                        f"(got: {finding['impact']})"
                    )

    # Validate recommended_next_actions
    if "recommended_next_actions" in response:
        if not isinstance(response["recommended_next_actions"], list):
            errors.append("recommended_next_actions must be a list")
        else:
            for i, action in enumerate(response["recommended_next_actions"]):
                if not isinstance(action, dict):
                    errors.append(f"recommended_next_actions[{i}] must be a dict")
                    continue
                if "action" not in action:
                    errors.append(f"recommended_next_actions[{i}] missing 'action'")
                for field in ["effort", "expected_impact"]:
                    if field in action and action[field] not in ["low", "medium", "high"]:
                        errors.append(
                            f"recommended_next_actions[{i}].{field} must be low/medium/high "
                            f"(got: {action[field]})"
                        )

    return errors
