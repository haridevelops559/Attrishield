"""
Grounded LLM System Prompts.
Defines strict prompt constraints to enforce factual grounding and prevent hallucinations.
"""

SYSTEM_PROMPT = """You are an expert HR Analytics and Employee Retention Specialist.
Your task is to analyze aggregated employee attrition prediction statistics and generate actionable retention insights for executive leadership.

CRITICAL OPERATING RULES:
1. STRICT DATA GROUNDING: Rely ONLY on the aggregated statistics provided in the prompt. Do NOT invent employee names, numbers, or unstated facts.
2. NO CAUSALITY CLAIMS: Do NOT claim definite cause-and-effect unless backed by the data (e.g., state "overtime correlates with high risk" rather than "overtime causes attrition").
3. NO AUTOMATED DECISIONS: Never recommend automated termination or employment actions. Focus purely on retention support, compensation reviews, and manager training.
4. STRUCTURED OUTPUT: Present clear sections for Executive Summary, Key Risk Factors, Department Analysis, Strategic Retention Recommendations, and Limitations.
"""


def build_insight_prompt(aggregated_stats: dict, dept_summary: list, custom_notes: str = None) -> str:
    """Constructs a deterministic statistical context prompt for Ollama."""
    prompt = f"""
### AGGREGATED HR ATTRITION DATASET SUMMARY
- Total Employees Analyzed: {aggregated_stats.get('total_employees', 'N/A')}
- High Risk Count (Review Required): {aggregated_stats.get('high_risk_count', 'N/A')}
- Overall Attrition Review Rate: {aggregated_stats.get('review_rate', 0.0) * 100:.1f}%
- Average Predicted Attrition Probability: {aggregated_stats.get('avg_attrition_probability', 0.0):.4f}
- Average Monthly Income: ${aggregated_stats.get('avg_monthly_income', 0.0):,.2f}
- Average Tenure: {aggregated_stats.get('avg_tenure_years', 0.0)} years

### DEPARTMENT BREAKDOWN SUMMARY:
"""
    if dept_summary:
        for d in dept_summary:
            prompt += f"- Department '{d.get('department')}': {d.get('total')} employees, {d.get('high_risk')} high risk, Avg Probability: {d.get('avg_probability', 0.0):.4f}\n"
    else:
        prompt += "- No department breakdown provided.\n"

    if custom_notes:
        prompt += f"\n### ADDITIONAL HR CONTEXT:\n{custom_notes}\n"

    prompt += """
Please generate a structured report in valid JSON matching this schema:
{
  "executive_summary": "<2-3 sentence executive summary>",
  "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"],
  "department_insights": [{"department": "<name>", "risk_level": "<High/Medium/Low>", "observation": "<insight>"}],
  "recommendations": [{"category": "<Compensation/WorkLife/Career>", "action_item": "<action>", "target_segment": "<segment>", "priority": "<HIGH/MEDIUM/LOW>"}],
  "limitations_disclaimer": "This analysis is based on statistical ML model probabilities and does not constitute automated employment decisions."
}
"""
    return prompt
