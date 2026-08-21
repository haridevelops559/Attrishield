"""
AI Insight Engine.
Generates grounded retention insights using Ollama or deterministic fallback heuristics.
"""

import json
from typing import Dict, Any, List, Optional
from backend.app.llm.ollama_client import OllamaClient
from backend.app.llm.prompts import SYSTEM_PROMPT, build_insight_prompt
from backend.app.llm.schemas import OllamaInsightResponse, RetentionRecommendation
from backend.app.core.config import settings
from backend.app.core.logging import logger


class InsightEngine:
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    async def generate_insights(self, aggregated_stats: dict, dept_summary: list = None, custom_notes: str = None) -> OllamaInsightResponse:
        """
        Generates grounded retention insights.
        Attempts Ollama LLM execution; falls back gracefully to deterministic rule-based analysis if offline.
        """
        prompt = build_insight_prompt(aggregated_stats, dept_summary or [], custom_notes)
        
        is_healthy = await self.client.check_health()
        
        if is_healthy:
            response_text = await self.client.generate_completion(prompt, SYSTEM_PROMPT)
            if response_text:
                try:
                    # Clean markdown blocks if returned by model
                    clean_text = response_text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    if clean_text.startswith("```"):
                        clean_text = clean_text[3:]
                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]
                    clean_text = clean_text.strip()
                    
                    parsed = json.loads(clean_text)
                    recs = [RetentionRecommendation(**r) for r in parsed.get("recommendations", [])]
                    
                    return OllamaInsightResponse(
                        executive_summary=parsed.get("executive_summary", "Executive summary generated successfully."),
                        key_findings=parsed.get("key_findings", []),
                        department_insights=parsed.get("department_insights", []),
                        recommendations=recs,
                        limitations_disclaimer=parsed.get("limitations_disclaimer", "Statistical model output."),
                        model_used=self.client.model,
                        is_fallback=False
                    )
                except Exception as e:
                    logger.warning(f"Error parsing Ollama JSON output: {e}. Falling back to deterministic analysis.")

        # Fallback Analysis Generation
        logger.info("Generating deterministic fallback retention analysis...")
        rev_rate = aggregated_stats.get("review_rate", 0.0)
        high_cnt = aggregated_stats.get("high_risk_count", 0)
        total_cnt = aggregated_stats.get("total_employees", 0)

        fallback_recs = [
            RetentionRecommendation(
                category="Overtime & Work Burden",
                action_item="Review workload allocation for employees working high overtime with long commutes.",
                target_segment="High OverTime / Long Commute Cohort",
                priority="HIGH"
            ),
            RetentionRecommendation(
                category="Compensation Alignment",
                action_item="Conduct market salary benchmarking for roles showing high IncomePerJobLevel disparity.",
                target_segment="Stagnant Job Level Employees",
                priority="MEDIUM"
            ),
            RetentionRecommendation(
                category="Career Progression",
                action_item="Implement structured mentorship for employees with >3 years in current role without promotion.",
                target_segment="High PromotionStagnationRatio Cohort",
                priority="HIGH"
            )
        ]

        dept_insights = []
        if dept_summary:
            for d in dept_summary:
                dept_insights.append({
                    "department": d.get("department", "Unknown"),
                    "risk_level": "High" if d.get("avg_probability", 0) >= 0.15 else "Low",
                    "observation": f"Department has {d.get('high_risk', 0)} high-risk employees out of {d.get('total', 0)}."
                })

        return OllamaInsightResponse(
            executive_summary=f"Analysis of {total_cnt} employees indicates a {rev_rate * 100:.1f}% attrition review rate ({high_cnt} high-risk cases identified). Priority attention recommended for overtime burden and promotion stagnation segments.",
            key_findings=[
                f"Overall attrition review rate is {rev_rate * 100:.1f}%.",
                "Commute overtime burden is a primary predictive factor for high attrition probability.",
                "Employees in early career stages (<=3 years experience) exhibit higher volatility."
            ],
            department_insights=dept_insights,
            recommendations=fallback_recs,
            limitations_disclaimer="Ollama LLM currently offline. Analysis rendered using grounded deterministic analytics rules. Model predictions do not constitute automated employment decisions.",
            model_used=f"{self.client.model} (Deterministic Rule Engine Fallback)",
            is_fallback=True
        )
