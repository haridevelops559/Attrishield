import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  getAnalyticsCharts,
  runAnalyticsQuery,
  generateAIInsights,
} from '../services/api'

function AIInsights() {
  const [insights, setInsights] = useState(null)

  const mutation = useMutation({
    mutationFn: async () => {
      const [analytics, charts] = await Promise.all([
        runAnalyticsQuery({}),
        getAnalyticsCharts(),
      ])

      const kpis =
        analytics?.summary_kpis ?? {}

      return generateAIInsights({
        aggregated_statistics: {
          total_employees:
            kpis.total_employees ?? 0,
          high_risk_count:
            kpis.high_risk_count ?? 0,
          review_rate:
            kpis.review_rate ?? 0,
          avg_attrition_probability:
            kpis.avg_attrition_probability ?? 0,
          avg_monthly_income:
            kpis.avg_monthly_income ?? 0,
          avg_tenure_years:
            kpis.avg_tenure_years ?? 0,
        },

        department_summary:
          charts?.department_risk ?? [],

        custom_prompt_notes:
          'Provide concise, evidence-grounded retention insights for HR leadership. Do not make automated employment decisions.',
      })
    },

    onSuccess: (data) => {
      setInsights(data)
    },
  })

  return (
    <section className="mx-auto max-w-7xl">
      <div>
        <p className="text-sm font-medium text-brand-600">
          AI Workforce Intelligence
        </p>

        <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">
          AI Insights
        </h1>

        <p className="mt-2 max-w-3xl text-sm text-slate-600">
          Generate evidence-grounded workforce insights from
          the current attrition prediction and analytics data.
        </p>
      </div>

      <section className="mt-8 rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="font-semibold text-slate-900">
              Workforce Insight Generation
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Analytics evidence is passed to the backend
              InsightEngine and local Ollama model.
            </p>
          </div>

          <button
            type="button"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            {mutation.isPending
              ? 'Generating...'
              : 'Generate Insights'}
          </button>
        </div>

        {mutation.isError && (
          <div
            role="alert"
            className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4"
          >
            <p className="text-sm font-semibold text-red-700">
              Unable to generate AI insights.
            </p>

            <p className="mt-1 text-sm text-red-600">
              {mutation.error?.message ||
                'The insights service returned an error.'}
            </p>
          </div>
        )}
      </section>

      {insights && (
        <InsightResults insights={insights} />
      )}

      {!insights &&
        !mutation.isPending &&
        !mutation.isError && (
          <section className="mt-6 rounded-xl border border-slate-200 bg-white p-8 text-center">
            <p className="text-sm text-slate-500">
              Click "Generate Insights" to analyze the
              current workforce prediction data.
            </p>
          </section>
        )}
    </section>
  )
}

function InsightResults({ insights }) {
  return (
    <div className="mt-6 space-y-6">
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-semibold text-slate-900">
            Executive Summary
          </h2>

          <div className="flex gap-2">
            {insights.model_used && (
              <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                {insights.model_used}
              </span>
            )}

            <span
              className={`rounded-full px-3 py-1 text-xs font-semibold ${
                insights.is_fallback
                  ? 'bg-amber-100 text-amber-700'
                  : 'bg-green-100 text-green-700'
              }`}
            >
              {insights.is_fallback
                ? 'Fallback'
                : 'AI Generated'}
            </span>
          </div>
        </div>

        <p className="mt-4 text-sm leading-7 text-slate-700">
          {insights.executive_summary}
        </p>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">
          Key Findings
        </h2>

        <div className="mt-5 space-y-3">
          {insights.key_findings?.map(
            (finding, index) => (
              <div
                key={index}
                className="rounded-lg bg-slate-50 p-4"
              >
                <p className="text-sm leading-6 text-slate-700">
                  {finding}
                </p>
              </div>
            ),
          )}
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">
          Department Insights
        </h2>

        <div className="mt-5 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="px-3 py-3 font-semibold text-slate-600">
                  Department
                </th>

                <th className="px-3 py-3 font-semibold text-slate-600">
                  Risk
                </th>

                <th className="px-3 py-3 font-semibold text-slate-600">
                  Observation
                </th>
              </tr>
            </thead>

            <tbody>
              {insights.department_insights?.map(
                (item, index) => (
                  <tr
                    key={index}
                    className="border-b border-slate-100"
                  >
                    <td className="px-3 py-3 font-medium text-slate-900">
                      {item.department}
                    </td>

                    <td className="px-3 py-3">
                      <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700">
                        {item.risk_level}
                      </span>
                    </td>

                    <td className="px-3 py-3 text-slate-600">
                      {item.observation}
                    </td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold text-slate-900">
          Recommended Actions
        </h2>

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          {insights.recommendations?.map(
            (recommendation, index) => (
              <article
                key={index}
                className="rounded-lg bg-slate-50 p-5"
              >
                <div className="flex items-center justify-between gap-3">
                  <p className="font-semibold text-slate-900">
                    {recommendation.category}
                  </p>

                  <span className="rounded-full bg-slate-200 px-2 py-1 text-xs font-semibold text-slate-700">
                    {recommendation.priority}
                  </span>
                </div>

                <p className="mt-3 text-sm leading-6 text-slate-600">
                  {recommendation.action_item}
                </p>

                {recommendation.target_segment && (
                  <p className="mt-3 text-xs text-slate-500">
                    Target: {recommendation.target_segment}
                  </p>
                )}
              </article>
            ),
          )}
        </div>
      </section>

      {insights.limitations_disclaimer && (
        <section className="rounded-xl border border-amber-200 bg-amber-50 p-5">
          <h2 className="text-sm font-semibold text-amber-800">
            Responsible AI Notice
          </h2>

          <p className="mt-2 text-sm leading-6 text-amber-700">
            {insights.limitations_disclaimer}
          </p>
        </section>
      )}
    </div>
  )
}

export default AIInsights