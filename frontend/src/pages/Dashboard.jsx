import { useQuery } from '@tanstack/react-query'
import {
  getAnalyticsCharts,
  runAnalyticsQuery,
} from '../services/api'

function Dashboard() {
  const chartsQuery = useQuery({
    queryKey: ['dashboard', 'charts'],
    queryFn: () => getAnalyticsCharts(),
    staleTime: 30 * 1000,
  })

  const summaryQuery = useQuery({
    queryKey: ['dashboard', 'summary'],
    queryFn: () => runAnalyticsQuery({}),
    staleTime: 30 * 1000,
  })

  const charts = chartsQuery.data ?? {}

  const kpis =
    summaryQuery.data?.summary_kpis ?? {}

  const departmentRisk =
    charts.department_risk ?? []

  const riskDistribution =
    charts.risk_distribution ?? []

  const overtimeImpact =
    charts.overtime_impact ?? []

  const isLoading =
    chartsQuery.isPending ||
    summaryQuery.isPending

  const isError =
    chartsQuery.isError ||
    summaryQuery.isError

  if (isLoading) {
    return (
      <section className="mx-auto max-w-7xl">
        <DashboardHeader />

        <div className="mt-8 rounded-xl border border-slate-200 bg-white p-6">
          <p className="text-sm text-slate-500">
            Loading workforce intelligence...
          </p>
        </div>
      </section>
    )
  }

  if (isError) {
    const message =
      chartsQuery.error?.message ||
      summaryQuery.error?.message ||
      'Unable to load dashboard data.'

    return (
      <section className="mx-auto max-w-7xl">
        <DashboardHeader />

        <div
          role="alert"
          className="mt-8 rounded-xl border border-red-200 bg-red-50 p-6"
        >
          <p className="font-semibold text-red-700">
            Unable to load workforce dashboard.
          </p>

          <p className="mt-2 text-sm text-red-600">
            {message}
          </p>
        </div>
      </section>
    )
  }

  return (
    <section className="mx-auto max-w-7xl">
      <DashboardHeader />

      {/* Workforce Pulse */}
      <section className="mt-8">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Employees Analyzed"
            value={kpis.total_employees ?? 0}
          />

          <MetricCard
            label="High Risk Employees"
            value={kpis.high_risk_count ?? 0}
          />

          <MetricCard
            label="Review Rate"
            value={formatPercent(
              kpis.review_rate,
            )}
          />

          <MetricCard
            label="Average Attrition Risk"
            value={formatPercent(
              kpis.avg_attrition_probability,
            )}
          />
        </div>
      </section>

      {/* Risk + Department */}
      <section className="mt-8 grid gap-6 lg:grid-cols-2">
        <RiskDistribution
          data={riskDistribution}
        />

        <DepartmentRisk
          data={departmentRisk}
        />
      </section>

      {/* Overtime signal */}
      <section className="mt-6">
        <OvertimeImpact
          data={overtimeImpact}
        />
      </section>

      {/* Workforce signals */}
      <section className="mt-6 grid gap-6 lg:grid-cols-2">
        <WorkforceSignals
          kpis={kpis}
        />

        <PriorityActions
          departmentRisk={departmentRisk}
          overtimeImpact={overtimeImpact}
          highRiskCount={
            kpis.high_risk_count ?? 0
          }
        />
      </section>
    </section>
  )
}

function DashboardHeader() {
  return (
    <div>
      <p className="text-sm font-medium text-brand-600">
        Executive Overview
      </p>

      <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">
        Workforce Risk Dashboard
      </h1>

      <p className="mt-2 max-w-3xl text-sm text-slate-600">
        Identify workforce risk hotspots, retention signals,
        and priority areas requiring HR attention.
      </p>
    </div>
  )
}

function MetricCard({ label, value }) {
  return (
    <article className="rounded-xl border border-slate-200 bg-white p-5">
      <p className="text-sm text-slate-500">
        {label}
      </p>

      <p className="mt-2 text-2xl font-bold text-slate-900">
        {value}
      </p>
    </article>
  )
}

function RiskDistribution({ data }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6">
      <div>
        <h2 className="font-semibold text-slate-900">
          Workforce Risk Distribution
        </h2>

        <p className="mt-1 text-sm text-slate-500">
          Current employee population grouped by predicted
          attrition risk.
        </p>
      </div>

      <div className="mt-6 space-y-4">
        {data.length === 0 && (
          <EmptyState text="No risk distribution data available." />
        )}

        {data.map((item) => (
          <div
            key={item.category}
            className="flex items-center justify-between gap-4"
          >
            <span className="text-sm text-slate-600">
              {item.category}
            </span>

            <span className="rounded-full bg-slate-100 px-3 py-1 text-sm font-semibold text-slate-900">
              {item.count}
            </span>
          </div>
        ))}
      </div>
    </section>
  )
}

function DepartmentRisk({ data }) {
  const sorted = [...data].sort(
    (a, b) =>
      b.avg_probability - a.avg_probability,
  )

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6">
      <div>
        <h2 className="font-semibold text-slate-900">
          Department Risk Hotspots
        </h2>

        <p className="mt-1 text-sm text-slate-500">
          Departments with higher predicted attrition
          exposure.
        </p>
      </div>

      <div className="mt-6 space-y-5">
        {sorted.length === 0 && (
          <EmptyState text="No department data available." />
        )}

        {sorted.map((department) => (
          <div key={department.department}>
            <div className="flex items-center justify-between gap-4">
              <span className="text-sm font-medium text-slate-700">
                {department.department}
              </span>

              <span className="text-sm font-semibold text-slate-900">
                {formatPercent(
                  department.avg_probability,
                )}
              </span>
            </div>

            <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-brand-500"
                style={{
                  width: `${Math.min(
                    department.avg_probability * 100,
                    100,
                  )}%`,
                }}
              />
            </div>

            <div className="mt-1 flex justify-between text-xs text-slate-400">
              <span>
                {department.total} employees
              </span>

              <span>
                {department.high_risk} high risk
              </span>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

function OvertimeImpact({ data }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6">
      <div>
        <h2 className="font-semibold text-slate-900">
          Overtime Risk Signal
        </h2>

        <p className="mt-1 text-sm text-slate-500">
          Compare predicted attrition probability across
          overtime exposure groups.
        </p>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {data.length === 0 && (
          <EmptyState text="No overtime analysis available." />
        )}

        {data.map((item) => (
          <div
            key={item.overtime_status}
            className="rounded-lg bg-slate-50 p-5"
          >
            <p className="text-sm text-slate-500">
              {item.overtime_status}
            </p>

            <p className="mt-2 text-2xl font-bold text-slate-900">
              {formatPercent(
                item.avg_probability,
              )}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              {item.employee_count} employees
            </p>
          </div>
        ))}
      </div>
    </section>
  )
}

function WorkforceSignals({ kpis }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6">
      <h2 className="font-semibold text-slate-900">
        Workforce Signals
      </h2>

      <p className="mt-1 text-sm text-slate-500">
        Supporting workforce characteristics from the
        current prediction population.
      </p>

      <div className="mt-6 space-y-4">
        <SignalRow
          label="Average Monthly Income"
          value={formatNumber(
            kpis.avg_monthly_income,
          )}
        />

        <SignalRow
          label="Average Tenure"
          value={`${Number(
            kpis.avg_tenure_years ?? 0,
          ).toFixed(1)} years`}
        />

        <SignalRow
          label="Low Risk Employees"
          value={kpis.low_risk_count ?? 0}
        />

        <SignalRow
          label="Employees Requiring Review"
          value={kpis.high_risk_count ?? 0}
        />
      </div>
    </section>
  )
}

function PriorityActions({
  departmentRisk,
  overtimeImpact,
  highRiskCount,
}) {
  const topDepartment =
    [...departmentRisk].sort(
      (a, b) =>
        b.avg_probability - a.avg_probability,
    )[0]

  const overtimeGroup =
    [...overtimeImpact].sort(
      (a, b) =>
        b.avg_probability - a.avg_probability,
    )[0]

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6">
      <h2 className="font-semibold text-slate-900">
        Priority Actions
      </h2>

      <p className="mt-1 text-sm text-slate-500">
        Areas HR may want to investigate based on the
        current prediction signals.
      </p>

      <div className="mt-6 space-y-4">
        <ActionItem
          priority="High"
          title={`${highRiskCount} employees require review`}
          description="Review the highest-risk employee predictions before making retention decisions."
        />

        {topDepartment && (
          <ActionItem
            priority="Medium"
            title={`Review ${topDepartment.department}`}
            description={`This department currently has the highest average predicted attrition probability at ${formatPercent(topDepartment.avg_probability)}.`}
          />
        )}

        {overtimeGroup && (
          <ActionItem
            priority="Medium"
            title="Investigate overtime exposure"
            description={`${overtimeGroup.overtime_status} has the higher observed average predicted attrition probability in the current dataset.`}
          />
        )}
      </div>
    </section>
  )
}

function SignalRow({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-slate-100 pb-3">
      <span className="text-sm text-slate-500">
        {label}
      </span>

      <span className="text-sm font-semibold text-slate-900">
        {value}
      </span>
    </div>
  )
}

function ActionItem({
  priority,
  title,
  description,
}) {
  return (
    <div className="rounded-lg bg-slate-50 p-4">
      <div className="flex items-center gap-2">
        <span className="rounded-full bg-slate-200 px-2 py-1 text-xs font-semibold text-slate-700">
          {priority}
        </span>

        <p className="text-sm font-semibold text-slate-900">
          {title}
        </p>
      </div>

      <p className="mt-2 text-sm leading-6 text-slate-600">
        {description}
      </p>
    </div>
  )
}

function EmptyState({ text }) {
  return (
    <p className="text-sm text-slate-500">
      {text}
    </p>
  )
}

function formatPercent(value) {
  return `${(
    Number(value ?? 0) * 100
  ).toFixed(2)}%`
}

function formatNumber(value) {
  return Number(value ?? 0).toLocaleString()
}

export default Dashboard