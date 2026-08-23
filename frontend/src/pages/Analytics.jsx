import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import {
  useAnalyticsCharts,
  useAnalyticsQuery,
} from '../hooks/useAnalytics'

function Analytics() {
  const {
    data: analytics,
    isPending: isAnalyticsPending,
    isError: isAnalyticsError,
    error: analyticsError,
  } = useAnalyticsQuery()

  const {
    data: charts,
    isPending: isChartsPending,
    isError: isChartsError,
    error: chartsError,
  } = useAnalyticsCharts()

  const kpis = analytics?.summary_kpis ?? {}

  const riskDistribution =
    charts?.risk_distribution ?? []

  const departmentRisk =
    charts?.department_risk ?? []

  const overtimeImpact =
    charts?.overtime_impact ?? []

  const isLoading =
    isAnalyticsPending || isChartsPending

  const isError =
    isAnalyticsError || isChartsError

  if (isLoading) {
    return (
      <section className="mx-auto max-w-7xl">
        <AnalyticsHeader />

        <div className="mt-8 rounded-xl border border-slate-200 bg-white p-6">
          <p className="text-sm text-slate-500">
            Loading analytics...
          </p>
        </div>
      </section>
    )
  }

  if (isError) {
    return (
      <section className="mx-auto max-w-7xl">
        <AnalyticsHeader />

        <div className="mt-8 rounded-xl border border-red-200 bg-red-50 p-6">
          <p className="font-semibold text-red-700">
            Unable to load analytics.
          </p>

          <p className="mt-2 text-sm text-red-600">
            {analyticsError?.message ||
              chartsError?.message ||
              'Analytics request failed.'}
          </p>
        </div>
      </section>
    )
  }

  return (
    <section className="mx-auto max-w-7xl">
      <AnalyticsHeader />

      <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Total Employees"
          value={kpis.total_employees ?? 0}
        />

        <MetricCard
          label="High Risk"
          value={kpis.high_risk_count ?? 0}
        />

        <MetricCard
          label="Low Risk"
          value={kpis.low_risk_count ?? 0}
        />

        <MetricCard
          label="Review Rate"
          value={formatPercent(kpis.review_rate)}
        />

        <MetricCard
          label="Avg Attrition Probability"
          value={formatPercent(
            kpis.avg_attrition_probability,
          )}
        />

        <MetricCard
          label="Avg Monthly Income"
          value={formatCurrency(
            kpis.avg_monthly_income,
          )}
        />

        <MetricCard
          label="Avg Tenure"
          value={`${Number(
            kpis.avg_tenure_years ?? 0,
          ).toFixed(1)} years`}
        />

        <MetricCard
          label="Filtered Records"
          value={analytics?.filtered_records ?? 0}
        />
      </section>

      <section className="mt-8 grid gap-6 lg:grid-cols-2">
        <ChartCard
          title="Risk Distribution"
          description="Distribution of employees by model risk category."
        >
          {riskDistribution.length === 0 ? (
            <EmptyChart />
          ) : (
            <ResponsiveContainer
              width="100%"
              height={320}
            >
              <PieChart>
                <Pie
                  data={riskDistribution}
                  dataKey="count"
                  nameKey="category"
                  cx="50%"
                  cy="50%"
                  outerRadius={105}
                  label
                >
                  {riskDistribution.map(
                    (entry, index) => (
                      <Cell
                        key={`risk-${index}`}
                        fill={
                          index === 0
                            ? '#dc2626'
                            : '#10b981'
                        }
                      />
                    ),
                  )}
                </Pie>

                <Tooltip />

                <Legend />
              </PieChart>
            </ResponsiveContainer>
          )}
        </ChartCard>

        <ChartCard
          title="Department Risk"
          description="Average attrition probability and high-risk employees by department."
        >
          {departmentRisk.length === 0 ? (
            <EmptyChart />
          ) : (
            <ResponsiveContainer
              width="100%"
              height={320}
            >
              <BarChart
                data={departmentRisk}
                margin={{
                  top: 10,
                  right: 20,
                  left: 0,
                  bottom: 45,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />

                <XAxis
                  dataKey="department"
                  angle={-20}
                  textAnchor="end"
                  interval={0}
                />

                <YAxis />

                <Tooltip />

                <Legend />

                <Bar
                  dataKey="high_risk"
                  name="High Risk"
                  fill="#dc2626"
                  radius={[4, 4, 0, 0]}
                />

                <Bar
                  dataKey="total"
                  name="Total"
                  fill="#64748b"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartCard>
      </section>

      <section className="mt-6">
        <ChartCard
          title="Overtime Impact"
          description="Average predicted attrition probability for overtime and non-overtime employees."
        >
          {overtimeImpact.length === 0 ? (
            <EmptyChart />
          ) : (
            <ResponsiveContainer
              width="100%"
              height={320}
            >
              <BarChart
                data={overtimeImpact}
                margin={{
                  top: 10,
                  right: 20,
                  left: 0,
                  bottom: 20,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />

                <XAxis
                  dataKey="overtime_status"
                />

                <YAxis
                  tickFormatter={(value) =>
                    `${(
                      Number(value) * 100
                    ).toFixed(0)}%`
                  }
                />

                <Tooltip
                  formatter={(value) =>
                    `${(
                      Number(value) * 100
                    ).toFixed(2)}%`
                  }
                />

                <Legend />

                <Bar
                  dataKey="avg_probability"
                  name="Avg Attrition Probability"
                  fill="#4f46e5"
                  radius={[6, 6, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartCard>
      </section>

      <section className="mt-6 rounded-xl border border-slate-200 bg-white">
        <div className="border-b border-slate-200 p-6">
          <h2 className="font-semibold text-slate-900">
            Analytics Summary
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            Current prediction population used by the
            analytics engine.
          </p>
        </div>

        <div className="grid gap-6 p-6 sm:grid-cols-3">
          <SummaryItem
            label="Total Records"
            value={analytics?.total_records ?? 0}
          />

          <SummaryItem
            label="Filtered Records"
            value={analytics?.filtered_records ?? 0}
          />

          <SummaryItem
            label="Average Risk"
            value={formatPercent(
              kpis.avg_attrition_probability,
            )}
          />
        </div>
      </section>
    </section>
  )
}

function AnalyticsHeader() {
  return (
    <div>
      <p className="text-sm font-medium text-brand-600">
        AttriShield HR
      </p>

      <h1 className="mt-1 text-2xl font-bold text-slate-900">
        Workforce Analytics
      </h1>

      <p className="mt-2 max-w-3xl text-sm text-slate-600">
        Analyze employee attrition risk, prediction
        distributions, department exposure, and
        overtime-related risk patterns.
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

function ChartCard({
  title,
  description,
  children,
}) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6">
      <div>
        <h2 className="font-semibold text-slate-900">
          {title}
        </h2>

        <p className="mt-1 text-sm text-slate-500">
          {description}
        </p>
      </div>

      <div className="mt-6">
        {children}
      </div>
    </section>
  )
}

function SummaryItem({ label, value }) {
  return (
    <div>
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </p>

      <p className="mt-1 text-lg font-semibold text-slate-900">
        {value}
      </p>
    </div>
  )
}

function EmptyChart() {
  return (
    <div className="flex h-80 items-center justify-center rounded-lg bg-slate-50">
      <p className="text-sm text-slate-500">
        No analytics data available.
      </p>
    </div>
  )
}

function formatPercent(value) {
  return `${(
    Number(value ?? 0) * 100
  ).toFixed(2)}%`
}

function formatCurrency(value) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(Number(value ?? 0))
}

export default Analytics