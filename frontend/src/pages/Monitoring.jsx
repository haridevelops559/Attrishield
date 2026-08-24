import { useMonitoring } from '../hooks/useMonitoring'

function Monitoring() {
  const {
    data,
    isPending,
    isError,
    error,
    isFetching,
  } = useMonitoring()

  if (isPending) {
    return (
      <section className="mx-auto max-w-7xl">
        <MonitoringHeader />

        <div className="mt-8 rounded-xl border border-slate-200 bg-white p-6">
          <p className="text-sm text-slate-500">
            Loading monitoring metrics...
          </p>
        </div>
      </section>
    )
  }

  if (isError) {
    return (
      <section className="mx-auto max-w-7xl">
        <MonitoringHeader />

        <div
          role="alert"
          className="mt-8 rounded-xl border border-red-200 bg-red-50 p-6"
        >
          <p className="font-semibold text-red-700">
            Unable to load monitoring metrics.
          </p>

          <p className="mt-2 text-sm text-red-600">
            {error?.message ||
              'Monitoring request failed.'}
          </p>
        </div>
      </section>
    )
  }

  const predictionSummary =
    data?.prediction_summary ?? {}

  return (
    <section className="mx-auto max-w-7xl">
      <div className="flex items-start justify-between gap-4">
        <MonitoringHeader />

        {isFetching && (
          <span className="mt-1 text-xs text-slate-400">
            Refreshing...
          </span>
        )}
      </div>

      <section className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Total Predictions"
          value={
            predictionSummary.total_predictions ?? 0
          }
        />

        <MetricCard
          label="High Risk"
          value={
            predictionSummary.high_risk_count ?? 0
          }
        />

        <MetricCard
          label="Review Rate"
          value={formatPercent(
            predictionSummary.review_rate,
          )}
        />

        <MetricCard
          label="Average Risk"
          value={formatPercent(
            predictionSummary.average_attrition_probability,
          )}
        />
      </section>

      <section className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Average Latency"
          value={`${Number(
            predictionSummary.average_latency_ms ?? 0,
          ).toFixed(2)} ms`}
        />

        <MetricCard
          label="Batch Jobs"
          value={data?.batch_count ?? 0}
        />

        <MetricCard
          label="Selected Threshold"
          value={formatPercent(
            data?.selected_threshold,
          )}
        />

        <MetricCard
          label="Brier Score"
          value={Number(
            data?.brier_score ?? 0,
          ).toFixed(4)}
        />
      </section>

      <section className="mt-8 grid gap-6 lg:grid-cols-2">
        <InfoCard title="Active Model">
          <InfoRow
            label="Model Version"
            value={
              data?.active_model_version ??
              'Unknown'
            }
          />

          <InfoRow
            label="Feature Version"
            value={
              data?.active_feature_version ??
              'Unknown'
            }
          />

          <InfoRow
            label="Selected Threshold"
            value={formatPercent(
              data?.selected_threshold,
            )}
          />
        </InfoCard>

        <InfoCard title="Model Performance">
          <InfoRow
            label="Cross-Validation ROC-AUC"
            value={Number(
              data?.cv_roc_auc ?? 0,
            ).toFixed(4)}
          />

          <InfoRow
            label="Test ROC-AUC"
            value={Number(
              data?.test_roc_auc ?? 0,
            ).toFixed(4)}
          />

          <InfoRow
            label="Brier Score"
            value={Number(
              data?.brier_score ?? 0,
            ).toFixed(4)}
          />
        </InfoCard>
      </section>

      <section className="mt-8 rounded-xl border border-slate-200 bg-white p-6">
        <div>
          <h2 className="font-semibold text-slate-900">
            Monitoring Overview
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            Operational health and prediction activity
            from the current inference system.
          </p>
        </div>

        <div className="mt-6 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <StatusItem
            label="Inference Service"
            value="Active"
          />

          <StatusItem
            label="Model"
            value="Loaded"
          />

          <StatusItem
            label="Prediction Store"
            value="Connected"
          />

          <StatusItem
            label="Analytics Data"
            value={
              predictionSummary.total_predictions > 0
                ? 'Available'
                : 'No Data'
            }
          />
        </div>
      </section>
    </section>
  )
}

function MonitoringHeader() {
  return (
    <div>
      <p className="text-sm font-medium text-brand-600">
        AttriShield HR
      </p>

      <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">
        Prediction Monitoring
      </h1>

      <p className="mt-2 max-w-3xl text-sm text-slate-600">
        Monitor prediction volume, risk levels, inference
        latency, model configuration, and model performance.
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

function InfoCard({ title, children }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6">
      <h2 className="font-semibold text-slate-900">
        {title}
      </h2>

      <div className="mt-5 divide-y divide-slate-100">
        {children}
      </div>
    </section>
  )
}

function InfoRow({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-4 py-3">
      <span className="text-sm text-slate-500">
        {label}
      </span>

      <span className="text-right text-sm font-medium text-slate-900">
        {value}
      </span>
    </div>
  )
}

function StatusItem({ label, value }) {
  return (
    <div className="rounded-lg bg-slate-50 p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </p>

      <p className="mt-2 text-sm font-semibold text-slate-900">
        {value}
      </p>
    </div>
  )
}

function formatPercent(value) {
  return `${(
    Number(value ?? 0) * 100
  ).toFixed(2)}%`
}

export default Monitoring