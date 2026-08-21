import { useModelInfo } from '../hooks/useModelInfo'

function Dashboard() {
  const {
    data: modelInfo,
    isPending,
    isError,
    error,
    isFetching,
  } = useModelInfo()

  return (
    <section className="mx-auto max-w-7xl">
      <div>
        <p className="text-sm font-medium text-brand-600">
          Overview
        </p>

        <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">
          Workforce Risk Dashboard
        </h1>

        <p className="mt-2 max-w-2xl text-sm text-slate-600">
          Monitor employee attrition risk, model activity, and
          workforce signals from batch inference.
        </p>
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Employees" value="—" />
        <MetricCard label="At Risk" value="—" />
        <MetricCard label="High Risk" value="—" />
        <MetricCard label="Average Risk" value="—" />
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-3">
        <section className="rounded-xl border border-slate-200 bg-white p-6 lg:col-span-2">
          <h2 className="font-semibold text-slate-900">
            Attrition Risk by Department
          </h2>

          <div className="mt-6 flex h-64 items-center justify-center rounded-lg bg-slate-50">
            <p className="text-sm text-slate-500">
              Analytics visualization will appear here.
            </p>
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-slate-900">
              Model Status
            </h2>

            {isFetching && !isPending && (
              <span className="text-xs text-slate-400">
                Refreshing...
              </span>
            )}
          </div>

          <div className="mt-6">
            {isPending && (
              <p className="text-sm text-slate-500">
                Loading model information...
              </p>
            )}

            {isError && (
              <div
                role="alert"
                className="rounded-lg bg-red-50 p-4"
              >
                <p className="text-sm font-medium text-red-700">
                  Unable to load model information.
                </p>

                <p className="mt-1 text-xs text-red-600">
                  {error.message}
                </p>
              </div>
            )}

            {modelInfo && (
              <pre className="max-h-80 overflow-auto rounded-lg bg-slate-50 p-4 text-xs text-slate-700">
                {JSON.stringify(modelInfo, null, 2)}
              </pre>
            )}
          </div>
        </section>
      </div>
    </section>
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

export default Dashboard