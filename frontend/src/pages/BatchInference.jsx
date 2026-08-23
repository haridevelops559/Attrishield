import { Link, useSearchParams } from 'react-router-dom'
import { useRef } from 'react'
import { useBatches } from '../hooks/useBatches'
import { useCreateBatch } from '../hooks/useCreateBatch'

const PAGE_SIZE = 10

function BatchInference() {
  const fileInputRef = useRef(null)

  const [searchParams, setSearchParams] =
    useSearchParams()

  const page = Math.max(
    1,
    Number(searchParams.get('page')) || 1,
  )

  const {
    data: batches = [],
    isPending,
    isError,
    error,
    isFetching,
  } = useBatches(page, PAGE_SIZE)

  const {
    mutate: createBatch,
    isPending: isCreating,
    isError: isCreateError,
    error: createError,
  } = useCreateBatch()

  function handleSubmit(event) {
    event.preventDefault()

    const file =
      fileInputRef.current?.files?.[0]

    if (!file) {
      return
    }

    createBatch(file)
  }

  function goToPage(nextPage) {
    setSearchParams({
      page: String(nextPage),
    })
  }

  const canGoPrevious = page > 1

  const canGoNext =
    batches.length === PAGE_SIZE

  return (
    <section className="mx-auto max-w-7xl">
      {/* Page Header */}
      <div>
        <p className="text-sm font-medium text-brand-600">
          Batch Inference
        </p>

        <h1 className="mt-1 text-2xl font-bold text-slate-900 md:text-3xl">
          Batch Inference
        </h1>

        <p className="mt-2 max-w-2xl text-sm text-slate-600">
          Upload a CSV employee dataset and run the V3
          attrition inference pipeline.
        </p>
      </div>

      {/* Upload Form */}
      <form
        onSubmit={handleSubmit}
        className="mt-8 rounded-xl border border-slate-200 bg-white p-6"
      >
        <label
          htmlFor="batch-file"
          className="block text-sm font-medium text-slate-700"
        >
          Employee CSV
        </label>

        <input
          ref={fileInputRef}
          id="batch-file"
          type="file"
          accept=".csv"
          className="mt-2 block w-full rounded-lg border border-slate-300 p-2 text-sm"
        />

        <p className="mt-2 text-xs text-slate-500">
          Upload a CSV containing the employee features
          required by the V3 model.
        </p>

        <button
          type="submit"
          disabled={isCreating}
          className="mt-4 rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isCreating
            ? 'Running Batch...'
            : 'Run Batch Inference'}
        </button>

        {isCreateError && (
          <div
            role="alert"
            className="mt-4 rounded-lg bg-red-50 p-4"
          >
            <p className="text-sm font-medium text-red-700">
              Batch creation failed.
            </p>

            <p className="mt-1 text-xs text-red-600">
              {createError?.message ||
                'Unable to create batch inference job.'}
            </p>
          </div>
        )}
      </form>

      {/* Batch History */}
      <section className="mt-8 rounded-xl border border-slate-200 bg-white">
        <div className="flex items-center justify-between border-b border-slate-200 p-6">
          <div>
            <h2 className="font-semibold text-slate-900">
              Recent Batches
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Server-owned batch history.
            </p>
          </div>

          {isFetching && (
            <span className="text-xs text-slate-500">
              Refreshing...
            </span>
          )}
        </div>

        {/* Loading */}
        {isPending && (
          <p className="p-6 text-sm text-slate-500">
            Loading batches...
          </p>
        )}

        {/* Error */}
        {isError && (
          <div className="p-6">
            <p className="text-sm font-medium text-red-700">
              Unable to load batches.
            </p>

            <p className="mt-1 text-xs text-red-600">
              {error?.message ||
                'Unable to retrieve batch history.'}
            </p>
          </div>
        )}

        {/* Empty State */}
        {!isPending &&
          !isError &&
          batches.length === 0 && (
            <div className="p-6">
              <p className="text-sm font-medium text-slate-700">
                No batch jobs found.
              </p>

              <p className="mt-1 text-xs text-slate-500">
                Upload a CSV above to create your first
                batch inference job.
              </p>
            </div>
          )}

        {/* Batch Table */}
        {!isPending &&
          !isError &&
          batches.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="border-b border-slate-200 bg-slate-50">
                  <tr>
                    <th className="px-6 py-3 font-medium text-slate-600">
                      Batch
                    </th>

                    <th className="px-6 py-3 font-medium text-slate-600">
                      File
                    </th>

                    <th className="px-6 py-3 font-medium text-slate-600">
                      Status
                    </th>

                    <th className="px-6 py-3 font-medium text-slate-600">
                      Rows
                    </th>

                    <th className="px-6 py-3 font-medium text-slate-600">
                      High Risk
                    </th>

                    <th className="px-6 py-3 font-medium text-slate-600">
                      Action
                    </th>
                  </tr>
                </thead>

                <tbody>
                  {batches.map((batch) => (
                    <tr
                      key={batch.batch_id}
                      className="border-b border-slate-100 last:border-0 hover:bg-slate-50"
                    >
                      {/* Batch ID */}
                      <td className="px-6 py-4 font-medium">
                        <Link
                          to={`/batches/${batch.batch_id}`}
                          className="font-mono text-xs text-brand-600 hover:underline"
                        >
                          {batch.batch_id}
                        </Link>
                      </td>

                      {/* Filename */}
                      <td className="px-6 py-4 text-slate-600">
                        {batch.filename}
                      </td>

                      {/* Status */}
                      <td className="px-6 py-4">
                        <StatusBadge
                          status={batch.status}
                        />
                      </td>

                      {/* Row Count */}
                      <td className="px-6 py-4 text-slate-600">
                        {batch.row_count ?? '—'}
                      </td>

                      {/* High Risk */}
                      <td className="px-6 py-4 text-slate-600">
                        {batch.high_risk_count ?? 0}
                      </td>

                      {/* Action */}
                      <td className="px-6 py-4">
                        <Link
                          to={`/batches/${batch.batch_id}`}
                          className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100"
                        >
                          View Results
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

        {/* Pagination */}
        {!isPending &&
          !isError && (
            <div className="flex items-center justify-between border-t border-slate-200 p-4">
              <div>
                <p className="text-sm text-slate-500">
                  Page {page}
                </p>

                <p className="mt-1 text-xs text-slate-400">
                  Showing up to {PAGE_SIZE} batches.
                </p>
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={
                    !canGoPrevious ||
                    isFetching
                  }
                  onClick={() =>
                    goToPage(page - 1)
                  }
                  className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Previous
                </button>

                <button
                  type="button"
                  disabled={
                    !canGoNext ||
                    isFetching
                  }
                  onClick={() =>
                    goToPage(page + 1)
                  }
                  className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            </div>
          )}
      </section>
    </section>
  )
}

function StatusBadge({ status }) {
  const normalizedStatus =
    String(status || '').toLowerCase()

  let className =
    'bg-slate-100 text-slate-700'

  if (
    normalizedStatus === 'completed' ||
    normalizedStatus === 'success'
  ) {
    className =
      'bg-emerald-100 text-emerald-700'
  }

  if (
    normalizedStatus === 'running' ||
    normalizedStatus === 'processing' ||
    normalizedStatus === 'pending'
  ) {
    className =
      'bg-amber-100 text-amber-700'
  }

  if (
    normalizedStatus === 'failed' ||
    normalizedStatus === 'error'
  ) {
    className =
      'bg-red-100 text-red-700'
  }

  return (
    <span
      className={`rounded-full px-2.5 py-1 text-xs font-semibold ${className}`}
    >
      {status || 'Unknown'}
    </span>
  )
}

export default BatchInference