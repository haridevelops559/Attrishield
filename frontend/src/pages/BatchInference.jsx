import { useRef } from 'react'
import { useBatches } from '../hooks/useBatches'
import { useCreateBatch } from '../hooks/useCreateBatch'

function BatchInference() {
  const fileInputRef = useRef(null)

  const {
    data: batches = [],
    isPending,
    isError,
    error,
    isFetching,
  } = useBatches()

  const {
    mutate: createBatch,
    isPending: isCreating,
    isError: isCreateError,
    error: createError,
  } = useCreateBatch()

  function handleSubmit(event) {
    event.preventDefault()

    const file = fileInputRef.current?.files?.[0]

    if (!file) {
      return
    }

    createBatch(file)
  }

  return (
    <section className="mx-auto max-w-7xl">
      <div>
        <p className="text-sm font-medium text-brand-600">
          Batch Inference
        </p>

        <h1 className="mt-1 text-2xl font-bold text-slate-900">
          Batch Inference
        </h1>

        <p className="mt-2 text-sm text-slate-600">
          Upload a CSV employee dataset and run the V3
          attrition inference pipeline.
        </p>
      </div>

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
          <p
            role="alert"
            className="mt-3 text-sm text-red-600"
          >
            {createError?.message || 'Batch creation failed.'}
          </p>
        )}
      </form>

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

        {isPending && (
          <p className="p-6 text-sm text-slate-500">
            Loading batches...
          </p>
        )}

        {isError && (
          <div className="p-6">
            <p className="text-sm font-medium text-red-700">
              Unable to load batches.
            </p>

            <p className="mt-1 text-xs text-red-600">
              {error?.message}
            </p>
          </div>
        )}

        {!isPending && !isError && batches.length === 0 && (
          <p className="p-6 text-sm text-slate-500">
            No batch jobs found.
          </p>
        )}

        {!isPending && !isError && batches.length > 0 && (
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
                </tr>
              </thead>

              <tbody>
                {batches.map((batch) => (
                  <tr
                    key={batch.batch_id}
                    className="border-b border-slate-100 last:border-0"
                  >
                    <td className="px-6 py-4 font-medium text-slate-900">
                      {batch.batch_id}
                    </td>

                    <td className="px-6 py-4 text-slate-600">
                      {batch.filename}
                    </td>

                    <td className="px-6 py-4">
                      <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
                        {batch.status}
                      </span>
                    </td>

                    <td className="px-6 py-4 text-slate-600">
                      {batch.row_count}
                    </td>

                    <td className="px-6 py-4 text-slate-600">
                      {batch.high_risk_count ?? 0}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </section>
  )
}

export default BatchInference