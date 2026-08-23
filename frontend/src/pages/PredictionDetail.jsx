import { Link, useParams } from 'react-router-dom'
import { usePrediction } from '../hooks/usePrediction'
import PredictionResult from '../components/prediction/PredictionResult'

function PredictionDetail() {
  const { predictionId } = useParams()

  const {
    data: prediction,
    isPending,
    isError,
    error,
    isFetching,
  } = usePrediction(predictionId)

  return (
    <section className="mx-auto max-w-7xl">
      <Link
        to="/predict"
        className="text-sm font-medium text-brand-600 hover:underline"
      >
        ← Back to Prediction
      </Link>

      <div className="mt-6">
        <p className="text-sm font-medium text-brand-600">
          Prediction Detail
        </p>

        <h1 className="mt-1 text-2xl font-bold text-slate-900">
          Prediction Result
        </h1>

        <p className="mt-2 text-sm text-slate-600">
          Prediction ID:{' '}
          <span className="font-mono">
            {predictionId}
          </span>
        </p>
      </div>

      {isFetching && (
        <p className="mt-4 text-xs text-slate-500">
          Refreshing prediction...
        </p>
      )}

      {isPending && (
        <div className="mt-8 rounded-xl border border-slate-200 bg-white p-6">
          Loading prediction...
        </div>
      )}

      {isError && (
        <div className="mt-8 rounded-xl border border-red-200 bg-red-50 p-6">
          <p className="font-medium text-red-700">
            Unable to load prediction.
          </p>

          <p className="mt-1 text-sm text-red-600">
            {error?.message}
          </p>
        </div>
      )}

      {!isPending &&
        !isError &&
        prediction && (
          <PredictionResult
            prediction={prediction}
          />
        )}
    </section>
  )
}

export default PredictionDetail