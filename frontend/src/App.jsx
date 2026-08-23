import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
} from 'react-router-dom'

import AppShell from './components/layout/AppShell'
import ProtectedRoute from './routes/ProtectedRoute'

import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import IndividualPrediction from './pages/IndividualPrediction'
import PredictionDetail from './pages/PredictionDetail'
import BatchInference from './pages/BatchInference'
import BatchResults from './pages/BatchResults'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* =========================
            Public Routes
        ========================== */}

        <Route
          path="/login"
          element={<Login />}
        />

        {/* =========================
            Protected Application
        ========================== */}

        <Route
          path="/"
          element={
            <ProtectedRoute>
              <AppShell />
            </ProtectedRoute>
          }
        >
          {/* / → /dashboard */}
          <Route
            index
            element={
              <Navigate
                to="/dashboard"
                replace
              />
            }
          />

          {/* Dashboard */}
          <Route
            path="dashboard"
            element={<Dashboard />}
          />

          {/* Individual prediction form */}
          <Route
            path="predict"
            element={<IndividualPrediction />}
          />

          {/* Individual prediction detail */}
          <Route
            path="predictions/:predictionId"
            element={<PredictionDetail />}
          />

          {/* Batch inference upload/history */}
          <Route
            path="batches"
            element={<BatchInference />}
          />

          {/* Individual batch results */}
          <Route
            path="batches/:batchId"
            element={<BatchResults />}
          />

          {/* Future modules */}
          <Route
            path="analytics"
            element={
              <PlaceholderPage title="Analytics" />
            }
          />

          <Route
            path="features"
            element={
              <PlaceholderPage title="Feature Store" />
            }
          />

          <Route
            path="insights"
            element={
              <PlaceholderPage title="AI Insights" />
            }
          />

          <Route
            path="monitoring"
            element={
              <PlaceholderPage title="Monitoring" />
            }
          />

          {/* Unknown protected route */}
          <Route
            path="*"
            element={
              <Navigate
                to="/dashboard"
                replace
              />
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

function PlaceholderPage({ title }) {
  return (
    <section>
      <p className="text-sm font-medium text-brand-600">
        AttriShield HR
      </p>

      <h1 className="mt-1 text-2xl font-bold text-slate-900">
        {title}
      </h1>

      <p className="mt-2 text-sm text-slate-600">
        This page will be implemented in a later step.
      </p>
    </section>
  )
}

export default App