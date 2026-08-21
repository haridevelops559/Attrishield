import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
} from 'react-router-dom'

import AppShell from './components/layout/AppShell'
import Dashboard from './pages/Dashboard'
import IndividualPrediction from './pages/IndividualPrediction'
import BatchInference from './pages/BatchInference'

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route
            path="/dashboard"
            element={<Dashboard />}
          />

          <Route
            path="/predict"
            element={<IndividualPrediction />}
          />

          <Route
            path="/batches"
            element={<BatchInference />}
          />

          <Route
            path="/analytics"
            element={
              <PlaceholderPage title="Analytics" />
            }
          />

          <Route
            path="/features"
            element={
              <PlaceholderPage title="Feature Store" />
            }
          />

          <Route
            path="/insights"
            element={
              <PlaceholderPage title="AI Insights" />
            }
          />

          <Route
            path="/monitoring"
            element={
              <PlaceholderPage title="Monitoring" />
            }
          />

          <Route
            path="/"
            element={
              <Navigate
                to="/dashboard"
                replace
              />
            }
          />

          <Route
            path="*"
            element={
              <Navigate
                to="/dashboard"
                replace
              />
            }
          />
        </Routes>
      </AppShell>
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