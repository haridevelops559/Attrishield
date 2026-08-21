import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../../context/AuthContext'

function AppShell() {
  const { auth, logout } = useAuth()

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="flex min-h-screen">
        <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-white md:block">
          <div className="flex h-16 items-center border-b border-slate-200 px-6">
            <div>
              <p className="text-lg font-bold text-brand-700">
                AttriShield
              </p>

              <p className="text-xs text-slate-500">
                HR Intelligence
              </p>
            </div>
          </div>

          <nav
            className="p-4"
            aria-label="Main navigation"
          >
            <p className="px-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
              Workspace
            </p>

            <div className="mt-3 space-y-1">
              <NavItem to="/dashboard">
                Dashboard
              </NavItem>

              <NavItem to="/predict">
                Individual Prediction
              </NavItem>

              <NavItem to="/batches">
                Batch Inference
              </NavItem>

              <NavItem to="/analytics">
                Analytics
              </NavItem>

              <NavItem to="/features">
                Feature Store
              </NavItem>

              <NavItem to="/insights">
                AI Insights
              </NavItem>

              <NavItem to="/monitoring">
                Monitoring
              </NavItem>
            </div>
          </nav>
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="flex min-h-16 items-center justify-between border-b border-slate-200 bg-white px-4 py-3 md:px-8">
            <div>
              <p className="text-sm font-medium text-slate-900">
                Employee Attrition Intelligence
              </p>

              <p className="hidden text-xs text-slate-500 sm:block">
                ML-powered workforce risk analysis
              </p>
            </div>

            <div className="flex items-center gap-3">
              <div className="hidden text-right sm:block">
                <p className="text-sm font-medium text-slate-800">
                  {auth?.userEmail ?? 'HR Analyst'}
                </p>

                <p className="text-xs text-slate-500">
                  {auth?.userRole ?? 'HR Analyst'}
                </p>
              </div>

              <div
                aria-label="Current user profile"
                className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-100 text-sm font-semibold text-brand-700"
              >
                HA
              </div>

              <button
                type="button"
                onClick={logout}
                className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 hover:text-slate-900"
              >
                Logout
              </button>
            </div>
          </header>

          <main className="flex-1 p-4 md:p-8">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  )
}

function NavItem({ to, children }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `block w-full rounded-lg px-3 py-2 text-sm font-medium transition ${
          isActive
            ? 'bg-brand-50 text-brand-700'
            : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
        }`
      }
    >
      {children}
    </NavLink>
  )
}

export default AppShell