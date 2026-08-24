import { useMemo, useState } from 'react'
import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  Database,
  GitBranch,
  Layers3,
  RefreshCw,
  Search,
} from 'lucide-react'

import {
  useFeatureDefinitions,
  useFeatureGroups,
  useFeatureLineage,
  useOnlineFeatures,
  useFeatureMaterializations,
} from '../hooks/useFeatureStore'

import { getPointInTimeFeatures } from '../services/api'

const CANONICAL_FEATURES = [
  'IncomePerJobLevel',
  'PromotionStagnationRatio',
  'ManagerTenureRatio',
  'RoleTenureRatio',
  'OverTimeBinary',
  'CommuteOvertimeBurden',
  'EarlyCareerFlag',
]

function FeatureStore() {
  const [activeTab, setActiveTab] =
    useState('registry')

  const [search, setSearch] =
    useState('')

  const [entityId, setEntityId] =
    useState('emp_1')

  const [featureVersion, setFeatureVersion] =
    useState('v3')

  // Point-in-time retrieval state
  const [pointInTimeEntities, setPointInTimeEntities] =
    useState('emp_1')

  const [pointInTimeFeatures, setPointInTimeFeatures] =
    useState([
      'IncomePerJobLevel',
      'PromotionStagnationRatio',
      'ManagerTenureRatio',
      'RoleTenureRatio',
      'OverTimeBinary',
      'CommuteOvertimeBurden',
      'EarlyCareerFlag',
    ])

  const [pointInTimeTimestamp, setPointInTimeTimestamp] =
    useState('2026-08-24T11:10:00+00:00')

  const [pointInTimeResult, setPointInTimeResult] =
    useState(null)

  const [pointInTimeLoading, setPointInTimeLoading] =
    useState(false)

  const [pointInTimeError, setPointInTimeError] =
    useState(null)

  const definitionsQuery =
    useFeatureDefinitions()

  const groupsQuery =
    useFeatureGroups()

  const lineageQuery =
    useFeatureLineage()

  const materializationsQuery =
    useFeatureMaterializations(20)

  const onlineQuery =
    useOnlineFeatures(
      entityId,
      featureVersion,
    )

  const definitions =
    definitionsQuery.data ?? []

  const groups =
    groupsQuery.data ?? []

  const materializations =
    materializationsQuery.data ?? []

  const filteredDefinitions =
    useMemo(() => {
      const value =
        search.trim().toLowerCase()

      if (!value) {
        return definitions
      }

      return definitions.filter(
        (feature) =>
          feature.feature_name
            ?.toLowerCase()
            .includes(value) ||
          feature.description
            ?.toLowerCase()
            .includes(value) ||
          feature.formula
            ?.toLowerCase()
            .includes(value),
      )
    }, [definitions, search])

  const isLoading =
    definitionsQuery.isPending ||
    groupsQuery.isPending ||
    lineageQuery.isPending ||
    materializationsQuery.isPending

  const hasError =
    definitionsQuery.isError ||
    groupsQuery.isError ||
    lineageQuery.isError ||
    materializationsQuery.isError

  function refreshAll() {
    definitionsQuery.refetch()
    groupsQuery.refetch()
    lineageQuery.refetch()
    materializationsQuery.refetch()
    onlineQuery.refetch()
  }

  async function handlePointInTimeLookup() {
    setPointInTimeLoading(true)
    setPointInTimeError(null)
    setPointInTimeResult(null)

    try {
      const entityIds =
        pointInTimeEntities
          .split(',')
          .map((value) => value.trim())
          .filter(Boolean)

      if (entityIds.length === 0) {
        throw new Error(
          'Enter at least one entity ID.',
        )
      }

      if (pointInTimeFeatures.length === 0) {
        throw new Error(
          'Select at least one feature.',
        )
      }

      if (!pointInTimeTimestamp) {
        throw new Error(
          'Select an as-of timestamp.',
        )
      }

      const payload = {
        entity_ids: entityIds,
        features: pointInTimeFeatures,
        as_of_timestamp:
          pointInTimeTimestamp,
      }

      const result =
        await getPointInTimeFeatures(
          payload,
          featureVersion,
        )

      setPointInTimeResult(result)
    } catch (error) {
      setPointInTimeError(
        error?.message ||
          'Point-in-time retrieval failed.',
      )
    } finally {
      setPointInTimeLoading(false)
    }
  }

  return (
    <section className="mx-auto max-w-7xl">
      <header>
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div>
            <p className="text-sm font-medium text-brand-600">
              ML Infrastructure
            </p>

            <h1 className="mt-1 text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">
              Feature Store
            </h1>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
              Explore registered features, feature
              groups, lineage, materialization jobs,
              version-aware online retrieval, and
              point-in-time feature retrieval.
            </p>
          </div>

          <button
            type="button"
            onClick={refreshAll}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50"
          >
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>
      </header>

      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <SummaryCard
          icon={<Database size={18} />}
          label="Registered Features"
          value={definitions.length}
        />

        <SummaryCard
          icon={<Layers3 size={18} />}
          label="Feature Groups"
          value={groups.length}
        />

        <SummaryCard
          icon={<GitBranch size={18} />}
          label="Materializations"
          value={materializations.length}
        />

        <SummaryCard
          icon={<CheckCircle2 size={18} />}
          label="Feature Version"
          value="v3"
        />
      </div>

      {isLoading && (
        <div className="mt-6 rounded-xl border border-slate-200 bg-white p-5 text-sm text-slate-500">
          Loading feature store metadata...
        </div>
      )}

      {hasError && (
        <div className="mt-6 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-5">
          <AlertCircle
            size={18}
            className="mt-0.5 text-red-600"
          />

          <div>
            <p className="text-sm font-semibold text-red-800">
              Unable to load feature store data.
            </p>

            <p className="mt-1 text-xs text-red-700">
              Check that the backend is running and
              that your authentication session is
              valid.
            </p>
          </div>
        </div>
      )}

      <div className="mt-8 overflow-hidden rounded-xl border border-slate-200 bg-white">
        <nav className="flex overflow-x-auto border-b border-slate-200">
          <TabButton
            active={activeTab === 'registry'}
            onClick={() =>
              setActiveTab('registry')
            }
          >
            Feature Registry
          </TabButton>

          <TabButton
            active={activeTab === 'groups'}
            onClick={() =>
              setActiveTab('groups')
            }
          >
            Feature Groups
          </TabButton>

          <TabButton
            active={activeTab === 'lineage'}
            onClick={() =>
              setActiveTab('lineage')
            }
          >
            Lineage
          </TabButton>

          <TabButton
            active={activeTab === 'online'}
            onClick={() =>
              setActiveTab('online')
            }
          >
            Online Features
          </TabButton>

          <TabButton
            active={
              activeTab === 'point-in-time'
            }
            onClick={() =>
              setActiveTab('point-in-time')
            }
          >
            Point-in-Time
          </TabButton>

          <TabButton
            active={
              activeTab === 'materializations'
            }
            onClick={() =>
              setActiveTab('materializations')
            }
          >
            Materializations
          </TabButton>
        </nav>

        <div className="p-6">
          {activeTab === 'registry' && (
            <RegistryTab
              definitions={
                filteredDefinitions
              }
              search={search}
              setSearch={setSearch}
            />
          )}

          {activeTab === 'groups' && (
            <GroupsTab groups={groups} />
          )}

          {activeTab === 'lineage' && (
            <LineageTab
              lineage={lineageQuery.data}
            />
          )}

          {activeTab === 'online' && (
            <OnlineFeaturesTab
              entityId={entityId}
              setEntityId={setEntityId}
              featureVersion={
                featureVersion
              }
              setFeatureVersion={
                setFeatureVersion
              }
              query={onlineQuery}
            />
          )}

          {activeTab === 'point-in-time' && (
            <PointInTimeTab
              entities={
                pointInTimeEntities
              }
              setEntities={
                setPointInTimeEntities
              }
              features={
                pointInTimeFeatures
              }
              setFeatures={
                setPointInTimeFeatures
              }
              timestamp={
                pointInTimeTimestamp
              }
              setTimestamp={
                setPointInTimeTimestamp
              }
              featureVersion={
                featureVersion
              }
              setFeatureVersion={
                setFeatureVersion
              }
              result={
                pointInTimeResult
              }
              loading={
                pointInTimeLoading
              }
              error={
                pointInTimeError
              }
              onLookup={
                handlePointInTimeLookup
              }
            />
          )}

          {activeTab ===
            'materializations' && (
            <MaterializationsTab
              materializations={
                materializations
              }
            />
          )}
        </div>
      </div>
    </section>
  )
}

function SummaryCard({
  icon,
  label,
  value,
}) {
  return (
    <article className="rounded-xl border border-slate-200 bg-white p-5">
      <div className="flex items-center gap-2 text-slate-500">
        {icon}

        <p className="text-sm">
          {label}
        </p>
      </div>

      <p className="mt-3 text-2xl font-bold text-slate-900">
        {value}
      </p>
    </article>
  )
}

function TabButton({
  active,
  onClick,
  children,
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'whitespace-nowrap border-b-2 px-5 py-4 text-sm font-medium transition',
        active
          ? 'border-brand-600 text-brand-700'
          : 'border-transparent text-slate-500 hover:text-slate-800',
      ].join(' ')}
    >
      {children}
    </button>
  )
}

function RegistryTab({
  definitions,
  search,
  setSearch,
}) {
  return (
    <div>
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="font-semibold text-slate-900">
            Registered Features
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            Canonical features registered for the
            ML feature contract.
          </p>
        </div>

        <div className="relative w-full md:w-80">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
          />

          <input
            value={search}
            onChange={(event) =>
              setSearch(event.target.value)
            }
            placeholder="Search features..."
            className="w-full rounded-lg border border-slate-200 bg-white py-2 pl-9 pr-3 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
          />
        </div>
      </div>

      <div className="mt-6 overflow-x-auto">
        <table className="min-w-full text-left">
          <thead>
            <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
              <th className="px-3 py-3 font-medium">
                Feature
              </th>

              <th className="px-3 py-3 font-medium">
                Type
              </th>

              <th className="px-3 py-3 font-medium">
                Version
              </th>

              <th className="px-3 py-3 font-medium">
                Formula
              </th>
            </tr>
          </thead>

          <tbody>
            {definitions.map(
              (feature) => (
                <tr
                  key={
                    feature.feature_name
                  }
                  className="border-b border-slate-100 last:border-0"
                >
                  <td className="px-3 py-4">
                    <p className="text-sm font-semibold text-slate-900">
                      {
                        feature.feature_name
                      }
                    </p>

                    <p className="mt-1 max-w-md text-xs text-slate-500">
                      {
                        feature.description
                      }
                    </p>
                  </td>

                  <td className="px-3 py-4 text-sm text-slate-600">
                    {feature.data_type}
                  </td>

                  <td className="px-3 py-4">
                    <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700">
                      {
                        feature.feature_version
                      }
                    </span>
                  </td>

                  <td className="px-3 py-4">
                    <code className="text-xs text-slate-600">
                      {feature.formula}
                    </code>
                  </td>
                </tr>
              ),
            )}
          </tbody>
        </table>

        {definitions.length ===
          0 && (
          <EmptyState message="No feature definitions found." />
        )}
      </div>
    </div>
  )
}

function GroupsTab({ groups }) {
  return (
    <div>
      <h2 className="font-semibold text-slate-900">
        Feature Groups
      </h2>

      <p className="mt-1 text-sm text-slate-500">
        Logical collections of features used by
        the model contract.
      </p>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        {groups.map((group) => (
          <article
            key={group.group_name}
            className="rounded-lg border border-slate-200 p-5"
          >
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-slate-100 p-2 text-slate-600">
                <Layers3 size={18} />
              </div>

              <div>
                <h3 className="text-sm font-semibold text-slate-900">
                  {group.group_name}
                </h3>

                <p className="mt-1 text-xs text-slate-500">
                  {group.description}
                </p>
              </div>
            </div>

            <div className="mt-5 flex flex-wrap gap-2">
              {group.features?.map(
                (feature) => (
                  <span
                    key={feature}
                    className="rounded-md bg-slate-100 px-2.5 py-1.5 text-xs font-medium text-slate-700"
                  >
                    {feature}
                  </span>
                ),
              )}
            </div>
          </article>
        ))}
      </div>

      {groups.length === 0 && (
        <EmptyState message="No feature groups found." />
      )}
    </div>
  )
}

function LineageTab({ lineage }) {
  return (
    <div>
      <h2 className="font-semibold text-slate-900">
        Feature Lineage
      </h2>

      <p className="mt-1 text-sm text-slate-500">
        Transformation logic and dependencies
        behind the registered feature contract.
      </p>

      <div className="mt-6 space-y-3">
        {Array.isArray(lineage) &&
          lineage.map((item) => (
            <article
              key={
                item.feature_name
              }
              className="rounded-lg border border-slate-200 p-5"
            >
              <div className="flex flex-wrap items-center gap-3">
                <span className="font-semibold text-slate-900">
                  {item.feature_name}
                </span>

                <span className="text-slate-400">
                  →
                </span>

                <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700">
                  {item.feature_version}
                </span>
              </div>

              <p className="mt-3 text-sm text-slate-600">
                {item.description}
              </p>

              <div className="mt-3 rounded-lg bg-slate-50 p-3">
                <p className="text-xs font-medium text-slate-500">
                  Transformation
                </p>

                <code className="mt-1 block text-xs text-slate-700">
                  {
                    item.transformation_logic
                  }
                </code>
              </div>
            </article>
          ))}

        {(!Array.isArray(lineage) ||
          lineage.length === 0) && (
          <EmptyState message="No lineage information found." />
        )}
      </div>
    </div>
  )
}

function OnlineFeaturesTab({
  entityId,
  setEntityId,
  featureVersion,
  setFeatureVersion,
  query,
}) {
  return (
    <div>
      <div>
        <h2 className="font-semibold text-slate-900">
          Online Feature Retrieval
        </h2>

        <p className="mt-1 text-sm text-slate-500">
          Retrieve the materialized feature vector
          for an employee and explicit feature
          version.
        </p>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        <Field
          label="Entity ID"
          value={entityId}
          onChange={setEntityId}
          placeholder="emp_1"
        />

        <Field
          label="Feature Version"
          value={featureVersion}
          onChange={setFeatureVersion}
          placeholder="v3"
        />

        <div className="flex items-end">
          <div className="w-full rounded-lg bg-slate-50 p-3">
            <p className="text-xs text-slate-500">
              Retrieval status
            </p>

            <p className="mt-1 text-sm font-semibold text-slate-900">
              {query.isFetching
                ? 'Loading...'
                : query.isError
                  ? 'Not found'
                  : 'Ready'}
            </p>
          </div>
        </div>
      </div>

      {query.isError && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4">
          <p className="text-sm font-medium text-red-800">
            Feature retrieval failed.
          </p>

          <p className="mt-1 text-xs text-red-700">
            {query.error?.message}
          </p>
        </div>
      )}

      {query.data && (
        <div className="mt-6 rounded-lg border border-slate-200">
          <div className="border-b border-slate-200 bg-slate-50 px-5 py-4">
            <p className="text-xs text-slate-500">
              Entity
            </p>

            <p className="mt-1 text-sm font-semibold text-slate-900">
              {query.data.entity_id}
            </p>
          </div>

          <div className="divide-y divide-slate-100">
            {Object.entries(
              query.data.features ?? {},
            ).map(
              ([name, value]) => (
                <div
                  key={name}
                  className="flex items-center justify-between px-5 py-4"
                >
                  <span className="text-sm text-slate-700">
                    {name}
                  </span>

                  <code className="text-sm font-semibold text-slate-900">
                    {String(value)}
                  </code>
                </div>
              ),
            )}
          </div>
        </div>
      )}

      {!query.data &&
        !query.isFetching &&
        !query.isError && (
          <EmptyState message="Enter an entity ID to retrieve online features." />
        )}
    </div>
  )
}

function PointInTimeTab({
  entities,
  setEntities,
  features,
  setFeatures,
  timestamp,
  setTimestamp,
  featureVersion,
  setFeatureVersion,
  result,
  loading,
  error,
  onLookup,
}) {
  function toggleFeature(feature) {
    if (features.includes(feature)) {
      setFeatures(
        features.filter(
          (item) => item !== feature,
        ),
      )

      return
    }

    setFeatures([
      ...features,
      feature,
    ])
  }

  return (
    <div>
      <div>
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-slate-100 p-2 text-slate-600">
            <Clock3 size={19} />
          </div>

          <div>
            <h2 className="font-semibold text-slate-900">
              Point-in-Time Retrieval
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              Retrieve the feature representation that
              was available at or before a specific
              timestamp.
            </p>
          </div>
        </div>
      </div>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <Field
          label="Entity IDs"
          value={entities}
          onChange={setEntities}
          placeholder="emp_1, emp_2, emp_3"
        />

        <Field
          label="Feature Version"
          value={featureVersion}
          onChange={setFeatureVersion}
          placeholder="v3"
        />

        <div>
          <label className="block">
            <span className="text-xs font-medium text-slate-600">
              As-of Timestamp
            </span>

            <input
              type="datetime-local"
              value={toDatetimeLocal(
                timestamp,
              )}
              onChange={(event) =>
                setTimestamp(
                  fromDatetimeLocal(
                    event.target.value,
                  ),
                )
              }
              className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
            />
          </label>

          <p className="mt-1.5 text-xs text-slate-400">
            Only feature values available at this
            point in time are returned.
          </p>
        </div>
      </div>

      <div className="mt-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="text-sm font-semibold text-slate-900">
              Requested Features
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Select the features needed for the
              historical feature vector.
            </p>
          </div>

          <div className="flex gap-3">
            <button
              type="button"
              onClick={() =>
                setFeatures([
                  ...CANONICAL_FEATURES,
                ])
              }
              className="text-xs font-medium text-brand-600 hover:text-brand-700"
            >
              Select all
            </button>

            <button
              type="button"
              onClick={() =>
                setFeatures([])
              }
              className="text-xs font-medium text-slate-500 hover:text-slate-700"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          {CANONICAL_FEATURES.map(
            (feature) => {
              const selected =
                features.includes(feature)

              return (
                <button
                  key={feature}
                  type="button"
                  onClick={() =>
                    toggleFeature(feature)
                  }
                  className={[
                    'flex items-center justify-between rounded-lg border p-3 text-left transition',
                    selected
                      ? 'border-brand-300 bg-brand-50'
                      : 'border-slate-200 bg-white hover:bg-slate-50',
                  ].join(' ')}
                >
                  <span className="text-sm text-slate-700">
                    {feature}
                  </span>

                  {selected && (
                    <CheckCircle2
                      size={16}
                      className="text-brand-600"
                    />
                  )}
                </button>
              )
            },
          )}
        </div>
      </div>

      <button
        type="button"
        disabled={
          loading ||
          !entities.trim() ||
          features.length === 0 ||
          !timestamp
        }
        onClick={onLookup}
        className="mt-6 inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Clock3 size={16} />

        {loading
          ? 'Retrieving...'
          : 'Retrieve Features'}
      </button>

      {error && (
        <div className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4">
          <div className="flex items-start gap-2">
            <AlertCircle
              size={17}
              className="mt-0.5 text-red-600"
            />

            <div>
              <p className="text-sm font-semibold text-red-800">
                Point-in-time retrieval failed
              </p>

              <p className="mt-1 text-xs text-red-700">
                {error}
              </p>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="mt-8 rounded-xl border border-slate-200">
          <div className="border-b border-slate-200 bg-slate-50 px-5 py-4">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-xs text-slate-500">
                  Feature Version
                </p>

                <p className="mt-1 text-sm font-semibold text-slate-900">
                  {result.feature_version}
                </p>
              </div>

              <div className="md:text-right">
                <p className="text-xs text-slate-500">
                  As-of
                </p>

                <p className="mt-1 text-sm font-semibold text-slate-900">
                  {formatTimestamp(
                    result.as_of_timestamp,
                  )}
                </p>
              </div>
            </div>
          </div>

          <div className="divide-y divide-slate-100">
            {Object.entries(
              result.entities ?? {},
            ).map(
              ([entity, entityFeatures]) => (
                <div
                  key={entity}
                  className="p-5"
                >
                  <div className="flex items-center gap-2">
                    <Database
                      size={16}
                      className="text-slate-400"
                    />

                    <h3 className="text-sm font-semibold text-slate-900">
                      {entity}
                    </h3>
                  </div>

                  <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {Object.entries(
                      entityFeatures ?? {},
                    ).map(
                      ([featureName, value]) => (
                        <div
                          key={featureName}
                          className="rounded-lg bg-slate-50 p-3"
                        >
                          <p className="text-xs text-slate-500">
                            {featureName}
                          </p>

                          <p className="mt-1 text-sm font-semibold text-slate-900">
                            {String(value)}
                          </p>
                        </div>
                      ),
                    )}
                  </div>
                </div>
              ),
            )}
          </div>

          {Object.keys(
            result.entities ?? {},
          ).length === 0 && (
            <div className="p-8 text-center">
              <p className="text-sm font-medium text-slate-700">
                No features were available at this
                timestamp.
              </p>

              <p className="mt-1 text-xs text-slate-500">
                Try a timestamp after the relevant
                materialization.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function MaterializationsTab({
  materializations,
}) {
  return (
    <div>
      <div className="flex items-center gap-3">
        <Clock3
          size={19}
          className="text-slate-500"
        />

        <div>
          <h2 className="font-semibold text-slate-900">
            Materialization History
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            Recent feature store materialization
            runs.
          </p>
        </div>
      </div>

      <div className="mt-6 space-y-3">
        {materializations.map(
          (item) => (
            <article
              key={
                item.materialization_id
              }
              className="rounded-lg border border-slate-200 p-5"
            >
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold text-slate-900">
                      {
                        item.materialization_id
                      }
                    </span>

                    <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700">
                      {
                        item.feature_version
                      }
                    </span>

                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700">
                      <CheckCircle2
                        size={12}
                      />

                      {item.status}
                    </span>
                  </div>

                  <p className="mt-2 text-xs text-slate-500">
                    Batch: {item.batch_id}
                  </p>
                </div>

                <div className="text-left md:text-right">
                  <p className="text-sm font-semibold text-slate-900">
                    {item.entity_count}{' '}
                    entities
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    {formatTimestamp(
                      item.timestamp,
                    )}
                  </p>
                </div>
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                {item.features_materialized?.map(
                  (feature) => (
                    <span
                      key={feature}
                      className="rounded-md bg-slate-50 px-2 py-1 text-xs text-slate-600"
                    >
                      {feature}
                    </span>
                  ),
                )}
              </div>
            </article>
          ),
        )}

        {materializations.length ===
          0 && (
          <EmptyState message="No materialization history found." />
        )}
      </div>
    </div>
  )
}

function Field({
  label,
  value,
  onChange,
  placeholder,
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-600">
        {label}
      </span>

      <input
        value={value}
        onChange={(event) =>
          onChange(event.target.value)
        }
        placeholder={placeholder}
        className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
      />
    </label>
  )
}

function EmptyState({ message }) {
  return (
    <div className="rounded-lg bg-slate-50 p-8 text-center">
      <p className="text-sm text-slate-500">
        {message}
      </p>
    </div>
  )
}

function toDatetimeLocal(value) {
  if (!value) {
    return ''
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  const offset =
    date.getTimezoneOffset() * 60_000

  return new Date(
    date.getTime() - offset,
  )
    .toISOString()
    .slice(0, 16)
}

function fromDatetimeLocal(value) {
  if (!value) {
    return ''
  }

  return new Date(value).toISOString()
}

function formatTimestamp(value) {
  if (!value) {
    return '—'
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return value
  }

  return date.toLocaleString()
}

export default FeatureStore