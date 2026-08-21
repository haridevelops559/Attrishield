import { useState } from 'react'
import { predictIndividualEmployee } from '../services/api'

const initialFormData = {
  Age: 35,
  BusinessTravel: 'Travel_Rarely',
  DailyRate: 800,
  Department: 'Research & Development',
  DistanceFromHome: 10,
  Education: 3,
  EducationField: 'Life Sciences',
  EnvironmentSatisfaction: 3,
  Gender: 'Male',
  HourlyRate: 65,
  JobInvolvement: 3,
  JobLevel: 2,
  JobRole: 'Research Scientist',
  JobSatisfaction: 4,
  MaritalStatus: 'Single',
  MonthlyIncome: 5000,
  MonthlyRate: 15000,
  NumCompaniesWorked: 2,
  OverTime: 'Yes',
  PercentSalaryHike: 15,
  PerformanceRating: 3,
  RelationshipSatisfaction: 3,
  StockOptionLevel: 1,
  TotalWorkingYears: 10,
  TrainingTimesLastYear: 2,
  WorkLifeBalance: 3,
  YearsAtCompany: 5,
  YearsInCurrentRole: 3,
  YearsSinceLastPromotion: 1,
  YearsWithCurrManager: 2,
}

function IndividualPrediction() {
  const [formData, setFormData] = useState(initialFormData)
  const [errors, setErrors] = useState({})
  const [status, setStatus] = useState('idle')
  const [prediction, setPrediction] = useState(null)
  const [submitError, setSubmitError] = useState(null)

  function handleChange(event) {
    const { name, value, type } = event.target

    setFormData((current) => ({
      ...current,
      [name]: type === 'number' ? Number(value) : value,
    }))

    setErrors((current) => ({
      ...current,
      [name]: undefined,
    }))
  }

  function validate() {
    const nextErrors = {}

    if (formData.Age < 18 || formData.Age > 100) {
      nextErrors.Age = 'Age must be between 18 and 100.'
    }

    if (formData.MonthlyIncome < 0) {
      nextErrors.MonthlyIncome =
        'Monthly income cannot be negative.'
    }

    if (formData.YearsAtCompany < 0) {
      nextErrors.YearsAtCompany =
        'Years at company cannot be negative.'
    }

    if (formData.YearsAtCompany > formData.TotalWorkingYears) {
      nextErrors.YearsAtCompany =
        'Years at company cannot exceed total working years.'
    }

    return nextErrors
  }

  async function handleSubmit(event) {
    event.preventDefault()

    const nextErrors = validate()

    setErrors(nextErrors)
    setSubmitError(null)
    setPrediction(null)

    if (Object.keys(nextErrors).length > 0) {
      setStatus('idle')
      return
    }

    try {
      setStatus('submitting')

      const result = await predictIndividualEmployee(formData)

      setPrediction(result)
      setStatus('success')
    } catch (error) {
      setSubmitError(error.message)
      setStatus('error')
    }
  }

  return (
    <section className="mx-auto max-w-6xl">
      <div>
        <p className="text-sm font-medium text-brand-600">
          ML Inference
        </p>

        <h1 className="mt-1 text-2xl font-bold text-slate-900">
          Individual Attrition Prediction
        </h1>

        <p className="mt-2 text-sm text-slate-600">
          Enter employee information to prepare an attrition
          prediction request.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="mt-8 space-y-6">
        <FormSection
          title="Employee Information"
          description="Basic employee demographic information."
        >
          <NumberField
            label="Age"
            name="Age"
            value={formData.Age}
            onChange={handleChange}
            min="18"
            max="100"
            error={errors.Age}
          />

          <SelectField
            label="Gender"
            name="Gender"
            value={formData.Gender}
            onChange={handleChange}
            options={['Male', 'Female']}
          />

          <SelectField
            label="Marital Status"
            name="MaritalStatus"
            value={formData.MaritalStatus}
            onChange={handleChange}
            options={['Single', 'Married', 'Divorced']}
          />

          <NumberField
            label="Distance From Home"
            name="DistanceFromHome"
            value={formData.DistanceFromHome}
            onChange={handleChange}
            min="0"
          />
        </FormSection>

        <FormSection
          title="Job Information"
          description="Current role, department and employment characteristics."
        >
          <SelectField
            label="Department"
            name="Department"
            value={formData.Department}
            onChange={handleChange}
            options={[
              'Research & Development',
              'Sales',
              'Human Resources',
            ]}
          />

          <SelectField
            label="Business Travel"
            name="BusinessTravel"
            value={formData.BusinessTravel}
            onChange={handleChange}
            options={[
              'Travel_Rarely',
              'Travel_Frequently',
              'Non-Travel',
            ]}
          />

          <SelectField
            label="Job Role"
            name="JobRole"
            value={formData.JobRole}
            onChange={handleChange}
            options={[
              'Research Scientist',
              'Laboratory Technician',
              'Sales Executive',
              'Sales Representative',
              'Manager',
              'Healthcare Representative',
              'Manufacturing Director',
              'Human Resources',
              'Research Director',
            ]}
          />

          <NumberField
            label="Job Level"
            name="JobLevel"
            value={formData.JobLevel}
            onChange={handleChange}
            min="1"
            max="5"
          />

          <SelectField
            label="Overtime"
            name="OverTime"
            value={formData.OverTime}
            onChange={handleChange}
            options={['Yes', 'No']}
          />
        </FormSection>

        <FormSection
          title="Compensation"
          description="Employee compensation and salary-related features."
        >
          <NumberField
            label="Daily Rate"
            name="DailyRate"
            value={formData.DailyRate}
            onChange={handleChange}
            min="0"
          />

          <NumberField
            label="Hourly Rate"
            name="HourlyRate"
            value={formData.HourlyRate}
            onChange={handleChange}
            min="0"
          />

          <NumberField
            label="Monthly Income"
            name="MonthlyIncome"
            value={formData.MonthlyIncome}
            onChange={handleChange}
            min="0"
            error={errors.MonthlyIncome}
          />

          <NumberField
            label="Monthly Rate"
            name="MonthlyRate"
            value={formData.MonthlyRate}
            onChange={handleChange}
            min="0"
          />

          <NumberField
            label="Percent Salary Hike"
            name="PercentSalaryHike"
            value={formData.PercentSalaryHike}
            onChange={handleChange}
            min="0"
          />
        </FormSection>

        <FormSection
          title="Satisfaction & Engagement"
          description="Work environment and employee engagement signals."
        >
          <NumberField
            label="Environment Satisfaction"
            name="EnvironmentSatisfaction"
            value={formData.EnvironmentSatisfaction}
            onChange={handleChange}
            min="1"
            max="4"
          />

          <NumberField
            label="Job Satisfaction"
            name="JobSatisfaction"
            value={formData.JobSatisfaction}
            onChange={handleChange}
            min="1"
            max="4"
          />

          <NumberField
            label="Job Involvement"
            name="JobInvolvement"
            value={formData.JobInvolvement}
            onChange={handleChange}
            min="1"
            max="4"
          />

          <NumberField
            label="Relationship Satisfaction"
            name="RelationshipSatisfaction"
            value={formData.RelationshipSatisfaction}
            onChange={handleChange}
            min="1"
            max="4"
          />

          <NumberField
            label="Work Life Balance"
            name="WorkLifeBalance"
            value={formData.WorkLifeBalance}
            onChange={handleChange}
            min="1"
            max="4"
          />
        </FormSection>

        <FormSection
          title="Career History"
          description="Employee tenure and career progression."
        >
          <NumberField
            label="Total Working Years"
            name="TotalWorkingYears"
            value={formData.TotalWorkingYears}
            onChange={handleChange}
            min="0"
          />

          <NumberField
            label="Years At Company"
            name="YearsAtCompany"
            value={formData.YearsAtCompany}
            onChange={handleChange}
            min="0"
            error={errors.YearsAtCompany}
          />

          <NumberField
            label="Years In Current Role"
            name="YearsInCurrentRole"
            value={formData.YearsInCurrentRole}
            onChange={handleChange}
            min="0"
          />

          <NumberField
            label="Years Since Last Promotion"
            name="YearsSinceLastPromotion"
            value={formData.YearsSinceLastPromotion}
            onChange={handleChange}
            min="0"
          />

          <NumberField
            label="Years With Current Manager"
            name="YearsWithCurrManager"
            value={formData.YearsWithCurrManager}
            onChange={handleChange}
            min="0"
          />

          <NumberField
            label="Number Of Companies Worked"
            name="NumCompaniesWorked"
            value={formData.NumCompaniesWorked}
            onChange={handleChange}
            min="0"
          />

          <NumberField
            label="Training Times Last Year"
            name="TrainingTimesLastYear"
            value={formData.TrainingTimesLastYear}
            onChange={handleChange}
            min="0"
          />
        </FormSection>

        <FormSection
          title="Additional Features"
          description="Remaining features required by the V3 inference contract."
        >
          <SelectField
            label="Education Field"
            name="EducationField"
            value={formData.EducationField}
            onChange={handleChange}
            options={[
              'Life Sciences',
              'Medical',
              'Marketing',
              'Technical Degree',
              'Human Resources',
              'Other',
            ]}
          />

          <NumberField
            label="Education"
            name="Education"
            value={formData.Education}
            onChange={handleChange}
            min="1"
            max="5"
          />

          <NumberField
            label="Performance Rating"
            name="PerformanceRating"
            value={formData.PerformanceRating}
            onChange={handleChange}
            min="1"
            max="5"
          />

          <NumberField
            label="Stock Option Level"
            name="StockOptionLevel"
            value={formData.StockOptionLevel}
            onChange={handleChange}
            min="0"
            max="3"
          />
        </FormSection>

        <div className="flex items-center justify-end gap-4">
          {status === 'error' && (
            <p className="text-sm text-red-600">
              {submitError}
            </p>
          )}

          {status === 'success' && (
            <p className="text-sm font-medium text-green-600">
              Prediction completed.
            </p>
          )}

          <button
            type="submit"
            disabled={status === 'submitting'}
            className="rounded-lg bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {status === 'submitting'
              ? 'Running Prediction...'
              : 'Run Prediction'}
          </button>
        </div>
      </form>

      {prediction && (
        <section className="mt-6 rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="font-semibold text-slate-900">
            Prediction Result
          </h2>

          <pre className="mt-4 overflow-auto rounded-lg bg-slate-50 p-4 text-xs text-slate-700">
            {JSON.stringify(prediction, null, 2)}
          </pre>
        </section>
      )}
    </section>
  )
}

function FormSection({ title, description, children }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6">
      <h2 className="font-semibold text-slate-900">
        {title}
      </h2>

      <p className="mt-1 text-sm text-slate-500">
        {description}
      </p>

      <div className="mt-6 grid gap-5 md:grid-cols-2">
        {children}
      </div>
    </section>
  )
}

function NumberField({
  label,
  name,
  value,
  onChange,
  min,
  max,
  error,
}) {
  return (
    <div>
      <label
        htmlFor={name}
        className="block text-sm font-medium text-slate-700"
      >
        {label}
      </label>

      <input
        id={name}
        name={name}
        type="number"
        value={value}
        onChange={onChange}
        min={min}
        max={max}
        className={`mt-1 w-full rounded-lg border px-3 py-2 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100 ${
          error
            ? 'border-red-400'
            : 'border-slate-300'
        }`}
      />

      {error && (
        <p className="mt-1 text-xs text-red-600">
          {error}
        </p>
      )}
    </div>
  )
}

function SelectField({
  label,
  name,
  value,
  onChange,
  options,
}) {
  return (
    <div>
      <label
        htmlFor={name}
        className="block text-sm font-medium text-slate-700"
      >
        {label}
      </label>

      <select
        id={name}
        name={name}
        value={value}
        onChange={onChange}
        className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  )
}

export default IndividualPrediction