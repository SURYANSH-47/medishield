/**
 * API Client for MediShield Backend
 * ===================================
 * Centralized API layer for communicating with the FastAPI backend.
 * Provides typed request/response interfaces, retry logic, and
 * comprehensive error handling to prevent frontend crashes.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Priority: explicit env var → localhost dev fallback
// In production (Vercel + Railway), NEXT_PUBLIC_API_URL must be set.
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000"

console.log("[MediShield] API_BASE_URL:", API_BASE_URL)

/** Maximum number of retry attempts for failed requests */
const MAX_RETRIES = 2

/** Delay between retries in ms */
const RETRY_DELAY = 1000

// ---------------------------------------------------------------------------
// Types — must match backend schemas exactly
// ---------------------------------------------------------------------------

/** POST /predict request body (snake_case to match FastAPI Pydantic schema) */
export interface PredictionRequest {
  pregnancies: number
  glucose: number
  blood_pressure: number
  skin_thickness: number
  insulin: number
  bmi: number
  diabetes_pedigree_function: number
  age: number
}

export interface ShapValueItem {
  feature: string
  value: number
}

/** POST /predict response body */
export interface PredictionResponse {
  risk_score: number              // 0.0 – 1.0
  risk_level: "LOW" | "MEDIUM" | "HIGH"
  top_factors: string[]           // e.g. ["Glucose", "BMI", "Age"]
  prediction?: string
  confidence?: number
  risk_factors?: number
  protective_factors?: number
  shap_values?: ShapValueItem[]
  shap_values_dict?: Record<string, number>
  probability_percent?: number
}

/** GET /health response body */
export interface HealthResponse {
  status: string
  model_loaded: boolean
  version: string
}

/** Error shape returned by the backend on 4xx / 5xx */
export interface ApiError {
  error: string
  detail?: string
}

// ---------------------------------------------------------------------------
// Custom error class
// ---------------------------------------------------------------------------

export class MediShieldApiError extends Error {
  status: number
  detail?: string

  constructor(message: string, status: number, detail?: string) {
    super(message)
    this.name = "MediShieldApiError"
    this.status = status
    this.detail = detail
  }
}

// ---------------------------------------------------------------------------
// Internal helper — fetch with timeout & retry
// ---------------------------------------------------------------------------

async function fetchWithRetry(
  url: string,
  options: RequestInit,
  retries = MAX_RETRIES
): Promise<Response> {
  // Add a 15-second timeout via AbortController
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 15_000)

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

    // On server errors, retry if we have attempts left
    if (!response.ok && response.status >= 500 && retries > 0) {
      console.warn(
        `[api] Server error ${response.status} — retrying in ${RETRY_DELAY}ms (${retries} left)`
      )
      await new Promise((r) => setTimeout(r, RETRY_DELAY))
      return fetchWithRetry(url, options, retries - 1)
    }

    return response
  } catch (err) {
    clearTimeout(timeoutId)

    // Retry on network errors (not aborts from timeout)
    if (retries > 0 && err instanceof Error && err.name !== "AbortError") {
      console.warn(
        `[api] Network error — retrying in ${RETRY_DELAY}ms (${retries} left)`
      )
      await new Promise((r) => setTimeout(r, RETRY_DELAY))
      return fetchWithRetry(url, options, retries - 1)
    }

    throw err
  }
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/**
 * Run a disease-risk prediction against the FastAPI backend.
 *
 * @throws {MediShieldApiError} on 4xx/5xx responses
 * @throws {Error} on network / connection failures
 */
export async function predict(
  data: PredictionRequest
): Promise<PredictionResponse> {
  const url = `${API_BASE_URL}/predict`

  const response = await fetchWithRetry(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    let detail: string | undefined
    try {
      const body = await response.json()
      detail = body.detail || body.error || JSON.stringify(body)
    } catch {
      detail = response.statusText
    }
    throw new MediShieldApiError(
      `Prediction failed (${response.status})`,
      response.status,
      detail
    )
  }

  return response.json()
}

/**
 * Check backend health and model readiness.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const url = `${API_BASE_URL}/health`

  const response = await fetchWithRetry(url, { method: "GET" }, 1)

  if (!response.ok) {
    throw new MediShieldApiError(
      "Health check failed",
      response.status,
      response.statusText
    )
  }

  return response.json()
}

/**
 * Ping the backend root endpoint to verify connectivity.
 */
export async function pingBackend(): Promise<boolean> {
  try {
    const response = await fetchWithRetry(
      API_BASE_URL,
      { method: "GET" },
      0 // no retries for a quick ping
    )
    return response.ok
  } catch {
    return false
  }
}

// ---------------------------------------------------------------------------
// New interfaces — Metrics, Add-Patient, Retrain
// ---------------------------------------------------------------------------

export interface MetricsResponse {
  accuracy: number
  test_accuracy: number
  train_accuracy: number
  final_loss: number
  epochs: number
  federated_round: number
  patient_count: number
  hospital_counts: Record<string, number>
  last_trained: string | null
  model_loaded: boolean
  is_training?: boolean
}

export interface AddPatientRequest {
  pregnancies: number
  glucose: number
  blood_pressure: number
  skin_thickness: number
  insulin: number
  bmi: number
  diabetes_pedigree_function: number
  age: number
  outcome: number
  hospital: string
}

export interface AddPatientResponse {
  success: boolean
  message: string
  new_patient_count: number
  hospital: string
}

export interface HospitalMetric {
  name: string
  patients: number
  accuracy: number
  status: "active" | "training" | "syncing" | "idle"
  sync_status: "synced" | "syncing" | "pending"
}

export interface ChartDataItem {
  round: number
  accuracy: number
}

export interface DashboardMetricsResponse {
  global_accuracy: number
  federated_round: number
  total_patients: number
  high_risk_patients: number
  hospitals: HospitalMetric[]
  chart_data: ChartDataItem[]
  train_accuracy?: number
  test_accuracy?: number
  final_loss?: number
  epochs?: number
  last_trained?: string | null
  is_training?: boolean
}

export interface FederatedRoundsResponse {
  federated_round: number
  total_rounds: number
  history: ChartDataItem[]
}

/**
 * Fetch live model metrics from the backend.
 */
export async function getMetrics(): Promise<MetricsResponse> {
  const response = await fetch(`${API_BASE_URL}/metrics`, { method: "GET", cache: "no-store" })
  if (!response.ok) throw new Error("Failed to fetch metrics")
  return response.json()
}

/**
 * Add a new patient record to a hospital node.
 */
export async function addPatient(data: AddPatientRequest): Promise<AddPatientResponse> {
  const response = await fetch(`${API_BASE_URL}/add-patient`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!response.ok) throw new Error("Failed to add patient")
  return response.json()
}

/**
 * Trigger a new federated learning training round.
 */
export async function triggerRetrain(): Promise<{ status: string; message: string; federated_round: number }> {
  const response = await fetch(`${API_BASE_URL}/retrain`, { method: "POST" })
  if (!response.ok) throw new Error("Failed to trigger retrain")
  return response.json()
}

/**
 * Fetch dynamic dashboard metrics.
 */
export async function getDashboardMetrics(): Promise<DashboardMetricsResponse> {
  const response = await fetch(`${API_BASE_URL}/dashboard-metrics`, { method: "GET", cache: "no-store" })
  if (!response.ok) throw new Error("Failed to fetch dashboard metrics")
  return response.json()
}

/**
 * Fetch local hospital-specific metrics.
 * Uses cache-busting (timestamp + no-store) to guarantee browser never
 * serves a stale response — hospital CSVs can change at any time.
 */
export async function getHospitalMetrics(): Promise<{ hospitals: HospitalMetric[] }> {
  const ts = Date.now()  // cache-buster query param
  const response = await fetch(`${API_BASE_URL}/hospital-metrics?_t=${ts}`, {
    method: "GET",
    cache: "no-store",
    headers: {
      "Cache-Control": "no-cache",
      "Pragma": "no-cache",
    },
  })
  if (!response.ok) throw new Error("Failed to fetch hospital metrics")
  return response.json()
}

/**
 * Fetch federated rounds info and history.
 */
export async function getFederatedRounds(): Promise<FederatedRoundsResponse> {
  const response = await fetch(`${API_BASE_URL}/federated-rounds`, { method: "GET", cache: "no-store" })
  if (!response.ok) throw new Error("Failed to fetch federated rounds")
  return response.json()
}

/**
 * Fetch chart data for learning curve visualization.
 */
export async function getChartData(): Promise<ChartDataItem[]> {
  const response = await fetch(`${API_BASE_URL}/chart-data`, { method: "GET", cache: "no-store" })
  if (!response.ok) throw new Error("Failed to fetch chart data")
  return response.json()
}

// ---------------------------------------------------------------------------
// Hospital-specific types and API functions (per-hospital management)
// ---------------------------------------------------------------------------

/** Real-time stats for a single hospital node */
export interface HospitalStats {
  hospital_id: string
  hospital_name: string
  full_name: string
  specialty: string
  patient_count: number
  local_accuracy: number | null   // null until first local training run
  last_trained: string | null
  is_training: boolean
  train_loss: number | null
}

/** POST body for adding a patient directly to a hospital CSV */
export interface HospitalPatientRequest {
  pregnancies: number
  glucose: number
  blood_pressure: number
  skin_thickness: number
  insulin: number
  bmi: number
  diabetes_pedigree_function: number
  age: number
  outcome: number                 // 0 = Healthy, 1 = Diabetic
}

/** A single row from the hospital dataset preview */
export interface DatasetRow {
  Pregnancies: number
  Glucose: number
  BloodPressure: number
  SkinThickness: number
  Insulin: number
  BMI: number
  DiabetesPedigreeFunction: number
  Age: number
  Outcome: number
}

/** GET /hospital/{id}/dataset-preview response */
export interface DatasetPreviewResponse {
  hospital_id: string
  page: number
  per_page: number
  total_rows: number
  total_pages: number
  rows: DatasetRow[]
}

/**
 * Fetch real-time stats for a single hospital node.
 */
export async function getHospitalStats(hospitalId: string): Promise<HospitalStats> {
  const ts = Date.now()
  const response = await fetch(`${API_BASE_URL}/hospital/${hospitalId}/stats?_t=${ts}`, {
    method: "GET",
    cache: "no-store",
    headers: { "Cache-Control": "no-cache", "Pragma": "no-cache" },
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error((err as any).detail || `Failed to fetch stats for hospital ${hospitalId}`)
  }
  return response.json()
}

/**
 * Add a new patient row directly to a hospital's CSV file.
 */
export async function addHospitalPatient(
  hospitalId: string,
  data: HospitalPatientRequest
): Promise<{ success: boolean; message: string; new_patient_count: number }> {
  const response = await fetch(`${API_BASE_URL}/hospital/${hospitalId}/add-patient`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error((err as any).detail || `Failed to add patient to hospital ${hospitalId}`)
  }
  return response.json()
}

/**
 * Trigger local model retraining for a specific hospital node.
 * Runs in a background thread — poll /hospital/{id}/stats to track progress.
 */
export async function retrainHospital(
  hospitalId: string
): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE_URL}/hospital/${hospitalId}/retrain`, {
    method: "POST",
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error((err as any).detail || `Failed to retrain hospital ${hospitalId}`)
  }
  return response.json()
}

/**
 * Fetch a paginated preview of the hospital's local dataset (newest rows first).
 */
export async function getHospitalDatasetPreview(
  hospitalId: string,
  page = 1,
  perPage = 10
): Promise<DatasetPreviewResponse> {
  const ts = Date.now()
  const response = await fetch(
    `${API_BASE_URL}/hospital/${hospitalId}/dataset-preview?page=${page}&per_page=${perPage}&_t=${ts}`,
    { method: "GET", cache: "no-store" }
  )
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error((err as any).detail || `Failed to fetch dataset preview for hospital ${hospitalId}`)
  }
  return response.json()
}

// ---------------------------------------------------------------------------
// Real-Time Analytics API
// ---------------------------------------------------------------------------

export interface AnalyticsOverview {
  total_predictions: number
  high_risk_patients: number
  participating_hospitals: number
  fairness_score: number
  global_accuracy: number
  federated_round: number
}

export interface TrendData {
  round: number
  accuracy: number
}

export interface AnalyticsTrends {
  diabetes: TrendData[]
  heart_disease: TrendData[]
  hypertension: TrendData[]
}

export interface HospitalContribution {
  hospital: string
  patients: number
  predictions: number
  accuracy: number
}

export interface HospitalContributionsResponse {
  hospitals: HospitalContribution[]
  total_patients: number
}

export interface RiskCategory {
  name: string
  value: number
  color: string
}

export interface RiskDistributionResponse {
  categories: RiskCategory[]
}

export interface AgeGroupData {
  age: string
  low: number
  medium: number
  high: number
}

export interface AgeGroupRiskResponse {
  age_groups: AgeGroupData[]
}

export interface GenderMetric {
  metric: string
  male: number
  female: number
}

export interface GenderPerformanceResponse {
  metrics: GenderMetric[]
}

export interface FairnessGroup {
  group: string
  value: number
}

export interface FairnessResponse {
  groups: FairnessGroup[]
  overall_status: string
}

export interface ModelPerformanceResponse {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  roc_auc: number
  epochs: number
  final_loss: number
  federated_round: number
}

/**
 * Fetch high-level analytics overview.
 */
export async function getAnalyticsOverview(): Promise<AnalyticsOverview> {
  const response = await fetch(`${API_BASE_URL}/analytics/overview`, {
    method: "GET",
    cache: "no-store",
  })
  if (!response.ok) throw new Error("Failed to fetch analytics overview")
  return response.json()
}

/**
 * Fetch disease trend analysis data.
 */
export async function getAnalyticsTrends(): Promise<AnalyticsTrends> {
  const response = await fetch(`${API_BASE_URL}/analytics/trends`, {
    method: "GET",
    cache: "no-store",
  })
  if (!response.ok) throw new Error("Failed to fetch analytics trends")
  return response.json()
}

/**
 * Fetch hospital contribution analytics.
 */
export async function getHospitalContributions(): Promise<HospitalContributionsResponse> {
  const response = await fetch(`${API_BASE_URL}/analytics/hospital-contributions`, {
    method: "GET",
    cache: "no-store",
  })
  if (!response.ok) throw new Error("Failed to fetch hospital contributions")
  return response.json()
}

/**
 * Fetch patient risk distribution.
 */
export async function getRiskDistribution(): Promise<RiskDistributionResponse> {
  const response = await fetch(`${API_BASE_URL}/analytics/risk-distribution`, {
    method: "GET",
    cache: "no-store",
  })
  if (!response.ok) throw new Error("Failed to fetch risk distribution")
  return response.json()
}

/**
 * Fetch age group risk analysis.
 */
export async function getAgeGroupRisk(): Promise<AgeGroupRiskResponse> {
  const response = await fetch(`${API_BASE_URL}/analytics/age-group-risk`, {
    method: "GET",
    cache: "no-store",
  })
  if (!response.ok) throw new Error("Failed to fetch age group risk")
  return response.json()
}

/**
 * Fetch gender-based model performance.
 */
export async function getGenderPerformance(): Promise<GenderPerformanceResponse> {
  const response = await fetch(`${API_BASE_URL}/analytics/gender-performance`, {
    method: "GET",
    cache: "no-store",
  })
  if (!response.ok) throw new Error("Failed to fetch gender performance")
  return response.json()
}

/**
 * Fetch fairness monitoring metrics.
 */
export async function getFairnessMonitoring(): Promise<FairnessResponse> {
  const response = await fetch(`${API_BASE_URL}/analytics/fairness`, {
    method: "GET",
    cache: "no-store",
  })
  if (!response.ok) throw new Error("Failed to fetch fairness metrics")
  return response.json()
}

/**
 * Fetch comprehensive model performance metrics.
 */
export async function getModelPerformance(): Promise<ModelPerformanceResponse> {
  const response = await fetch(`${API_BASE_URL}/analytics/model-performance`, {
    method: "GET",
    cache: "no-store",
  })
  if (!response.ok) throw new Error("Failed to fetch model performance")
  return response.json()
}

// ---------------------------------------------------------------------------
// AI Clinical Emergency Support Assistant — POST /clinical-support
// ---------------------------------------------------------------------------

/** Request body for the Clinical Support AI engine */
export interface ClinicalSupportRequest {
  risk_score: number
  risk_level: string
  top_factors: string[]
  shap_values_dict?: Record<string, number>
  // Patient values
  glucose: number
  bmi: number
  blood_pressure: number
  age: number
  insulin: number
  pregnancies?: number
  // Federated context (optional)
  federated_round?: number
  global_accuracy?: number
}

/** Structured clinical recommendations returned by the support engine */
export interface ClinicalSupportResponse {
  severity_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
  emergency_priority: "ROUTINE" | "ELEVATED" | "URGENT" | "EMERGENCY"
  ai_summary: string
  recommendations: string[]
  monitoring_checklist: string[]
  escalation_criteria: string[]
  next_steps: string[]
  primary_contributors: string[]
  federated_note: string
  disclaimer: string
}

/**
 * Call the Clinical Support AI engine.
 * Returns structured nursing/clinical recommendations for the given prediction.
 */
export async function getClinicalSupport(
  data: ClinicalSupportRequest
): Promise<ClinicalSupportResponse> {
  const response = await fetchWithRetry(`${API_BASE_URL}/clinical-support`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })

  if (!response.ok) {
    let detail: string | undefined
    try {
      const body = await response.json()
      detail = body.detail || body.error || JSON.stringify(body)
    } catch {
      detail = response.statusText
    }
    throw new MediShieldApiError(
      `Clinical support failed (${response.status})`,
      response.status,
      detail
    )
  }

  return response.json()
}

