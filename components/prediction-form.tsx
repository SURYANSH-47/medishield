"use client"

import { motion, AnimatePresence } from "framer-motion"
import { useState } from "react"
import {
  Activity,
  Heart,
  Droplet,
  Thermometer,
  Scale,
  User,
  Zap,
  AlertTriangle,
  CheckCircle,
  Info,
  Sparkles,
  ArrowRight,
  Dna,
  WifiOff,
  BarChart2,
  Stethoscope,
} from "lucide-react"

import { predict, getClinicalSupport } from "@/lib/api"
import type { ClinicalSupportResponse } from "@/lib/api"
import { savePrediction } from "@/lib/prediction-store"
import { ClinicalSupportPanel } from "@/components/clinical-support-panel"
import { useLanguage } from "@/hooks/use-language"
import { getTranslation, type TranslationDict } from "@/lib/translations"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FormData {
  age: number
  bmi: number
  glucose: number
  bloodPressure: number
  diabetesPedigreeFunction: number   // replaces heartRate — required by /predict
  insulin: number
  pregnancies: number
  skinThickness: number
}

/** Shape of the POST /predict request body */
interface ApiRequest {
  pregnancies: number
  glucose: number
  blood_pressure: number
  skin_thickness: number
  insulin: number
  bmi: number
  diabetes_pedigree_function: number
  age: number
}

/** Shape of the POST /predict response */
interface ApiResponse {
  risk_score: number                       // float 0–1
  risk_level: "LOW" | "MEDIUM" | "HIGH"
  top_factors: string[]                    // SHAP-derived feature names
  shap_values?: Record<string, number>     // per-feature SHAP contributions
  shap_values_dict?: Record<string, number> // keyed SHAP dict
  probability_percent?: number             // optional probability 0-100
  // federated context
  federated_round?: number
  global_accuracy?: number
}

/** Internal display model derived from the API response */
interface PredictionResult {
  riskScore: number                        // 0–100 percentage
  riskLevel: "LOW" | "MEDIUM" | "HIGH"
  topFactors: string[]
  recommendation: string
  shapValues?: Record<string, number>      // forwarded from API
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Maps risk level → client-side recommendation text */
const RECOMMENDATIONS: Record<"LOW" | "MEDIUM" | "HIGH", string> = {
  LOW:
    "Your health indicators suggest a low risk for diabetes. Continue maintaining a healthy lifestyle with regular exercise and a balanced diet. Schedule routine check-ups every 6–12 months.",
  MEDIUM:
    "Your health indicators suggest moderate risk factors. Consider consulting with a healthcare provider for personalized guidance. Focus on weight management, regular physical activity, and monitoring your glucose levels more frequently.",
  HIGH:
    "Your health indicators suggest elevated risk factors that warrant immediate attention. Please consult with a healthcare provider as soon as possible for comprehensive evaluation and potential intervention strategies. Early detection and management are crucial.",
}

/** Input field metadata */
const inputFields = [
  { name: "age",                       label: "Age",                        icon: User,         unit: "years",  min: 0,   max: 120 },
  { name: "bmi",                       label: "BMI",                        icon: Scale,        unit: "kg/m²",  min: 10,  max: 60  },
  { name: "glucose",                   label: "Glucose Level",              icon: Droplet,      unit: "mg/dL",  min: 0,   max: 300 },
  { name: "bloodPressure",             label: "Blood Pressure",             icon: Activity,     unit: "mmHg",   min: 0,   max: 200 },
  { name: "insulin",                   label: "Insulin",                    icon: Zap,          unit: "μU/mL",  min: 0,   max: 900 },
  { name: "pregnancies",               label: "Pregnancies",                icon: Heart,        unit: "",       min: 0,   max: 20  },
  { name: "skinThickness",             label: "Skin Thickness",             icon: Thermometer,  unit: "mm",     min: 0,   max: 100 },
  { name: "diabetesPedigreeFunction",  label: "Diabetes Pedigree Function", icon: Dna,          unit: "",       min: 0,   max: 2.5 },
]

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function InputField({
  name, label, icon: Icon, unit, value, onChange, min, max,
}: {
  name: string
  label: string
  icon: React.ElementType
  unit: string
  value: number
  onChange: (name: string, value: number) => void
  min: number
  max: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-xl p-4 hover:glow-border transition-all duration-300"
    >
      <div className="flex items-center gap-3 mb-3">
        <div className="p-2 rounded-lg bg-primary/20">
          <Icon className="h-4 w-4 text-primary" />
        </div>
        <label className="text-sm font-medium text-foreground">{label}</label>
      </div>
      <div className="relative">
        <input
          type="number"
          value={value || ""}
          onChange={(e) => onChange(name, parseFloat(e.target.value) || 0)}
          min={min}
          max={max}
          step={name === "diabetesPedigreeFunction" || name === "bmi" ? 0.01 : 1}
          className="w-full bg-secondary/50 border border-border rounded-lg px-4 py-3 text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all"
          placeholder={`Enter ${label.toLowerCase()}`}
        />
        {unit && (
          <span className="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-muted-foreground">
            {unit}
          </span>
        )}
      </div>
    </motion.div>
  )
}

/** Colour-coded risk level badge — label translated to the active clinical language */
function RiskBadge({ level }: { level: "LOW" | "MEDIUM" | "HIGH" }) {
  const { language } = useLanguage()

  const styles = {
    LOW:    { bg: "bg-success/20",     border: "border-success",     text: "text-success",     icon: CheckCircle  },
    MEDIUM: { bg: "bg-warning/20",     border: "border-warning",     text: "text-warning",     icon: Info         },
    HIGH:   { bg: "bg-destructive/20", border: "border-destructive", text: "text-destructive", icon: AlertTriangle },
  }

  const riskLabelKey: Record<"LOW" | "MEDIUM" | "HIGH", keyof TranslationDict> = {
    LOW:    "riskLevelLow",
    MEDIUM: "riskLevelMedium",
    HIGH:   "riskLevelHigh",
  }

  const s = styles[level]
  const Icon = s.icon
  const label = getTranslation(language, riskLabelKey[level])

  return (
    <motion.div
      key={language}
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 200, damping: 15 }}
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${s.bg} border ${s.border} shadow-lg`}
    >
      <Icon className={`h-5 w-5 ${s.text}`} />
      <span className={`font-semibold ${s.text}`}>{label}</span>
    </motion.div>
  )
}

/** Animated SHAP bar chart — shows real signed SHAP values or falls back to top-factors */
function ShapFactors({ factors, shapValues }: { factors: string[]; shapValues?: Record<string, number> }) {
  if (shapValues && Object.keys(shapValues).length > 0) {
    const sorted = Object.entries(shapValues)
      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
      .slice(0, 6)
    const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)))
    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2 mb-1">
          <BarChart2 className="h-4 w-4 text-primary" />
          <span className="text-sm font-semibold text-foreground">SHAP Feature Impact</span>
        </div>
        {sorted.map(([name, value]) => {
          const pct = (Math.abs(value) / maxAbs) * 100
          const isPositive = value > 0
          return (
            <motion.div
              key={name}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-0.5"
            >
              <div className="flex justify-between items-center">
                <span className="text-xs text-foreground font-medium">{name}</span>
                <span className={`text-xs font-mono font-semibold ${isPositive ? "text-red-400" : "text-green-400"}`}>
                  {isPositive ? "+" : ""}{value.toFixed(4)}
                </span>
              </div>
              <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.8 }}
                  className={`h-full rounded-full ${isPositive ? "bg-red-400" : "bg-green-400"}`}
                />
              </div>
            </motion.div>
          )
        })}
        <p className="text-[10px] text-muted-foreground mt-1">Red = increases risk · Green = decreases risk</p>
      </div>
    )
  }

  // Fallback to original bar display
  const barColors = ["bg-primary", "bg-accent", "bg-success"]
  const barWidths = ["w-full", "w-4/5", "w-3/5"]
  return (
    <div className="flex flex-col justify-center gap-3">
      <div className="flex items-center gap-2 mb-1">
        <BarChart2 className="h-4 w-4 text-primary" />
        <span className="text-sm font-semibold text-foreground">Top Risk Factors</span>
      </div>
      {factors.map((factor, i) => (
        <motion.div
          key={factor}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 + i * 0.15 }}
          className="space-y-1"
        >
          <div className="flex justify-between items-center">
            <span className="text-sm text-foreground font-medium">{factor}</span>
            <span className="text-xs text-muted-foreground">#{i + 1}</span>
          </div>
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: "100%" }}
              transition={{ duration: 0.8, delay: 0.4 + i * 0.15 }}
              className={`h-full ${barColors[i] ?? "bg-primary"} ${barWidths[i] ?? "w-1/2"} rounded-full`}
            />
          </div>
        </motion.div>
      ))}
    </div>
  )
}

/** Full result card — shown after a successful API response */
function PredictionResultCard({ result }: { result: PredictionResult }) {
  const riskColor =
    result.riskLevel === "LOW"
      ? "text-success"
      : result.riskLevel === "MEDIUM"
      ? "text-warning"
      : "text-destructive"

  const circumference = 2 * Math.PI * 56   // r = 56
  const dashOffset = circumference - (result.riskScore / 100) * circumference

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="glass-card rounded-2xl p-8 glow-border"
    >
      {/* Card header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 rounded-xl bg-primary/20">
          <Sparkles className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-foreground">AI Prediction Result</h3>
          <p className="text-sm text-muted-foreground">Powered by Federated Learning + SHAP</p>
        </div>
      </div>

      {/* Three-column metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">

        {/* 1 — Animated circular risk gauge */}
        <div className="text-center">
          <div className="relative inline-flex items-center justify-center">
            <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 128 128">
              {/* Track */}
              <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="8"
                fill="none" className="text-secondary" />
              {/* Animated fill */}
              <motion.circle
                cx="64" cy="64" r="56"
                stroke="currentColor" strokeWidth="8" fill="none"
                strokeLinecap="round"
                className={riskColor}
                strokeDasharray={circumference}
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset: dashOffset }}
                transition={{ duration: 1.5, ease: "easeOut" }}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className={`text-3xl font-bold ${riskColor}`}
              >
                {result.riskScore}%
              </motion.span>
              <span className="text-xs text-muted-foreground">Risk Score</span>
            </div>
          </div>
        </div>

        {/* 2 — SHAP top factors */}
        <ShapFactors factors={result.topFactors} shapValues={result.shapValues} />

        {/* 3 — Risk level badge */}
        <div className="flex flex-col items-center justify-center gap-3">
          <RiskBadge level={result.riskLevel} />
          <p className="text-sm text-muted-foreground text-center">
            Explained by SHAP feature attribution
          </p>
        </div>
      </div>

      {/* Recommendation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="glass rounded-xl p-6"
      >
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-info/20 flex-shrink-0">
            <Info className="h-5 w-5 text-info" />
          </div>
          <div>
            <h4 className="font-semibold text-foreground mb-2">AI Recommendation</h4>
            <p className="text-muted-foreground leading-relaxed">
              {result.recommendation}
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}

/** Error card — shown when the API call fails */
function ErrorCard({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  const isOffline =
    message.toLowerCase().includes("connect") ||
    message.toLowerCase().includes("fetch") ||
    message.toLowerCase().includes("network")

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="glass-card rounded-2xl p-8 border border-destructive/40"
    >
      <div className="flex items-start gap-4">
        <div className="p-3 rounded-xl bg-destructive/20 flex-shrink-0">
          {isOffline
            ? <WifiOff className="h-6 w-6 text-destructive" />
            : <AlertTriangle className="h-6 w-6 text-destructive" />}
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-bold text-foreground mb-1">
            {isOffline ? "Backend Offline" : "Prediction Failed"}
          </h3>
          <p className="text-sm text-muted-foreground mb-4">
            {isOffline
              ? "Cannot reach the MediShield AI backend. Make sure the FastAPI server is running and NEXT_PUBLIC_API_URL is configured correctly."
              : message}
          </p>
          {isOffline && (
            <pre className="text-xs text-muted-foreground bg-secondary/50 rounded-lg px-4 py-3 mb-4">
              uvicorn ml.api:app --host 0.0.0.0 --port 8000
            </pre>
          )}
          <button
            onClick={onDismiss}
            className="text-sm text-destructive hover:text-destructive/80 underline underline-offset-2 transition-colors"
          >
            Dismiss
          </button>
        </div>
      </div>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Main exported component
// ---------------------------------------------------------------------------

export function PredictionForm() {
  const [formData, setFormData] = useState<FormData>({
    age: 45,
    bmi: 28.5,
    glucose: 140,
    bloodPressure: 85,
    diabetesPedigreeFunction: 0.51,
    insulin: 150,
    pregnancies: 2,
    skinThickness: 25,
  })
  const [result, setResult]     = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError]       = useState<string | null>(null)
  // Clinical Support AI state
  const [clinicalData, setClinicalData]     = useState<ClinicalSupportResponse | null>(null)
  const [clinicalLoading, setClinicalLoading] = useState(false)

  const handleInputChange = (name: string, value: number) => {
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  // -------------------------------------------------------------------------
  // API call — replaces all mock/hardcoded prediction logic
  // -------------------------------------------------------------------------
  const runPrediction = async () => {
    setIsLoading(true)
    setResult(null)
    setError(null)
    setClinicalData(null)

    // Map camelCase form fields → snake_case API fields
    const requestBody = {
      pregnancies:                formData.pregnancies,
      glucose:                    formData.glucose,
      blood_pressure:             formData.bloodPressure,
      skin_thickness:             formData.skinThickness,
      insulin:                    formData.insulin,
      bmi:                        formData.bmi,
      diabetes_pedigree_function: formData.diabetesPedigreeFunction,
      age:                        formData.age,
    }

    try {
      const data = await predict(requestBody)

      // Convert 0–1 float → 0–100 integer percentage
      const riskScore = Math.round(data.risk_score * 100)

      setResult({
        riskScore,
        riskLevel:      data.risk_level,
        topFactors:     data.top_factors,
        recommendation: RECOMMENDATIONS[data.risk_level],
        shapValues:     data.shap_values_dict, // use dict for keys mapping to bars
      })

      // Persist to shared store so Explainable AI page can read it
      savePrediction({
        inputs:    requestBody,
        response:  data,
        timestamp: Date.now(),
      })

      // ── Clinical Support AI — call in background ────────────────────────
      setClinicalLoading(true)
      try {
        const clinicalPayload = {
          risk_score:        data.risk_score,
          risk_level:        data.risk_level,
          top_factors:       data.top_factors,
          shap_values_dict:  data.shap_values_dict,
          glucose:           formData.glucose,
          bmi:               formData.bmi,
          blood_pressure:    formData.bloodPressure,
          age:               formData.age,
          insulin:           formData.insulin,
          pregnancies:       formData.pregnancies,
          federated_round:   (data as any).federated_round,
          global_accuracy:   (data as any).global_accuracy,
        }
        const support = await getClinicalSupport(clinicalPayload)
        setClinicalData(support)
      } catch (clinErr) {
        console.warn("[clinical-support] Could not load recommendations:", clinErr)
      } finally {
        setClinicalLoading(false)
      }
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "An unexpected error occurred."

      // Distinguish network-offline from server errors
      const isNetworkError =
        msg.includes("Failed to fetch") ||
        msg.includes("NetworkError") ||
        msg.includes("ERR_CONNECTION_REFUSED") ||
        msg.includes("net::ERR")

      setError(
        isNetworkError
          ? "Failed to fetch — backend is offline or unreachable."
          : msg
      )
    } finally {
      setIsLoading(false)
    }
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="space-y-8">

      {/* Page header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm text-primary mb-4">
          <Activity className="h-4 w-4" />
          AI-Powered Prediction
        </span>
        <h1 className="text-3xl font-bold text-foreground mb-2">
          Disease Risk Assessment
        </h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Enter patient health metrics to receive a real-time AI risk prediction
          powered by our federated learning model with SHAP explainability.
        </p>
      </motion.div>

      {/* Input grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {inputFields.map((field, index) => (
          <motion.div
            key={field.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <InputField
              name={field.name}
              label={field.label}
              icon={field.icon}
              unit={field.unit}
              value={formData[field.name as keyof FormData]}
              onChange={handleInputChange}
              min={field.min}
              max={field.max}
            />
          </motion.div>
        ))}
      </div>

      {/* Submit button */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="flex justify-center"
      >
        <button
          onClick={runPrediction}
          disabled={isLoading}
          className="group flex items-center gap-3 px-8 py-4 rounded-xl bg-primary text-primary-foreground font-semibold hover:bg-primary/90 transition-all duration-300 glow-cyan disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <Sparkles className="h-5 w-5" />
              </motion.div>
              Running AI Prediction...
            </>
          ) : (
            <>
              <Sparkles className="h-5 w-5" />
              Run Prediction
              <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </>
          )}
        </button>
      </motion.div>

      {/* Loading animation */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glass-card rounded-2xl p-8 text-center"
          >
            <div className="flex flex-col items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-primary/30 rounded-full blur-xl animate-pulse" />
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="relative p-4 rounded-full bg-primary/20"
                >
                  <Sparkles className="h-8 w-8 text-primary" />
                </motion.div>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-foreground mb-1">
                  Analyzing Health Data
                </h3>
                <p className="text-sm text-muted-foreground">
                  Running DNN inference + SHAP explanation...
                </p>
              </div>
              <div className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    animate={{ scale: [1, 1.5, 1] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.2 }}
                    className="h-2 w-2 rounded-full bg-primary"
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error card */}
      <AnimatePresence>
        {error && !isLoading && (
          <ErrorCard message={error} onDismiss={() => setError(null)} />
        )}
      </AnimatePresence>

      {/* Real prediction result */}
      <AnimatePresence>
        {result && !isLoading && <PredictionResultCard result={result} />}
      </AnimatePresence>

      {/* ── AI Clinical Emergency Support Assistant ───────────────────────── */}
      <AnimatePresence>
        {(clinicalData || clinicalLoading) && !isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {/* Section divider */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="flex items-center gap-4"
            >
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
              <div className="flex items-center gap-2 px-4 py-1.5 rounded-full glass-card border border-primary/20">
                <Stethoscope className="h-4 w-4 text-primary" />
                <span className="text-xs font-semibold text-primary uppercase tracking-wider">
                  Clinical Recommendations
                </span>
              </div>
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
            </motion.div>

            <ClinicalSupportPanel
              data={clinicalData!}
              isLoading={clinicalLoading}
            />
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  )
}
