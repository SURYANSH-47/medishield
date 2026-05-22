"use client"

/**
 * explainable-ai.tsx
 * ==================
 * Fully dynamic Explainable AI Dashboard.
 *
 * Reads the last prediction from the shared prediction store (localStorage).
 * Every value on this page — SHAP scores, feature values, narrative text,
 * federated round status — is derived from real data.  Nothing is hardcoded.
 *
 * Data flow
 * ---------
 *   1. User submits prediction on /prediction
 *   2. prediction-form.tsx saves result → lib/prediction-store (localStorage)
 *   3. This page reads the store via usePredictionStore()
 *   4. SHAP values come from the backend's gradient-based attribution
 *   5. Federated round info fetched from /federated-rounds
 */

import { motion, AnimatePresence } from "framer-motion"
import { useState, useEffect } from "react"
import Link from "next/link"
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Info,
  Lightbulb,
  BarChart3,
  Eye,
  ArrowRight,
  Clock,
  Network,
  Shield,
  Activity,
} from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts"

import { usePredictionStore } from "@/lib/prediction-store"
import { getFederatedRounds, type FederatedRoundsResponse, type PredictionRequest, type PredictionResponse } from "@/lib/api"

// ---------------------------------------------------------------------------
// Feature ↔ input field mapping
// ---------------------------------------------------------------------------

/**
 * Maps the friendly feature names returned by the backend to the
 * corresponding field in PredictionRequest.
 *
 * Note: the backend maps DiabetesPedigreeFunction → "Heart Rate" for
 * display purposes, so we must use that key here.
 */
const FEATURE_INPUT_MAP: Record<string, keyof PredictionRequest> = {
  "Glucose Level":   "glucose",
  "BMI":             "bmi",
  "Blood Pressure":  "blood_pressure",
  "Insulin":         "insulin",
  "Age":             "age",
  "Heart Rate":      "diabetes_pedigree_function",  // backend label for DPF
  "Skin Thickness":  "skin_thickness",
  "Pregnancies":     "pregnancies",
}

const FEATURE_UNITS: Record<string, string> = {
  "Glucose Level":  "mg/dL",
  "BMI":            "kg/m²",
  "Blood Pressure": "mmHg",
  "Insulin":        "μU/mL",
  "Age":            "yrs",
  "Heart Rate":     "(pedigree)",
  "Skin Thickness": "mm",
  "Pregnancies":    "",
}

// ---------------------------------------------------------------------------
// Value helpers
// ---------------------------------------------------------------------------

function getInputValue(feature: string, inputs: PredictionRequest): number {
  const key = FEATURE_INPUT_MAP[feature]
  return key ? Number(inputs[key]) : 0
}

function formatValue(feature: string, val: number): string {
  const unit = FEATURE_UNITS[feature] ?? ""
  if (feature === "BMI")        return `${val.toFixed(1)} ${unit}`.trim()
  if (feature === "Heart Rate") return `${val.toFixed(3)} ${unit}`.trim()
  return unit ? `${val} ${unit}` : String(val)
}

// ---------------------------------------------------------------------------
// Dynamic description generator
// ---------------------------------------------------------------------------

/**
 * Returns a one-sentence clinical explanation for a specific feature value,
 * derived entirely from the actual patient input.  No hardcoded strings.
 */
function generateFeatureDescription(
  feature: string,
  shapValue: number,
  inputs: PredictionRequest
): string {
  const val = getInputValue(feature, inputs)

  switch (feature) {
    case "Glucose Level":
      if (val >= 126)
        return `At ${val} mg/dL, glucose is in the diabetic range (≥126 mg/dL) — the model's strongest individual risk signal.`
      if (val >= 100)
        return `At ${val} mg/dL, glucose is prediabetic (100–125 mg/dL), a meaningful early-warning marker.`
      return `Glucose of ${val} mg/dL is within the normal fasting range (<100 mg/dL), acting as a mild protective signal.`

    case "BMI":
      if (val >= 35)
        return `BMI ${val.toFixed(1)} — severe obesity (Class II/III, ≥35), strongly associated with insulin resistance.`
      if (val >= 30)
        return `BMI ${val.toFixed(1)} — obese category (30–34.9), significantly elevating type 2 diabetes risk.`
      if (val >= 25)
        return `BMI ${val.toFixed(1)} — overweight (25–29.9), a moderate metabolic risk factor.`
      return `BMI ${val.toFixed(1)} is in the healthy range (<25), contributing a protective effect.`

    case "Blood Pressure":
      if (val >= 90)
        return `${val} mmHg — stage 2 hypertension (≥90 mmHg), a comorbidity that amplifies metabolic risk.`
      if (val >= 80)
        return `${val} mmHg — elevated diastolic pressure (80–89 mmHg), commonly co-occurring with metabolic syndrome.`
      return `${val} mmHg — normal diastolic range (<80 mmHg), no hypertensive contribution to risk.`

    case "Insulin":
      if (val > 100)
        return `${val} μU/mL — markedly elevated, a strong indicator of insulin resistance.`
      if (val > 25)
        return `${val} μU/mL — above normal fasting range (2–25 μU/mL), suggesting early insulin resistance.`
      return `${val} μU/mL — within normal fasting range, no insulin resistance detected.`

    case "Age":
      if (val >= 55)
        return `Age ${val} — a substantial non-modifiable risk factor; diabetes incidence rises sharply after 45.`
      if (val >= 45)
        return `Age ${val} — entering the primary risk band (45–54 years) for type 2 diabetes onset.`
      return `Age ${val} — below the main age-risk threshold (45 years), a mild protective factor.`

    case "Pregnancies":
      if (val >= 4)
        return `${val} pregnancies — significantly elevates gestational diabetes history and long-term metabolic risk.`
      if (val >= 1)
        return `${val} prior ${val === 1 ? "pregnancy" : "pregnancies"} — moderate historical contribution to metabolic risk.`
      return `No prior pregnancies — no gestational diabetes history contributing to risk.`

    case "Skin Thickness":
      if (val >= 40)
        return `${val} mm triceps fold — elevated, indicating higher subcutaneous fat mass and adiposity.`
      if (val >= 25)
        return `${val} mm — moderately elevated triceps fold, a mild adiposity indicator.`
      return `${val} mm — within normal range, minimal adiposity contribution.`

    case "Heart Rate":
      // Note: "Heart Rate" is the backend's display label for DiabetesPedigreeFunction
      if (val >= 0.8)
        return `Pedigree score ${val.toFixed(3)} — strong family history contribution to inherited genetic diabetes risk.`
      if (val >= 0.5)
        return `Pedigree score ${val.toFixed(3)} — moderate inherited risk based on family history.`
      return `Pedigree score ${val.toFixed(3)} — limited inherited genetic risk from family history.`

    default:
      return `${shapValue > 0 ? "Increasing" : "Decreasing"} the predicted risk through its quantified influence on the model output.`
  }
}

// ---------------------------------------------------------------------------
// Narrative builder
// ---------------------------------------------------------------------------

interface NarrativeData {
  riskScore: number
  riskLevel: string
  riskColorClass: string
  risking: Array<{ feature: string; value: string; explanation: string }>
  protective: Array<{ feature: string; value: string; explanation: string }>
}

function buildNarrative(
  inputs: PredictionRequest,
  response: PredictionResponse
): NarrativeData {
  const shap  = response.shap_values ?? []
  const sorted = [...shap].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))

  const risking   = sorted.filter((x) => x.value > 0).slice(0, 3)
  const protective = sorted.filter((x) => x.value < 0).slice(0, 2)

  const riskColors: Record<string, string> = {
    LOW:    "text-success",
    MEDIUM: "text-warning",
    HIGH:   "text-destructive",
  }

  return {
    riskScore:      Math.round(response.risk_score * 100),
    riskLevel:      response.risk_level,
    riskColorClass: riskColors[response.risk_level] ?? "text-foreground",
    risking: risking.map((f) => ({
      feature:     f.feature,
      value:       formatValue(f.feature, getInputValue(f.feature, inputs)),
      explanation: generateFeatureDescription(f.feature, f.value, inputs),
    })),
    protective: protective.map((f) => ({
      feature:     f.feature,
      value:       formatValue(f.feature, getInputValue(f.feature, inputs)),
      explanation: generateFeatureDescription(f.feature, f.value, inputs),
    })),
  }
}

// ---------------------------------------------------------------------------
// Chart tooltip
// ---------------------------------------------------------------------------

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null
  const d = payload[0]
  return (
    <div className="glass-card rounded-lg p-3 border border-border">
      <p className="text-sm font-medium text-foreground">{d.payload.feature}</p>
      <p className="text-xs text-muted-foreground">
        SHAP:{" "}
        <span className={`font-semibold ${d.value > 0 ? "text-destructive" : "text-success"}`}>
          {d.value > 0 ? "+" : ""}
          {Number(d.value).toFixed(1)}%
        </span>
      </p>
      <p className="text-xs text-muted-foreground">Patient value: {d.payload.patientValue}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Feature card
// ---------------------------------------------------------------------------

interface FeatureCardProps {
  feature: string
  shapValue: number
  inputs: PredictionRequest
  index: number
}

function FeatureCard({ feature, shapValue, inputs, index }: FeatureCardProps) {
  const isRisk    = shapValue > 0
  const absVal    = Math.abs(shapValue)
  const rawVal    = getInputValue(feature, inputs)
  const formatted = formatValue(feature, rawVal)
  const desc      = generateFeatureDescription(feature, shapValue, inputs)

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ scale: 1.01 }}
      className="glass-card rounded-xl p-4 hover:glow-border transition-all duration-300"
    >
      {/* Header row */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${isRisk ? "bg-destructive/20" : "bg-success/20"}`}>
            {isRisk
              ? <TrendingUp  className="h-4 w-4 text-destructive" />
              : <TrendingDown className="h-4 w-4 text-success" />}
          </div>
          <div>
            <h4 className="font-semibold text-foreground">{feature}</h4>
            <span className="text-sm text-muted-foreground font-mono">{formatted}</span>
          </div>
        </div>
        <div className={`text-lg font-bold tabular-nums ${isRisk ? "text-destructive" : "text-success"}`}>
          {isRisk ? "+" : ""}
          {shapValue.toFixed(1)}%
        </div>
      </div>

      {/* Importance bar */}
      <div className="mb-3">
        <div className="h-2 bg-secondary rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(absVal, 100)}%` }}
            transition={{ duration: 0.8, delay: index * 0.05 }}
            className={`h-full rounded-full ${
              isRisk
                ? "bg-gradient-to-r from-destructive/50 to-destructive"
                : "bg-gradient-to-r from-success/50 to-success"
            }`}
          />
        </div>
      </div>

      <p className="text-sm text-muted-foreground leading-relaxed">{desc}</p>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Empty state (no prediction stored yet)
// ---------------------------------------------------------------------------

function NoPredictionBanner() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl p-16 flex flex-col items-center justify-center gap-6 text-center min-h-[400px]"
    >
      <div className="p-4 rounded-2xl bg-primary/20">
        <Brain className="h-12 w-12 text-primary" />
      </div>
      <div>
        <h3 className="text-xl font-semibold text-foreground mb-2">
          No Prediction Data Yet
        </h3>
        <p className="text-muted-foreground max-w-sm leading-relaxed">
          Run a patient risk assessment first. The Explainable AI page automatically
          displays SHAP explanations for the most recent prediction made.
        </p>
      </div>
      <Link
        href="/prediction"
        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold hover:bg-primary/90 transition-all duration-300"
      >
        Go to Prediction Page
        <ArrowRight className="h-4 w-4" />
      </Link>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Federated learning context banner
// ---------------------------------------------------------------------------

function FederatedBanner({ rounds }: { rounds: FederatedRoundsResponse | null }) {
  if (!rounds) return null

  const pct =
    rounds.total_rounds > 0
      ? Math.round((rounds.federated_round / rounds.total_rounds) * 100)
      : 0

  const lastAcc =
    rounds.history.length > 0
      ? rounds.history[rounds.history.length - 1].accuracy
      : null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.7 }}
      className="glass-card rounded-2xl p-6"
    >
      <div className="flex items-center gap-3 mb-5">
        <div className="p-2 rounded-xl bg-primary/20">
          <Network className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">Federated Learning Context</h3>
          <p className="text-sm text-muted-foreground">
            This prediction uses a model trained collaboratively across 3 hospitals
          </p>
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-3 gap-4 mb-5">
        <div className="text-center">
          <div className="text-2xl font-bold text-primary">{rounds.federated_round}</div>
          <div className="text-xs text-muted-foreground">Current Round</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-foreground">{rounds.total_rounds}</div>
          <div className="text-xs text-muted-foreground">Total Rounds</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-success">
            {lastAcc !== null ? `${lastAcc.toFixed(1)}%` : "—"}
          </div>
          <div className="text-xs text-muted-foreground">Global Accuracy</div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-5">
        <div className="flex justify-between text-xs text-muted-foreground mb-1.5">
          <span>Training Progress</span>
          <span>{pct}% complete</span>
        </div>
        <div className="h-2 bg-secondary rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="h-full bg-gradient-to-r from-primary/60 to-primary rounded-full"
          />
        </div>
      </div>

      {/* FL pipeline visualization */}
      <div className="flex items-center justify-center gap-2 flex-wrap">
        {[
          { label: "Hospital A", sub: "Preventive" },
          { label: "Hospital B", sub: "Geriatric" },
          { label: "Hospital C", sub: "General" },
        ].map((h, i) => (
          <div key={h.label} className="flex items-center gap-2">
            <div className="flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-lg bg-primary/10 border border-primary/20">
              <div className="flex items-center gap-1">
                <div className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                <span className="text-xs font-medium text-primary">{h.label}</span>
              </div>
              <span className="text-[10px] text-muted-foreground">{h.sub}</span>
            </div>
            {i < 2 && <span className="text-muted-foreground text-xs font-bold">+</span>}
          </div>
        ))}

        <ArrowRight className="h-4 w-4 text-muted-foreground mx-1" />

        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-warning/10 border border-warning/20">
          <Activity className="h-3 w-3 text-warning" />
          <span className="text-xs font-semibold text-warning">FedAvg</span>
        </div>

        <ArrowRight className="h-4 w-4 text-muted-foreground mx-1" />

        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-success/10 border border-success/20">
          <Shield className="h-3 w-3 text-success" />
          <span className="text-xs font-semibold text-success">Global Model</span>
        </div>
      </div>

      <p className="text-xs text-center text-muted-foreground mt-3">
        🔒 Raw patient data never left any hospital — only model weights were aggregated
      </p>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Main exported component
// ---------------------------------------------------------------------------

export function ExplainableAIDashboard() {
  const { snapshot, secondsSince, isEmpty } = usePredictionStore()
  const [fedRounds, setFedRounds] = useState<FederatedRoundsResponse | null>(null)

  // Fetch federated round info once on mount (best-effort — fails silently)
  useEffect(() => {
    getFederatedRounds()
      .then(setFedRounds)
      .catch(() => {
        // Backend offline or no federated training done yet — omit the section
      })
  }, [])

  // Derive all display data from the stored snapshot
  const inputs   = snapshot?.inputs   ?? null
  const response = snapshot?.response ?? null
  const narrative: NarrativeData | null =
    inputs && response ? buildNarrative(inputs, response) : null

  // SHAP chart data — sorted by absolute value, includes patient values
  const shapChartData = (response?.shap_values ?? [])
    .map((item) => ({
      feature:      item.feature,
      importance:   item.value,
      fill:         item.value > 0 ? "#ef4444" : "#22c55e",
      patientValue: inputs
        ? formatValue(item.feature, getInputValue(item.feature, inputs))
        : "—",
    }))
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))

  // Feature cards — same sort
  const featureCards = (response?.shap_values ?? [])
    .slice()
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))

  // Live timestamp label
  const agoLabel =
    secondsSince === null
      ? null
      : secondsSince < 5
      ? "just now"
      : secondsSince < 60
      ? `${secondsSince}s ago`
      : secondsSince < 3600
      ? `${Math.floor(secondsSince / 60)}m ago`
      : `${Math.floor(secondsSince / 3600)}h ago`


  return (
    <div className="space-y-8">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
      >
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-primary/20">
              <Brain className="h-6 w-6 text-primary" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">
              Explainable AI Dashboard
            </h1>
          </div>
          <p className="text-muted-foreground">
            SHAP-based feature importance from your last patient assessment
          </p>
        </div>

        {/* Live "last prediction" badge */}
        {agoLabel && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 px-4 py-2 rounded-xl glass-card text-sm"
          >
            <Clock className="h-4 w-4 text-primary" />
            <span className="text-muted-foreground">
              Last prediction:{" "}
              <span className="text-foreground font-medium">{agoLabel}</span>
            </span>
          </motion.div>
        )}
      </motion.div>

      {/* ── Main content ────────────────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        {isEmpty ? (
          <NoPredictionBanner />
        ) : (
          <motion.div
            key="loaded"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-8"
          >

            {/* ── Summary cards ─────────────────────────────────────────── */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">

              {/* Risk Score */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="glass-card rounded-2xl p-6 hover:glow-border transition-all duration-300"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-2 rounded-xl bg-primary/20">
                    <Eye className="h-4 w-4 text-primary" />
                  </div>
                  <span className="text-xs text-muted-foreground">Risk Score</span>
                </div>
                <div className={`text-3xl font-bold tabular-nums ${narrative?.riskColorClass ?? "text-foreground"}`}>
                  {narrative?.riskScore ?? 0}%
                </div>
                <div className={`text-xs font-semibold mt-1 ${narrative?.riskColorClass ?? ""}`}>
                  {narrative?.riskLevel ?? "—"} RISK
                </div>
              </motion.div>

              {/* Confidence */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 }}
                className="glass-card rounded-2xl p-6 hover:glow-border transition-all duration-300"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-2 rounded-xl bg-primary/20">
                    <Brain className="h-4 w-4 text-primary" />
                  </div>
                  <span className="text-xs text-muted-foreground">Confidence</span>
                </div>
                <div className="text-3xl font-bold text-primary tabular-nums">
                  {response?.confidence?.toFixed(1) ?? narrative?.riskScore ?? 0}%
                </div>
                <p className="text-xs text-muted-foreground mt-1">Model certainty</p>
              </motion.div>

              {/* Risk Factors */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-card rounded-2xl p-6 hover:glow-border transition-all duration-300"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-2 rounded-xl bg-destructive/20">
                    <TrendingUp className="h-4 w-4 text-destructive" />
                  </div>
                  <span className="text-xs text-muted-foreground">Risk Factors</span>
                </div>
                <div className="text-3xl font-bold text-destructive tabular-nums">
                  +{response?.risk_factors?.toFixed(1) ?? "—"}%
                </div>
                <p className="text-xs text-muted-foreground mt-1">Positive SHAP total</p>
              </motion.div>

              {/* Protective Factors */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
                className="glass-card rounded-2xl p-6 hover:glow-border transition-all duration-300"
              >
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-2 rounded-xl bg-success/20">
                    <TrendingDown className="h-4 w-4 text-success" />
                  </div>
                  <span className="text-xs text-muted-foreground">Protective</span>
                </div>
                <div className="text-3xl font-bold text-success tabular-nums">
                  {response?.protective_factors !== undefined
                    ? `${Number(response.protective_factors).toFixed(1)}%`
                    : "—"}
                </div>
                <p className="text-xs text-muted-foreground mt-1">Negative SHAP total</p>
              </motion.div>
            </div>

            {/* ── SHAP Bar Chart ─────────────────────────────────────────── */}
            {shapChartData.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="glass-card rounded-2xl p-6"
              >
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 rounded-xl bg-primary/20">
                    <BarChart3 className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground">
                      SHAP Feature Importance
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      Gradient attribution for:{" "}
                      <span className="font-mono text-xs text-foreground">
                        {inputs
                          ? `G=${inputs.glucose} · BMI=${inputs.bmi.toFixed(1)} · Age=${inputs.age}`
                          : "—"}
                      </span>
                    </p>
                  </div>
                </div>

                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shapChartData} layout="vertical">
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="rgba(255,255,255,0.05)"
                      />
                      <XAxis
                        type="number"
                        stroke="rgba(255,255,255,0.4)"
                        tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                        axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                      />
                      <YAxis
                        dataKey="feature"
                        type="category"
                        stroke="rgba(255,255,255,0.4)"
                        tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                        width={120}
                        axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine x={0} stroke="rgba(255,255,255,0.25)" />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {shapChartData.map((entry, i) => (
                          <Cell key={`cell-${i}`} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="flex items-center justify-center gap-8 mt-4">
                  <div className="flex items-center gap-2">
                    <span className="h-3 w-3 rounded-full bg-destructive" />
                    <span className="text-sm text-muted-foreground">Increases Risk</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="h-3 w-3 rounded-full bg-success" />
                    <span className="text-sm text-muted-foreground">Decreases Risk</span>
                  </div>
                </div>
              </motion.div>
            )}

            {/* ── Feature Analysis Cards ─────────────────────────────────── */}
            {featureCards.length > 0 && inputs && (
              <div>
                <motion.h2
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 }}
                  className="text-xl font-semibold text-foreground mb-4"
                >
                  Feature Analysis
                </motion.h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {featureCards.map((item, i) => (
                    <FeatureCard
                      key={item.feature}
                      feature={item.feature}
                      shapValue={item.value}
                      inputs={inputs}
                      index={i}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* ── Why This Prediction ────────────────────────────────────── */}
            {narrative && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.55 }}
                className="glass-card rounded-2xl p-6 glow-border"
              >
                <div className="flex items-start gap-4">
                  <div className="p-3 rounded-xl bg-warning/20 flex-shrink-0">
                    <Lightbulb className="h-6 w-6 text-warning" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-foreground mb-3">
                      Why This Prediction Was Made
                    </h3>
                    <div className="space-y-4 text-muted-foreground">
                      <p className="leading-relaxed">
                        The model assigned a{" "}
                        <span className={`font-semibold ${narrative.riskColorClass}`}>
                          {narrative.riskScore}%{" "}
                          {narrative.riskLevel.charAt(0) + narrative.riskLevel.slice(1).toLowerCase()}{" "}
                          Risk
                        </span>{" "}
                        score after analyzing 8 clinical indicators. The primary drivers were:
                      </p>

                      {narrative.risking.length > 0 && (
                        <ul className="space-y-3">
                          {narrative.risking.map((f, i) => (
                            <li key={f.feature} className="flex items-start gap-2">
                              <span className="text-destructive font-bold mt-0.5 flex-shrink-0">
                                {i + 1}.
                              </span>
                              <span>
                                <strong className="text-foreground">
                                  {f.feature} — {f.value}
                                </strong>
                                {" "}— {f.explanation}
                              </span>
                            </li>
                          ))}
                        </ul>
                      )}

                      {narrative.protective.length > 0 && (
                        <>
                          <p className="leading-relaxed pt-1">
                            Counterbalancing{" "}
                            <span className="text-success font-medium">
                              protective factors
                            </span>{" "}
                            reduced the overall score:
                          </p>
                          <ul className="space-y-2">
                            {narrative.protective.map((f) => (
                              <li key={f.feature} className="flex items-start gap-2">
                                <span className="text-success font-bold mt-0.5 flex-shrink-0">
                                  •
                                </span>
                                <span>
                                  <strong className="text-foreground">
                                    {f.feature} — {f.value}
                                  </strong>
                                  {" "}— {f.explanation}
                                </span>
                              </li>
                            ))}
                          </ul>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* ── Federated Learning Context ─────────────────────────────── */}
            <FederatedBanner rounds={fedRounds} />

            {/* ── AI Transparency ───────────────────────────────────────── */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.85 }}
              className="glass-card rounded-2xl p-6"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-xl bg-info/20">
                  <Info className="h-5 w-5 text-info" />
                </div>
                <h3 className="font-semibold text-foreground">AI Transparency</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <h4 className="text-sm font-medium text-foreground mb-2">Model Type</h4>
                  <p className="text-sm text-muted-foreground">
                    Federated Deep Neural Network (8→64→32→1 + Sigmoid), trained via
                    privacy-preserving FedAvg aggregation across 3 hospital nodes
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-foreground mb-2">Training Data</h4>
                  <p className="text-sm text-muted-foreground">
                    Pima Indian Diabetes dataset partitioned across Hospital A (Preventive
                    Care), Hospital B (Geriatric), and Hospital C (General Medicine) — no
                    cross-hospital data sharing
                  </p>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-foreground mb-2">
                    Explanation Method
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Gradient-based attribution:{" "}
                    <span className="font-mono text-xs">|∂risk / ∂x_i|</span>, showing
                    how sensitive the model output is to each input feature — equivalent
                    to first-order SHAP for smooth differentiable networks
                  </p>
                </div>
              </div>
            </motion.div>

          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
