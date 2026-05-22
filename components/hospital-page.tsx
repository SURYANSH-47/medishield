"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import Link from "next/link"
import { toast } from "sonner"
import {
  ArrowLeft,
  Building2,
  Users,
  Activity,
  Clock,
  RefreshCw,
  Plus,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  AlertTriangle,
  Cpu,
  Database,
  Loader2,
  Stethoscope,
  TrendingUp,
} from "lucide-react"
import {
  getHospitalStats,
  addHospitalPatient,
  retrainHospital,
  getHospitalDatasetPreview,
  type HospitalStats,
  type HospitalPatientRequest,
  type DatasetPreviewResponse,
} from "@/lib/api"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HOSPITAL_CONFIG = {
  a: {
    label: "Hospital Alpha",
    color: "cyan",
    colorClass: "text-cyan-400",
    borderClass: "border-cyan-400/40",
    bgClass: "bg-cyan-400/10",
    glowClass: "shadow-cyan-400/20",
    barClass: "bg-cyan-400",
  },
  b: {
    label: "Hospital Beta",
    color: "teal",
    colorClass: "text-teal-400",
    borderClass: "border-teal-400/40",
    bgClass: "bg-teal-400/10",
    glowClass: "shadow-teal-400/20",
    barClass: "bg-teal-400",
  },
  c: {
    label: "Hospital Gamma",
    color: "indigo",
    colorClass: "text-indigo-400",
    borderClass: "border-indigo-400/40",
    bgClass: "bg-indigo-400/10",
    glowClass: "shadow-indigo-400/20",
    barClass: "bg-indigo-400",
  },
} as const

type HospitalId = keyof typeof HOSPITAL_CONFIG

const INITIAL_FORM: HospitalPatientRequest = {
  pregnancies: 0,
  glucose: 0,
  blood_pressure: 0,
  skin_thickness: 0,
  insulin: 0,
  bmi: 0,
  diabetes_pedigree_function: 0,
  age: 0,
  outcome: 0,
}

const FORM_FIELDS = [
  { key: "pregnancies" as const,               label: "Pregnancies",                   unit: "",        min: 0, max: 20,  step: 1,    placeholder: "0–20" },
  { key: "glucose" as const,                   label: "Glucose (mg/dL)",               unit: "mg/dL",   min: 0, max: 300, step: 1,    placeholder: "0–300" },
  { key: "blood_pressure" as const,            label: "Blood Pressure (mmHg)",         unit: "mmHg",    min: 0, max: 200, step: 1,    placeholder: "0–200" },
  { key: "skin_thickness" as const,            label: "Skin Thickness (mm)",           unit: "mm",      min: 0, max: 100, step: 1,    placeholder: "0–100" },
  { key: "insulin" as const,                   label: "Insulin (μU/mL)",               unit: "μU/mL",   min: 0, max: 900, step: 1,    placeholder: "0–900" },
  { key: "bmi" as const,                       label: "BMI (kg/m²)",                   unit: "kg/m²",   min: 0, max: 70,  step: 0.1,  placeholder: "0–70" },
  { key: "diabetes_pedigree_function" as const, label: "Diabetes Pedigree Function",   unit: "",        min: 0, max: 3,   step: 0.01, placeholder: "0–3" },
  { key: "age" as const,                       label: "Age (years)",                   unit: "yrs",     min: 0, max: 120, step: 1,    placeholder: "0–120" },
]

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  colorClass,
  bgClass,
}: {
  icon: React.ElementType
  label: string
  value: string
  sub?: string
  colorClass: string
  bgClass: string
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl p-5 border border-border"
    >
      <div className={`inline-flex p-2.5 rounded-xl ${bgClass} mb-3`}>
        <Icon className={`h-5 w-5 ${colorClass}`} />
      </div>
      <div className="text-2xl font-bold text-foreground leading-none mb-1">{value}</div>
      <div className="text-sm text-muted-foreground">{label}</div>
      {sub && <div className="text-xs text-muted-foreground/60 mt-1">{sub}</div>}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function HospitalPage({ hospitalId }: { hospitalId: string }) {
  const id = hospitalId as HospitalId
  const cfg = HOSPITAL_CONFIG[id] ?? HOSPITAL_CONFIG.a

  // ── State ────────────────────────────────────────────────────────────────
  const [stats, setStats] = useState<HospitalStats | null>(null)
  const [statsLoading, setStatsLoading] = useState(true)

  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null)
  const [previewPage, setPreviewPage] = useState(1)
  const [previewLoading, setPreviewLoading] = useState(false)

  const [form, setForm] = useState<HospitalPatientRequest>(INITIAL_FORM)
  const [formErrors, setFormErrors] = useState<Partial<Record<keyof HospitalPatientRequest, string>>>({})
  const [isAddingPatient, setIsAddingPatient] = useState(false)

  const [isRetraining, setIsRetraining] = useState(false)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Data fetching ────────────────────────────────────────────────────────

  const fetchStats = useCallback(async (silent = false) => {
    if (!silent) setStatsLoading(true)
    try {
      const data = await getHospitalStats(id)
      setStats(data)
      setIsRetraining(data.is_training)
    } catch (err) {
      if (!silent) toast.error("Could not load hospital stats. Is the backend running?")
    } finally {
      if (!silent) setStatsLoading(false)
    }
  }, [id])

  const fetchPreview = useCallback(async (page: number) => {
    setPreviewLoading(true)
    try {
      const data = await getHospitalDatasetPreview(id, page, 10)
      setPreview(data)
    } catch {
      toast.error("Failed to load dataset preview.")
    } finally {
      setPreviewLoading(false)
    }
  }, [id])

  // ── Polling ──────────────────────────────────────────────────────────────

  useEffect(() => {
    fetchStats(false)
    fetchPreview(1)
  }, [fetchStats, fetchPreview])

  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    const interval = isRetraining ? 3_000 : 10_000
    pollRef.current = setInterval(() => fetchStats(true), interval)
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [isRetraining, fetchStats])

  // Refresh preview after retraining completes
  const prevRetraining = useRef(isRetraining)
  useEffect(() => {
    if (prevRetraining.current && !isRetraining) {
      // Training just finished — reload preview to show new data
      fetchPreview(previewPage)
    }
    prevRetraining.current = isRetraining
  }, [isRetraining, fetchPreview, previewPage])

  // ── Form helpers ─────────────────────────────────────────────────────────

  const validate = (): boolean => {
    const errors: Partial<Record<keyof HospitalPatientRequest, string>> = {}

    for (const f of FORM_FIELDS) {
      const val = form[f.key] as number
      if (isNaN(val)) { errors[f.key] = "Required"; continue }
      if (val < f.min) errors[f.key] = `Min ${f.min}`
      if (val > f.max) errors[f.key] = `Max ${f.max}`
    }
    if (form.outcome !== 0 && form.outcome !== 1) {
      errors.outcome = "Must be 0 or 1"
    }

    setFormErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleFormChange = (key: keyof HospitalPatientRequest, rawValue: string) => {
    const parsed = key === "bmi" || key === "diabetes_pedigree_function"
      ? parseFloat(rawValue)
      : parseInt(rawValue, 10)
    setForm((prev) => ({ ...prev, [key]: isNaN(parsed) ? 0 : parsed }))
    setFormErrors((prev) => { const n = { ...prev }; delete n[key]; return n })
  }

  const handleAddPatient = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!validate()) return

    setIsAddingPatient(true)
    try {
      const result = await addHospitalPatient(id, form)
      toast.success(`Patient added successfully. Total: ${result.new_patient_count} patients.`)
      setForm(INITIAL_FORM)
      setFormErrors({})
      // Reload both stats and preview to reflect the new patient
      await Promise.all([fetchStats(true), fetchPreview(previewPage)])
    } catch (err: any) {
      toast.error(err.message || "Failed to add patient.")
    } finally {
      setIsAddingPatient(false)
    }
  }

  const handleRetrain = async () => {
    if (isRetraining) return
    try {
      await retrainHospital(id)
      setIsRetraining(true)
      toast.info("Local training started. This will take a few seconds…", { duration: 4000 })
    } catch (err: any) {
      toast.error(err.message || "Failed to start retraining.")
    }
  }

  const handlePageChange = (page: number) => {
    setPreviewPage(page)
    fetchPreview(page)
  }

  // ── Render ───────────────────────────────────────────────────────────────

  if (statsLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh]">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          className={`w-14 h-14 rounded-full border-2 border-t-transparent ${cfg.borderClass} mb-4`}
        />
        <p className="text-muted-foreground text-sm animate-pulse">
          Connecting to {cfg.label}…
        </p>
      </div>
    )
  }

  const accuracy = stats?.local_accuracy ?? null
  const patientCount = stats?.patient_count ?? 0
  const lastTrained = stats?.last_trained
    ? new Date(stats.last_trained).toLocaleString()
    : "Never trained"

  return (
    <div className="space-y-8 max-w-6xl">
      {/* ── Back nav + Header ─────────────────────────────────────────── */}
      <div>
        <Link
          href="/dashboard"
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors mb-5"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Dashboard
        </Link>

        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
        >
          <div className="flex items-center gap-4">
            <div className={`p-3 rounded-2xl ${cfg.bgClass} border ${cfg.borderClass}`}>
              <Building2 className={`h-8 w-8 ${cfg.colorClass}`} />
            </div>
            <div>
              <h1 className={`text-2xl font-bold ${cfg.colorClass}`}>
                {stats?.full_name ?? cfg.label}
              </h1>
              <p className="text-sm text-muted-foreground mt-0.5">
                {stats?.specialty ?? "Medical Center"} · Node ID:{" "}
                <code className="text-xs bg-secondary px-1.5 py-0.5 rounded font-mono">
                  hospital_{id}
                </code>
              </p>
            </div>
          </div>

          {/* Training status badge */}
          <div
            className={`flex items-center gap-2 px-4 py-2 rounded-xl border text-sm font-medium ${
              isRetraining
                ? "border-yellow-500/40 bg-yellow-500/10 text-yellow-400"
                : accuracy !== null
                ? "border-green-500/40 bg-green-500/10 text-green-400"
                : "border-border bg-secondary text-muted-foreground"
            }`}
          >
            {isRetraining ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Training in progress…
              </>
            ) : accuracy !== null ? (
              <>
                <CheckCircle className="h-4 w-4" />
                Model Ready
              </>
            ) : (
              <>
                <AlertTriangle className="h-4 w-4" />
                Not yet trained
              </>
            )}
          </div>
        </motion.div>
      </div>

      {/* ── Stats row ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Activity}
          label="Local Accuracy"
          value={accuracy !== null ? `${accuracy.toFixed(1)}%` : "—"}
          sub={isRetraining ? "Updating…" : undefined}
          colorClass={cfg.colorClass}
          bgClass={cfg.bgClass}
        />
        <StatCard
          icon={Users}
          label="Total Patients"
          value={patientCount.toLocaleString()}
          colorClass={cfg.colorClass}
          bgClass={cfg.bgClass}
        />
        <StatCard
          icon={Clock}
          label="Last Trained"
          value={lastTrained === "Never trained" ? "—" : lastTrained.split(",")[0]}
          sub={lastTrained === "Never trained" ? "Never trained" : lastTrained.split(",")[1]?.trim()}
          colorClass={cfg.colorClass}
          bgClass={cfg.bgClass}
        />
        <StatCard
          icon={TrendingUp}
          label="Train Loss"
          value={stats?.train_loss !== null && stats?.train_loss !== undefined
            ? stats.train_loss.toFixed(4)
            : "—"}
          colorClass={cfg.colorClass}
          bgClass={cfg.bgClass}
        />
      </div>

      {/* ── Two-column layout: Add Patient (left) + Retrain (right) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Add Patient form */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-2 glass-card rounded-2xl p-6 border border-border"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className={`p-2 rounded-xl ${cfg.bgClass}`}>
              <Plus className={`h-5 w-5 ${cfg.colorClass}`} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-foreground">Add Patient</h2>
              <p className="text-xs text-muted-foreground">
                Data stays local — only model weights are federated.
              </p>
            </div>
          </div>

          <form onSubmit={handleAddPatient} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {FORM_FIELDS.map((f) => (
                <div key={f.key}>
                  <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                    {f.label}
                  </label>
                  <input
                    type="number"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    placeholder={f.placeholder}
                    value={form[f.key] === 0 ? "" : String(form[f.key])}
                    onChange={(e) => handleFormChange(f.key, e.target.value)}
                    className={`w-full bg-secondary border rounded-lg px-3 py-2.5 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 transition-all ${
                      formErrors[f.key]
                        ? "border-destructive focus:ring-destructive/40"
                        : `border-border focus:ring-${cfg.color}-400/40 focus:${cfg.borderClass}`
                    }`}
                  />
                  {formErrors[f.key] && (
                    <p className="text-xs text-destructive mt-1">{formErrors[f.key]}</p>
                  )}
                </div>
              ))}
            </div>

            {/* Outcome field — special toggle */}
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                Outcome (Diagnosis)
              </label>
              <div className="flex gap-3">
                {[
                  { value: 0, label: "0 — Healthy", classes: "border-green-500/40 bg-green-500/10 text-green-400" },
                  { value: 1, label: "1 — Diabetic", classes: "border-red-500/40 bg-red-500/10 text-red-400" },
                ].map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setForm((p) => ({ ...p, outcome: opt.value }))}
                    className={`flex-1 py-2.5 rounded-lg border text-sm font-medium transition-all ${
                      form.outcome === opt.value
                        ? opt.classes
                        : "border-border text-muted-foreground bg-secondary hover:border-border/80"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              {formErrors.outcome && (
                <p className="text-xs text-destructive mt-1">{formErrors.outcome}</p>
              )}
            </div>

            <button
              type="submit"
              disabled={isAddingPatient}
              className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm transition-all ${cfg.bgClass} border ${cfg.borderClass} ${cfg.colorClass} hover:opacity-80 disabled:opacity-50`}
            >
              {isAddingPatient ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Adding Patient…
                </>
              ) : (
                <>
                  <Plus className="h-4 w-4" />
                  Add Patient Record
                </>
              )}
            </button>
          </form>
        </motion.div>

        {/* Retrain panel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.15 }}
          className="glass-card rounded-2xl p-6 border border-border flex flex-col"
        >
          <div className="flex items-center gap-3 mb-5">
            <div className={`p-2 rounded-xl ${cfg.bgClass}`}>
              <Cpu className={`h-5 w-5 ${cfg.colorClass}`} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-foreground">Local Training</h2>
              <p className="text-xs text-muted-foreground">
                Train on this hospital's data only.
              </p>
            </div>
          </div>

          <div className="flex-1 space-y-4">
            {/* Accuracy bar */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-muted-foreground">Local Accuracy</span>
                <span className={`font-semibold ${cfg.colorClass}`}>
                  {accuracy !== null ? `${accuracy.toFixed(1)}%` : "—"}
                </span>
              </div>
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: accuracy !== null ? `${accuracy}%` : "0%" }}
                  transition={{ duration: 1.2, ease: "easeOut" }}
                  className={`h-full rounded-full ${cfg.barClass}`}
                />
              </div>
            </div>

            {/* Info grid */}
            <div className="space-y-2 text-sm">
              {[
                { label: "Patients", value: patientCount.toLocaleString() },
                { label: "Last trained", value: lastTrained === "Never trained" ? "—" : lastTrained },
                { label: "Train loss", value: stats?.train_loss !== null && stats?.train_loss !== undefined ? stats.train_loss.toFixed(4) : "—" },
              ].map((item) => (
                <div key={item.label} className="flex justify-between p-2.5 rounded-lg bg-secondary/40">
                  <span className="text-muted-foreground">{item.label}</span>
                  <span className="font-medium text-foreground truncate max-w-[60%] text-right">{item.value}</span>
                </div>
              ))}
            </div>

            {/* Privacy note */}
            <div className={`rounded-xl p-3 text-xs ${cfg.bgClass} ${cfg.colorClass} border ${cfg.borderClass}`}>
              <Stethoscope className="h-3.5 w-3.5 inline mr-1.5 mb-0.5" />
              Only gradient weights leave this node — patient records stay local.
            </div>
          </div>

          <button
            onClick={handleRetrain}
            disabled={isRetraining || patientCount < 5}
            className={`mt-5 w-full flex items-center justify-center gap-2 py-3 rounded-xl font-semibold text-sm transition-all disabled:opacity-50 ${
              isRetraining
                ? "bg-yellow-500/20 border border-yellow-500/40 text-yellow-400"
                : `${cfg.bgClass} border ${cfg.borderClass} ${cfg.colorClass} hover:opacity-80`
            }`}
          >
            {isRetraining ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                Training…
              </>
            ) : (
              <>
                <RefreshCw className="h-4 w-4" />
                Retrain Local Model
              </>
            )}
          </button>
          {patientCount < 5 && !isRetraining && (
            <p className="text-xs text-muted-foreground/60 text-center mt-2">
              Need at least 5 patients to train.
            </p>
          )}
        </motion.div>
      </div>

      {/* ── Dataset Preview ─────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25 }}
        className="glass-card rounded-2xl p-6 border border-border"
      >
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-xl ${cfg.bgClass}`}>
              <Database className={`h-5 w-5 ${cfg.colorClass}`} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-foreground">Patient Dataset</h2>
              <p className="text-xs text-muted-foreground">
                {preview
                  ? `${preview.total_rows} total records · showing newest first`
                  : "Loading…"}
              </p>
            </div>
          </div>
          <button
            onClick={() => fetchPreview(previewPage)}
            disabled={previewLoading}
            className="p-2 rounded-lg hover:bg-secondary/60 transition-colors text-muted-foreground"
          >
            <RefreshCw className={`h-4 w-4 ${previewLoading ? "animate-spin" : ""}`} />
          </button>
        </div>

        <AnimatePresence mode="wait">
          {previewLoading ? (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center justify-center py-12"
            >
              <Loader2 className={`h-8 w-8 animate-spin ${cfg.colorClass}`} />
            </motion.div>
          ) : preview && preview.rows.length > 0 ? (
            <motion.div
              key="table"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="overflow-x-auto rounded-xl border border-border">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border bg-secondary/40">
                      {["Preg.", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age", "Outcome"].map((h) => (
                        <th
                          key={h}
                          className="px-3 py-2.5 text-left font-semibold text-muted-foreground whitespace-nowrap"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, i) => (
                      <motion.tr
                        key={i}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.03 }}
                        className="border-b border-border/50 hover:bg-secondary/20 transition-colors"
                      >
                        <td className="px-3 py-2 text-foreground/80">{row.Pregnancies}</td>
                        <td className="px-3 py-2 text-foreground/80">{row.Glucose}</td>
                        <td className="px-3 py-2 text-foreground/80">{row.BloodPressure}</td>
                        <td className="px-3 py-2 text-foreground/80">{row.SkinThickness}</td>
                        <td className="px-3 py-2 text-foreground/80">{row.Insulin}</td>
                        <td className="px-3 py-2 text-foreground/80">{row.BMI}</td>
                        <td className="px-3 py-2 text-foreground/80">{row.DiabetesPedigreeFunction}</td>
                        <td className="px-3 py-2 text-foreground/80">{row.Age}</td>
                        <td className="px-3 py-2">
                          <span
                            className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                              row.Outcome === 1
                                ? "bg-red-500/20 text-red-400"
                                : "bg-green-500/20 text-green-400"
                            }`}
                          >
                            {row.Outcome === 1 ? "Diabetic" : "Healthy"}
                          </span>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {preview.total_pages > 1 && (
                <div className="flex items-center justify-between mt-4">
                  <span className="text-xs text-muted-foreground">
                    Page {preview.page} of {preview.total_pages}
                  </span>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handlePageChange(previewPage - 1)}
                      disabled={previewPage <= 1}
                      className="p-2 rounded-lg border border-border hover:bg-secondary/60 disabled:opacity-40 transition-all"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </button>
                    {Array.from({ length: Math.min(5, preview.total_pages) }, (_, k) => {
                      const start = Math.max(1, previewPage - 2)
                      const page = start + k
                      if (page > preview.total_pages) return null
                      return (
                        <button
                          key={page}
                          onClick={() => handlePageChange(page)}
                          className={`w-8 h-8 rounded-lg text-xs font-medium transition-all border ${
                            page === previewPage
                              ? `${cfg.bgClass} ${cfg.borderClass} ${cfg.colorClass}`
                              : "border-border hover:bg-secondary/60 text-muted-foreground"
                          }`}
                        >
                          {page}
                        </button>
                      )
                    })}
                    <button
                      onClick={() => handlePageChange(previewPage + 1)}
                      disabled={previewPage >= preview.total_pages}
                      className="p-2 rounded-lg border border-border hover:bg-secondary/60 disabled:opacity-40 transition-all"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-12 text-center"
            >
              <Database className="h-10 w-10 text-muted-foreground/30 mb-3" />
              <p className="text-sm text-muted-foreground">No patient records yet.</p>
              <p className="text-xs text-muted-foreground/60 mt-1">
                Use the form above to add the first patient.
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}
