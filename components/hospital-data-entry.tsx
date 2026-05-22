"use client"

import { motion, AnimatePresence } from "framer-motion"
import { useState } from "react"
import {
  Building2, Plus, RefreshCw, CheckCircle, User, Droplet,
  Activity, Thermometer, Zap, Scale, Dna, Heart, AlertTriangle
} from "lucide-react"
import { addPatient, triggerRetrain } from "@/lib/api"

const HOSPITALS = ["Hospital A", "Hospital B", "Hospital C"]

const fields = [
  { key: "pregnancies", label: "Pregnancies", icon: Heart, min: 0, max: 20, step: 1, unit: "" },
  { key: "glucose", label: "Glucose", icon: Droplet, min: 0, max: 300, step: 1, unit: "mg/dL" },
  { key: "blood_pressure", label: "Blood Pressure", icon: Activity, min: 0, max: 200, step: 1, unit: "mmHg" },
  { key: "skin_thickness", label: "Skin Thickness", icon: Thermometer, min: 0, max: 100, step: 1, unit: "mm" },
  { key: "insulin", label: "Insulin", icon: Zap, min: 0, max: 900, step: 1, unit: "μU/mL" },
  { key: "bmi", label: "BMI", icon: Scale, min: 10, max: 70, step: 0.1, unit: "kg/m²" },
  { key: "diabetes_pedigree_function", label: "Diabetes Pedigree", icon: Dna, min: 0, max: 2.5, step: 0.01, unit: "" },
  { key: "age", label: "Age", icon: User, min: 1, max: 120, step: 1, unit: "yrs" },
]

export function HospitalDataEntry({ onDataAdded }: { onDataAdded?: () => void }) {
  const [hospital, setHospital] = useState("Hospital A")
  const [outcome, setOutcome] = useState<0 | 1>(0)
  const [form, setForm] = useState({
    pregnancies: 2,
    glucose: 120,
    blood_pressure: 70,
    skin_thickness: 25,
    insulin: 100,
    bmi: 28.0,
    diabetes_pedigree_function: 0.45,
    age: 35,
  })
  const [status, setStatus] = useState<"idle" | "submitting" | "success" | "retraining" | "done">("idle")
  const [successMsg, setSuccessMsg] = useState("")
  const [retrainMsg, setRetrainMsg] = useState("")
  const [error, setError] = useState("")

  const handleSubmit = async () => {
    setStatus("submitting")
    setError("")
    try {
      const res = await addPatient({ ...form, outcome, hospital })
      setSuccessMsg(`Patient added to ${res.hospital}. Dataset now has ${res.new_patient_count} patients.`)
      setStatus("success")
      onDataAdded?.()
    } catch {
      setError("Failed to add patient — is the backend running?")
      setStatus("idle")
    }
  }

  const handleRetrain = async () => {
    setStatus("retraining")
    setRetrainMsg("")
    try {
      const res = await triggerRetrain()
      setRetrainMsg(`${res.message} — training in background (~30s)`)
      setStatus("done")
      setTimeout(() => window.location.reload(), 35000)
    } catch {
      setError("Retrain failed — is the backend running?")
      setStatus("success")
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl p-6 border border-border"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="p-3 rounded-xl bg-accent/20">
          <Building2 className="h-6 w-6 text-accent" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-foreground">Add Hospital Patient Data</h2>
          <p className="text-sm text-muted-foreground">New data trains the federated model</p>
        </div>
      </div>

      {/* Hospital selector */}
      <div className="mb-4">
        <label className="text-sm font-medium text-foreground mb-2 block">Select Hospital Node</label>
        <div className="flex gap-2">
          {HOSPITALS.map(h => (
            <button
              key={h}
              onClick={() => setHospital(h)}
              className={`flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all border ${
                hospital === h
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-secondary/50 text-muted-foreground border-border hover:border-primary/50"
              }`}
            >
              {h}
            </button>
          ))}
        </div>
      </div>

      {/* Feature inputs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        {fields.map(f => (
          <div key={f.key}>
            <label className="text-xs text-muted-foreground mb-1 flex items-center gap-1">
              <f.icon className="h-3 w-3" />
              {f.label}
              {f.unit && <span className="text-[10px]">({f.unit})</span>}
            </label>
            <input
              type="number"
              min={f.min}
              max={f.max}
              step={f.step}
              value={form[f.key as keyof typeof form]}
              onChange={e => setForm(prev => ({ ...prev, [f.key]: parseFloat(e.target.value) || 0 }))}
              className="w-full bg-secondary/50 border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>
        ))}
      </div>

      {/* Outcome selector */}
      <div className="mb-6">
        <label className="text-sm font-medium text-foreground mb-2 block">Diagnosis (Ground Truth)</label>
        <div className="flex gap-4">
          {[
            { v: 0, label: "Healthy (No Diabetes)", color: "text-green-400" },
            { v: 1, label: "Diabetic", color: "text-red-400" },
          ].map(opt => (
            <label key={opt.v} className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="outcome"
                value={opt.v}
                checked={outcome === opt.v}
                onChange={() => setOutcome(opt.v as 0 | 1)}
                className="accent-primary"
              />
              <span className={`text-sm font-medium ${opt.color}`}>{opt.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="mb-4 p-3 rounded-lg bg-destructive/20 border border-destructive/40 flex items-center gap-2"
          >
            <AlertTriangle className="h-4 w-4 text-destructive flex-shrink-0" />
            <p className="text-sm text-destructive">{error}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Submit button */}
      {(status === "idle" || status === "submitting") && (
        <button
          onClick={handleSubmit}
          disabled={status === "submitting"}
          className="flex items-center gap-2 px-6 py-3 rounded-xl bg-accent text-white font-semibold hover:bg-accent/90 transition-all disabled:opacity-50"
        >
          {status === "submitting" ? (
            <>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              >
                <RefreshCw className="h-4 w-4" />
              </motion.div>
              Adding Patient...
            </>
          ) : (
            <>
              <Plus className="h-4 w-4" />
              Add Patient to {hospital}
            </>
          )}
        </button>
      )}

      {/* Success state */}
      <AnimatePresence>
        {(status === "success" || status === "retraining" || status === "done") && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <div className="flex items-center gap-3 p-4 rounded-xl bg-green-500/10 border border-green-500/30">
              <CheckCircle className="h-5 w-5 text-green-400 flex-shrink-0" />
              <p className="text-sm text-green-300 font-medium">{successMsg}</p>
            </div>

            {status === "success" && (
              <motion.button
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                onClick={handleRetrain}
                className="w-full flex items-center justify-center gap-3 px-6 py-4 rounded-xl bg-gradient-to-r from-primary to-accent text-white font-semibold hover:opacity-90 transition-all glow-cyan"
              >
                <RefreshCw className="h-5 w-5" />
                Trigger Federated Learning Round →
              </motion.button>
            )}

            {status === "retraining" && (
              <div className="flex items-center justify-center gap-3 p-4 rounded-xl bg-primary/10 border border-primary/30">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                >
                  <RefreshCw className="h-5 w-5 text-primary" />
                </motion.div>
                <span className="text-sm text-primary font-medium">Starting federated round...</span>
              </div>
            )}

            {status === "done" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="p-4 rounded-xl bg-primary/10 border border-primary/30"
              >
                <div className="flex items-center gap-2 mb-2">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  >
                    <RefreshCw className="h-4 w-4 text-primary" />
                  </motion.div>
                  <span className="text-sm font-semibold text-primary">Federated Learning Active</span>
                </div>
                <p className="text-xs text-muted-foreground">{retrainMsg}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Dashboard will auto-refresh when training completes (~30s)
                </p>
                <div className="flex gap-1 mt-3">
                  {[0, 1, 2].map(i => (
                    <motion.div
                      key={i}
                      animate={{ scale: [1, 1.5, 1] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.2 }}
                      className="h-1.5 w-1.5 rounded-full bg-primary"
                    />
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
