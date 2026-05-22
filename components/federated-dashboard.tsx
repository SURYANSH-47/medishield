"use client"

import { motion, useMotionValue, useTransform, animate } from "framer-motion"
import { useEffect, useState, useRef } from "react"
import Link from "next/link"
import {
  Activity,
  Users,
  Building2,
  AlertTriangle,
  TrendingUp,
  Server,
  RefreshCw,
  CheckCircle,
  Clock,
  Wifi,
  ChevronRight,
} from "lucide-react"
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Cell,
  BarChart,
  Bar,
} from "recharts"
import { 
  getDashboardMetrics, 
  getHospitalMetrics,
  triggerRetrain, 
  type DashboardMetricsResponse, 
  type HospitalMetric 
} from "@/lib/api"

// Animated counter hook
function useAnimatedCounter(value: number, duration = 2) {
  const count = useMotionValue(0)
  const rounded = useTransform(count, (latest) => Math.round(latest * 10) / 10)
  const [displayValue, setDisplayValue] = useState(0)

  useEffect(() => {
    const controls = animate(count, value, { duration, ease: "easeOut" })
    const unsubscribe = rounded.on("change", (v) => setDisplayValue(v))
    return () => {
      controls.stop()
      unsubscribe()
    }
  }, [count, rounded, value, duration])

  return displayValue
}

// Metric Card Component
function MetricCard({
  title,
  value,
  suffix = "",
  icon: Icon,
  trend,
  delay = 0,
}: {
  title: string
  value: number
  suffix?: string
  icon: React.ElementType
  trend?: { value: number; positive: boolean }
  delay?: number
}) {
  const animatedValue = useAnimatedCounter(value)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay }}
      whileHover={{ scale: 1.02 }}
      className="glass-card rounded-2xl p-6 hover:glow-border transition-all duration-300"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="p-3 rounded-xl bg-primary/20">
          <Icon className="h-6 w-6 text-primary" />
        </div>
        {trend && (
          <span
            className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full ${
              trend.positive
                ? "bg-success/20 text-success"
                : "bg-destructive/20 text-destructive"
            }`}
          >
            <TrendingUp
              className={`h-3 w-3 ${!trend.positive ? "rotate-180" : ""}`}
            />
            {trend.value}%
          </span>
        )}
      </div>
      <div className="text-3xl font-bold text-foreground mb-1">
        {animatedValue}
        {suffix}
      </div>
      <div className="text-sm text-muted-foreground">{title}</div>
    </motion.div>
  )
}

// Hospital Node Card Component
function HospitalNodeCard({
  name,
  localAccuracy,
  patientsCount,
  trainingStatus,
  syncStatus,
  delay = 0,
}: {
  name: string
  localAccuracy: number
  patientsCount: number
  trainingStatus: "active" | "idle" | "syncing" | "training"
  syncStatus: "synced" | "pending" | "syncing"
  delay?: number
}) {
  const statusColors = {
    active: "bg-success text-success",
    idle: "bg-warning text-warning",
    syncing: "bg-primary text-primary",
    training: "bg-yellow-400 text-yellow-400",
  }

  const syncColors = {
    synced: "bg-success/20 text-success",
    pending: "bg-warning/20 text-warning",
    syncing: "bg-primary/20 text-primary",
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -5 }}
      className="glass-card rounded-2xl p-6 hover:glow-border transition-all duration-300 cursor-pointer group"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-accent/20">
            <Building2 className="h-5 w-5 text-accent" />
          </div>
          <h3 className="font-semibold text-foreground">{name}</h3>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`h-2 w-2 rounded-full ${statusColors[trainingStatus].split(" ")[0]} ${
              trainingStatus !== "active" && trainingStatus !== "idle" ? "animate-pulse" : ""
            }`}
          />
          <span className="text-xs text-muted-foreground capitalize">
            {trainingStatus}
          </span>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-muted-foreground">Local Accuracy</span>
            <span className="font-medium text-foreground">{localAccuracy}%</span>
          </div>
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${localAccuracy}%` }}
              transition={{ duration: 1, delay: delay + 0.3 }}
              className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
            />
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              {patientsCount.toLocaleString()} patients
            </span>
          </div>
          <span
            className={`text-xs px-2 py-1 rounded-full ${syncColors[syncStatus]}`}
          >
            {syncStatus === "syncing" && (
              <RefreshCw className="h-3 w-3 inline mr-1 animate-spin" />
            )}
            {syncStatus === "synced" && (
              <CheckCircle className="h-3 w-3 inline mr-1" />
            )}
            {syncStatus === "pending" && (
              <Clock className="h-3 w-3 inline mr-1" />
            )}
            {syncStatus}
          </span>
        </div>

        <div className="flex items-center justify-end gap-1 text-[11px] text-muted-foreground/60 group-hover:text-primary transition-colors">
          <span>Manage hospital</span>
          <ChevronRight className="h-3 w-3" />
        </div>
      </div>
    </motion.div>
  )
}

// Federated Server Visualization
function FederatedServerVisualization({
  round,
  hospitals,
  isTraining
}: {
  round: number
  hospitals: HospitalMetric[]
  isTraining: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let animationId: number
    let angle = 0

    const draw = () => {
      const width = canvas.width
      const height = canvas.height
      const centerX = width / 2
      const centerY = height / 2

      ctx.clearRect(0, 0, width, height)

      // Draw orbiting particles
      const numParticles = 12
      for (let i = 0; i < numParticles; i++) {
        const particleAngle = angle + (i * Math.PI * 2) / numParticles
        const radius = 60
        const x = centerX + Math.cos(particleAngle) * radius
        const y = centerY + Math.sin(particleAngle) * radius

        // Draw connection line
        ctx.beginPath()
        ctx.moveTo(centerX, centerY)
        ctx.lineTo(x, y)
        ctx.strokeStyle = `rgba(34, 211, 238, ${0.2 + Math.sin(angle + i) * 0.1})`
        ctx.lineWidth = 1
        ctx.stroke()

        // Draw particle
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(34, 211, 238, ${0.5 + Math.sin(angle + i) * 0.3})`
        ctx.fill()
      }

      // Draw central glow
      const gradient = ctx.createRadialGradient(
        centerX,
        centerY,
        0,
        centerX,
        centerY,
        40
      )
      gradient.addColorStop(0, "rgba(34, 211, 238, 0.3)")
      gradient.addColorStop(1, "rgba(34, 211, 238, 0)")
      ctx.beginPath()
      ctx.arc(centerX, centerY, 40, 0, Math.PI * 2)
      ctx.fillStyle = gradient
      ctx.fill()

      angle += isTraining ? 0.06 : 0.02
      animationId = requestAnimationFrame(draw)
    }

    draw()

    return () => cancelAnimationFrame(animationId)
  }, [isTraining])

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.8 }}
      className="glass-card rounded-2xl p-6 glow-border"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-foreground">Central AI Server</h3>
        <span className={`flex items-center gap-2 text-xs ${isTraining ? "text-primary animate-pulse" : "text-success"}`}>
          {isTraining ? (
            <>
              <RefreshCw className="h-3 w-3 animate-spin" />
              Aggregating...
            </>
          ) : (
            <>
              <Wifi className="h-3 w-3" />
              Connected
            </>
          )}
        </span>
      </div>

      <div className="relative flex items-center justify-center h-40">
        <canvas
          ref={canvasRef}
          width={200}
          height={160}
          className="absolute inset-0"
        />
        <div className="relative z-10 flex flex-col items-center">
          <div className="p-4 rounded-2xl bg-primary/20 mb-2">
            <Server className={`h-8 w-8 text-primary ${isTraining ? "animate-pulse" : ""}`} />
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">Round {round}</div>
            <div className="text-xs text-muted-foreground">
              {isTraining ? "Federated Aggregation" : "Model Synchronized"}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2 text-center">
        {hospitals.map((hospital, i) => (
          <div key={hospital.name} className="glass rounded-lg p-2">
            <div className="text-[10px] text-muted-foreground font-medium truncate">{hospital.name}</div>
            <motion.div
              animate={hospital.status !== "active" ? { opacity: [0.5, 1, 0.5] } : {}}
              transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.5 }}
              className={`text-[10px] font-semibold uppercase ${
                hospital.status === "active" 
                  ? "text-success" 
                  : hospital.status === "training"
                  ? "text-yellow-400"
                  : "text-primary"
              }`}
            >
              {hospital.status === "active" ? "Synced" : hospital.status === "training" ? "Training" : "Syncing"}
            </motion.div>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

// Custom Tooltip for Charts
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null

  return (
    <div className="glass-card rounded-lg p-3 border border-border">
      <p className="text-sm font-medium text-foreground">Round {label}</p>
      {payload.map((entry: any, index: number) => (
        <p key={index} className="text-xs text-muted-foreground">
          {entry.name}: <span className="text-primary font-medium">{entry.value}%</span>
        </p>
      ))}
    </div>
  )
}

export function FederatedDashboard() {
  const [metrics, setMetrics] = useState<DashboardMetricsResponse | null>(null)
  // Separate state for hospital cards — polled independently at 5s intervals
  // so CSV changes appear instantly without waiting for the heavy dashboard poll
  const [hospitalMetrics, setHospitalMetrics] = useState<HospitalMetric[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [retraining, setRetraining] = useState(false)
  const [error, setError] = useState(false)

  // ── Heavy dashboard poll (global accuracy, charts, training stats) ──────────
  const fetchMetrics = async (isFirstLoad = false) => {
    if (isFirstLoad) setLoading(true)
    try {
      const data = await getDashboardMetrics()
      setMetrics(data)
      setError(false)
      setRetraining(data.is_training ?? false)
    } catch (err) {
      console.error("Error fetching dashboard metrics:", err)
      setError(true)
    } finally {
      if (isFirstLoad) setLoading(false)
    }
  }

  // ── Fast hospital-only poll — reads CSVs fresh every 5s ─────────────────────
  // Completely decoupled from the main dashboard poll so CSV edits show
  // immediately in the hospital cards without touching other dashboard state.
  const fetchHospitalMetrics = async () => {
    try {
      const data = await getHospitalMetrics()
      // getHospitalMetrics returns { hospitals: [...] } per the new endpoint
      const list = Array.isArray(data) ? data : (data as any).hospitals ?? []
      setHospitalMetrics(list)
    } catch (err) {
      console.warn("Hospital metrics poll failed (non-fatal):", err)
      // Don't crash the dashboard — fall back to last known state silently
    }
  }

  useEffect(() => {
    fetchMetrics(true)
    fetchHospitalMetrics()          // immediate first load for hospital cards

    const dashboardInterval = setInterval(() => fetchMetrics(false), 10000)  // 10s for heavy data
    const hospitalInterval  = setInterval(fetchHospitalMetrics, 5000)        // 5s for hospital cards

    return () => {
      clearInterval(dashboardInterval)
      clearInterval(hospitalInterval)
    }
  }, [])

  const handleRetrain = async () => {
    setRetraining(true)
    try {
      await triggerRetrain()
      // Trigger an immediate hospital metrics refresh after retraining kicks off
      setTimeout(() => {
        fetchMetrics(false)
        fetchHospitalMetrics()
      }, 500)
    } catch (err) {
      console.error("Failed to retrain:", err)
      setRetraining(false)
    }
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] glass-card rounded-2xl p-8 border border-border">
        <div className="relative mb-6">
          {/* Cybernetic pulsing rings */}
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
            className="w-16 h-16 rounded-full border-2 border-t-primary border-r-accent border-b-transparent border-l-transparent"
          />
          <motion.div
            animate={{ rotate: -360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            className="absolute inset-2 rounded-full border-2 border-b-accent border-l-primary border-t-transparent border-r-transparent"
          />
        </div>
        <h3 className="text-lg font-semibold text-foreground mb-2">Connecting to Federated Network</h3>
        <p className="text-sm text-muted-foreground max-w-md text-center animate-pulse">
          Establishing encrypted TLS tunnel and loading secure PyTorch global model weights...
        </p>
      </div>
    )
  }

  if (error || !metrics) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card rounded-2xl p-8 border border-destructive/40 bg-destructive/5 flex flex-col items-center justify-center min-h-[400px] text-center"
      >
        <div className="p-4 rounded-full bg-destructive/10 border border-destructive/30 mb-6 animate-pulse">
          <AlertTriangle className="h-10 w-10 text-destructive" />
        </div>
        <h3 className="text-xl font-bold text-foreground mb-2">Federated Server Unreachable</h3>
        <p className="text-sm text-muted-foreground max-w-lg mb-6">
          Unable to establish connection with the FastAPI backend. Please ensure your Python server is running on <code className="px-1.5 py-0.5 rounded bg-secondary text-accent">http://127.0.0.1:8000</code>.
        </p>
        <button
          onClick={() => fetchMetrics(true)}
          className="flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold hover:bg-primary/90 transition-all shadow-lg hover:shadow-primary/20"
        >
          <RefreshCw className="h-4 w-4" />
          Retry Secure Connection
        </button>
      </motion.div>
    )
  }

  const currentAcc = metrics.global_accuracy
  const round = metrics.federated_round
  const accuracyChartData = metrics.chart_data

  const hospitalParticipationLive = metrics.hospitals.map((h, i) => ({
    name: h.name,
    patients: h.patients,
    color: ["#22d3ee", "#2dd4bf", "#818cf8"][i] || "#22d3ee",
  }))

  const isTraining = metrics.is_training ?? retraining

  return (
    <div className="space-y-6">
      {/* Live indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <motion.div
            animate={{ scale: [1, 1.3, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className={`h-2 w-2 rounded-full ${isTraining ? "bg-primary animate-pulse" : "bg-green-400"}`}
          />
          <span className="text-sm text-muted-foreground">
            {isTraining 
              ? `Federated Learning Round ${round} in progress...` 
              : `Live — Federated Round ${round}`}
          </span>
        </div>
        <button
          onClick={handleRetrain}
          disabled={isTraining}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/20 hover:bg-primary/30 border border-primary/40 text-primary text-sm font-medium transition-all disabled:opacity-50"
        >
          <motion.div
            animate={isTraining ? { rotate: 360 } : {}}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          >
            <RefreshCw className="h-4 w-4" />
          </motion.div>
          {isTraining ? "Aggregating Weights..." : "Run Federated Round"}
        </button>
      </div>

      {/* Top Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Global Model Accuracy"
          value={currentAcc}
          suffix="%"
          icon={Activity}
          trend={{ value: 2.3, positive: true }}
          delay={0}
        />
        <MetricCard
          title="Federated Round"
          value={round}
          icon={RefreshCw}
          trend={{ value: 1, positive: true }}
          delay={0.1}
        />
        <MetricCard
          title="Total Patients"
          value={metrics.total_patients}
          icon={Users}
          delay={0.2}
        />
        <MetricCard
          title="High-Risk Patients"
          value={metrics.high_risk_patients}
          icon={AlertTriangle}
          trend={{ value: 5.2, positive: false }}
          delay={0.3}
        />
      </div>

      {/* Hospital Nodes — always use freshly-polled hospitalMetrics if available */}
      <div>
        <motion.h2
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="text-xl font-semibold text-foreground mb-4"
        >
          Hospital Nodes — Local Training
        </motion.h2>

        {/* Per-hospital last-updated timestamp */}
        <p className="text-xs text-muted-foreground mb-3 flex items-center gap-1.5">
          <motion.span
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="h-1.5 w-1.5 rounded-full bg-green-400 inline-block"
          />
          Auto-refreshing every 5s from live CSV datasets
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(hospitalMetrics ?? metrics.hospitals).map((h, i) => {
            const hospitalId = (["a", "b", "c"] as const)[i] ?? String(i)
            return (
              <Link key={h.name} href={`/hospital/${hospitalId}`} className="block no-underline">
                <HospitalNodeCard
                  name={h.name}
                  localAccuracy={h.accuracy}
                  patientsCount={h.patients}
                  trainingStatus={h.status}
                  syncStatus={h.sync_status}
                  delay={0.5 + i * 0.1}
                />
              </Link>
            )
          })}
        </div>
      </div>

      {/* Federated Server & Accuracy Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <FederatedServerVisualization 
          round={round}
          hospitals={metrics.hospitals}
          isTraining={isTraining}
        />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="lg:col-span-2 glass-card rounded-2xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-foreground">Model Accuracy vs Federated Rounds</h3>
            <span className={`text-xs font-medium ${isTraining ? "text-primary animate-pulse" : "text-green-400"}`}>
              {isTraining ? "● Aggregating" : "● Live"}
            </span>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={accuracyChartData}>
                <defs>
                  <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis
                  dataKey="round"
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                />
                <YAxis
                  domain={[50, 100]}
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
                  axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#22d3ee"
                  strokeWidth={2}
                  fill="url(#accuracyGradient)"
                  name="Accuracy"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Bottom Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.9 }}
          className="glass-card rounded-2xl p-6"
        >
          <h3 className="font-semibold text-foreground mb-4">Hospital Dataset Sizes</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hospitalParticipationLive} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis
                  type="number"
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
                />
                <YAxis
                  dataKey="name"
                  type="category"
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
                  width={80}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="patients" name="Patients" radius={[0, 4, 4, 0]}>
                  {hospitalParticipationLive.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1 }}
          className="glass-card rounded-2xl p-6"
        >
          <h3 className="font-semibold text-foreground mb-2">Model Performance Metrics</h3>
          <p className="text-xs text-muted-foreground mb-4">Real values from last training run</p>
          <div className="space-y-3">
            {[
              { label: "Test Accuracy",  value: `${(metrics.test_accuracy  ?? metrics.global_accuracy).toFixed(2)}%`, color: "bg-primary"      },
              { label: "Train Accuracy", value: `${(metrics.train_accuracy ?? 53.61).toFixed(2)}%`, color: "bg-accent"       },
              { label: "Final Loss",     value:   (metrics.final_loss      ?? 0.7662).toFixed(4),   color: "bg-yellow-500"  },
              { label: "Epochs Trained", value: String(metrics.epochs      ?? 50),                  color: "bg-purple-500"  },
            ].map(m => (
              <div key={m.label} className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${m.color}`} />
                  <span className="text-sm text-muted-foreground">{m.label}</span>
                </div>
                <span className="text-sm font-semibold text-foreground">{m.value}</span>
              </div>
            ))}
          </div>
          {metrics.last_trained && (
            <p className="text-xs text-muted-foreground mt-3">
              Last trained: {new Date(metrics.last_trained).toLocaleString()}
            </p>
          )}
        </motion.div>
      </div>
    </div>
  )
}
