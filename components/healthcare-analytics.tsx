"use client"

import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import { toast } from "sonner"
import {
  BarChart3,
  TrendingUp,
  Users,
  Building2,
  AlertTriangle,
  Scale,
  RefreshCw,
} from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts"
import {
  getAnalyticsOverview,
  getAnalyticsTrends,
  getHospitalContributions,
  getRiskDistribution,
  getAgeGroupRisk,
  getGenderPerformance,
  getFairnessMonitoring,
  getModelPerformance,
  type AnalyticsOverview,
  type TrendData,
  type HospitalContribution,
  type RiskCategory,
  type AgeGroupData,
  type GenderMetric,
  type FairnessGroup,
  type ModelPerformanceResponse,
} from "@/lib/api"

// Custom tooltip
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null

  return (
    <div className="glass-card rounded-lg p-3 border border-border">
      <p className="text-sm font-medium text-foreground mb-2">{label}</p>
      {payload.map((entry: any, index: number) => (
        <p key={index} className="text-xs text-muted-foreground">
          {entry.name}:{" "}
          <span style={{ color: entry.color }} className="font-medium">
            {typeof entry.value === "number" ? entry.value.toLocaleString() : entry.value}
          </span>
        </p>
      ))}
    </div>
  )
}

// Stat card component
function StatCard({
  title,
  value,
  suffix = "",
  icon: Icon,
  trend,
  description,
  delay = 0,
}: {
  title: string
  value: string | number
  suffix?: string
  icon: React.ElementType
  trend?: { value: number; positive: boolean }
  description: string
  delay?: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
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
        {value}
        {suffix}
      </div>
      <div className="text-sm text-muted-foreground">{title}</div>
      <p className="text-xs text-muted-foreground mt-2">{description}</p>
    </motion.div>
  )
}

export function HealthcareAnalytics() {
  const [timeRange, setTimeRange] = useState("year")
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  // All analytics data
  const [overview, setOverview] = useState<AnalyticsOverview | null>(null)
  const [trendData, setTrendData] = useState<TrendData[]>([])
  const [hospitals, setHospitals] = useState<HospitalContribution[]>([])
  const [riskCategories, setRiskCategories] = useState<RiskCategory[]>([])
  const [ageGroups, setAgeGroups] = useState<AgeGroupData[]>([])
  const [genderMetrics, setGenderMetrics] = useState<GenderMetric[]>([])
  const [fairnessGroups, setFairnessGroups] = useState<FairnessGroup[]>([])
  const [modelPerf, setModelPerf] = useState<ModelPerformanceResponse | null>(null)

  // Load all analytics data
  const loadAnalytics = async () => {
    try {
      const [
        overviewData,
        trendsData,
        hospitalsData,
        riskData,
        ageData,
        genderData,
        fairnessData,
        perfData,
      ] = await Promise.all([
        getAnalyticsOverview(),
        getAnalyticsTrends(),
        getHospitalContributions(),
        getRiskDistribution(),
        getAgeGroupRisk(),
        getGenderPerformance(),
        getFairnessMonitoring(),
        getModelPerformance(),
      ])

      setOverview(overviewData)
      setTrendData(trendsData.diabetes || [])
      setHospitals(hospitalsData.hospitals || [])
      setRiskCategories(riskData.categories || [])
      setAgeGroups(ageData.age_groups || [])
      setGenderMetrics(genderData.metrics || [])
      setFairnessGroups(fairnessData.groups || [])
      setModelPerf(perfData)
    } catch (err) {
      console.error("Failed to load analytics:", err)
      toast.error("Failed to load analytics data")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAnalytics()

    // Auto-refresh every 10 seconds
    const interval = setInterval(() => {
      loadAnalytics()
    }, 10000)

    return () => clearInterval(interval)
  }, [])

  const handleRefresh = async () => {
    setRefreshing(true)
    await loadAnalytics()
    setRefreshing(false)
    toast.success("Analytics updated")
  }

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh]">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          className="w-14 h-14 rounded-full border-2 border-t-transparent border-primary mb-4"
        />
        <p className="text-muted-foreground text-sm animate-pulse">
          Loading analytics…
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4"
      >
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-xl bg-primary/20">
              <BarChart3 className="h-6 w-6 text-primary" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">
              Healthcare Analytics
            </h1>
          </div>
          <p className="text-muted-foreground">
            Live federated AI monitoring system
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="bg-secondary border border-border rounded-lg px-4 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
          >
            <option value="week">Last 7 Days</option>
            <option value="month">Last 30 Days</option>
            <option value="quarter">Last Quarter</option>
            <option value="year">Last Year</option>
          </select>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>
      </motion.div>

      {/* Top Stats */}
      {overview && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Total Predictions"
            value={overview.total_predictions.toLocaleString()}
            icon={BarChart3}
            description="Predictions made across network"
            delay={0}
          />
          <StatCard
            title="High-Risk Patients"
            value={overview.high_risk_patients.toLocaleString()}
            icon={AlertTriangle}
            description="Requiring immediate attention"
            delay={0.1}
          />
          <StatCard
            title="Network Hospitals"
            value={overview.participating_hospitals}
            icon={Building2}
            description="Active in federated network"
            delay={0.2}
          />
          <StatCard
            title="Fairness Score"
            value={overview.fairness_score}
            suffix="%"
            icon={Scale}
            description="Model bias assessment"
            delay={0.3}
          />
        </div>
      )}

      {/* Disease Trends Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-card rounded-2xl p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="font-semibold text-foreground">Disease Trend Analysis</h3>
            <p className="text-sm text-muted-foreground">
              Model accuracy evolution across federated rounds
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-chart-1" />
              <span className="text-xs text-muted-foreground">Accuracy</span>
            </div>
          </div>
        </div>
        <div className="h-80">
          {trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trendData}>
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
                />
                <YAxis
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
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
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              No trend data available
            </div>
          )}
        </div>
      </motion.div>

      {/* Two Column Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-card rounded-2xl p-6"
        >
          <h3 className="font-semibold text-foreground mb-4">
            Patient Risk Distribution
          </h3>
          <div className="h-64 flex items-center justify-center">
            {riskCategories.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskCategories}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {riskCategories.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-muted-foreground">No risk data available</div>
            )}
          </div>
          <div className="grid grid-cols-2 gap-4 mt-4">
            {riskCategories.map((item) => (
              <div key={item.name} className="flex items-center gap-2">
                <span
                  className="h-3 w-3 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-sm text-muted-foreground">
                  {item.name}: {item.value.toLocaleString()}
                </span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Hospital Contribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass-card rounded-2xl p-6"
        >
          <h3 className="font-semibold text-foreground mb-4">
            Hospital Contribution Analytics
          </h3>
          <div className="h-64">
            {hospitals.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hospitals}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis
                    dataKey="hospital"
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
                  />
                  <YAxis
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="patients" name="Patients" fill="#22d3ee" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="predictions" name="Predictions" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                No hospital data available
              </div>
            )}
          </div>
          <div className="flex items-center justify-center gap-8 mt-4">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-primary" />
              <span className="text-sm text-muted-foreground">Patients</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-accent" />
              <span className="text-sm text-muted-foreground">Predictions</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Age Group Risk & Gender Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Age Group Risk */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="glass-card rounded-2xl p-6"
        >
          <h3 className="font-semibold text-foreground mb-4">
            Age Group Risk Analysis
          </h3>
          <div className="h-64">
            {ageGroups.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ageGroups}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis
                    dataKey="age"
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
                  />
                  <YAxis
                    stroke="rgba(255,255,255,0.5)"
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="low" name="Low Risk" stackId="a" fill="#22c55e" />
                  <Bar dataKey="medium" name="Medium Risk" stackId="a" fill="#eab308" />
                  <Bar dataKey="high" name="High Risk" stackId="a" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                No age group data available
              </div>
            )}
          </div>
          <div className="flex items-center justify-center gap-6 mt-4">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-success" />
              <span className="text-xs text-muted-foreground">Low</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-warning" />
              <span className="text-xs text-muted-foreground">Medium</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-destructive" />
              <span className="text-xs text-muted-foreground">High</span>
            </div>
          </div>
        </motion.div>

        {/* Gender Performance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="glass-card rounded-2xl p-6"
        >
          <h3 className="font-semibold text-foreground mb-4">
            Gender-Based Model Performance
          </h3>
          <div className="h-64">
            {genderMetrics.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={genderMetrics}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)" />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                  />
                  <PolarRadiusAxis
                    angle={30}
                    domain={[85, 100]}
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
                  />
                  <Radar
                    name="Male"
                    dataKey="male"
                    stroke="#22d3ee"
                    fill="#22d3ee"
                    fillOpacity={0.3}
                  />
                  <Radar
                    name="Female"
                    dataKey="female"
                    stroke="#f472b6"
                    fill="#f472b6"
                    fillOpacity={0.3}
                  />
                  <Tooltip content={<CustomTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                No gender data available
              </div>
            )}
          </div>
          <div className="flex items-center justify-center gap-8 mt-4">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-primary" />
              <span className="text-sm text-muted-foreground">Male</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-pink-400" />
              <span className="text-sm text-muted-foreground">Female</span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Fairness Monitoring */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="glass-card rounded-2xl p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="font-semibold text-foreground">Fairness Monitoring</h3>
            <p className="text-sm text-muted-foreground">
              Model performance across demographic groups
            </p>
          </div>
          <span className="flex items-center gap-2 px-3 py-1 rounded-full bg-success/20 text-success text-sm">
            <Scale className="h-4 w-4" />
            All metrics within acceptable range
          </span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {fairnessGroups.map((item, index) => (
            <motion.div
              key={item.group}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.9 + index * 0.05 }}
              className="glass rounded-xl p-4 text-center"
            >
              <div className="text-2xl font-bold text-foreground mb-1">
                {item.value}%
              </div>
              <div className="text-xs text-muted-foreground">{item.group}</div>
              <div className="mt-2 h-1 bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-primary to-success rounded-full"
                  style={{ width: `${item.value}%` }}
                />
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Model Performance Summary */}
      {modelPerf && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1 }}
          className="glass-card rounded-2xl p-6"
        >
          <h3 className="font-semibold text-foreground mb-4">Model Performance Metrics</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {[
              { label: "Accuracy", value: modelPerf.accuracy },
              { label: "Precision", value: modelPerf.precision },
              { label: "Recall", value: modelPerf.recall },
              { label: "F1 Score", value: modelPerf.f1_score },
              { label: "ROC-AUC", value: modelPerf.roc_auc },
            ].map((metric) => (
              <div key={metric.label} className="text-center">
                <div className="text-2xl font-bold text-primary mb-1">
                  {metric.value}%
                </div>
                <div className="text-xs text-muted-foreground">{metric.label}</div>
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-center justify-center gap-8 text-sm">
            <div>
              <span className="text-muted-foreground">Federated Round: </span>
              <span className="font-semibold">{modelPerf.federated_round}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Final Loss: </span>
              <span className="font-semibold">{modelPerf.final_loss}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Epochs: </span>
              <span className="font-semibold">{modelPerf.epochs}</span>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}
