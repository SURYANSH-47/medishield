"use client"

import { motion } from "framer-motion"
import { useState } from "react"
import {
  BarChart3,
  TrendingUp,
  Users,
  Building2,
  AlertTriangle,
  Scale,
  Calendar,
  Filter,
  Download,
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

// Disease trend data
const diseaseTrendData = [
  { month: "Jan", diabetes: 245, heart: 189, hypertension: 312 },
  { month: "Feb", diabetes: 267, heart: 195, hypertension: 298 },
  { month: "Mar", diabetes: 298, heart: 210, hypertension: 325 },
  { month: "Apr", diabetes: 285, heart: 198, hypertension: 310 },
  { month: "May", diabetes: 312, heart: 225, hypertension: 342 },
  { month: "Jun", diabetes: 325, heart: 232, hypertension: 356 },
  { month: "Jul", diabetes: 340, heart: 245, hypertension: 368 },
  { month: "Aug", diabetes: 358, heart: 256, hypertension: 385 },
  { month: "Sep", diabetes: 375, heart: 268, hypertension: 398 },
  { month: "Oct", diabetes: 392, heart: 278, hypertension: 412 },
  { month: "Nov", diabetes: 410, heart: 290, hypertension: 425 },
  { month: "Dec", diabetes: 428, heart: 302, hypertension: 438 },
]

// Hospital contribution data
const hospitalContribution = [
  { hospital: "Hospital A", patients: 12500, predictions: 8450, accuracy: 94.2 },
  { hospital: "Hospital B", patients: 9800, predictions: 6720, accuracy: 92.8 },
  { hospital: "Hospital C", patients: 15200, predictions: 10340, accuracy: 95.1 },
]

// Age group risk data
const ageGroupRisk = [
  { age: "18-25", low: 85, medium: 12, high: 3 },
  { age: "26-35", low: 72, medium: 20, high: 8 },
  { age: "36-45", low: 58, medium: 28, high: 14 },
  { age: "46-55", low: 42, medium: 35, high: 23 },
  { age: "56-65", low: 28, medium: 38, high: 34 },
  { age: "65+", low: 18, medium: 35, high: 47 },
]

// Gender comparison data
const genderComparison = [
  { metric: "Accuracy", male: 93.5, female: 94.8 },
  { metric: "Precision", male: 91.2, female: 92.6 },
  { metric: "Recall", male: 89.8, female: 91.4 },
  { metric: "F1 Score", male: 90.5, female: 92.0 },
  { metric: "AUC-ROC", male: 94.2, female: 95.1 },
]

// Fairness metrics
const fairnessData = [
  { group: "Overall", value: 94.2 },
  { group: "Male", value: 93.5 },
  { group: "Female", value: 94.8 },
  { group: "Age < 40", value: 92.8 },
  { group: "Age 40-60", value: 94.5 },
  { group: "Age > 60", value: 93.9 },
]

// High-risk patient stats
const highRiskStats = [
  { name: "Critical", value: 847, color: "#ef4444" },
  { name: "Elevated", value: 1523, color: "#f97316" },
  { name: "Moderate", value: 2845, color: "#eab308" },
  { name: "Low", value: 5285, color: "#22c55e" },
]

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
            Enterprise-grade insights across the federated network
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
          <button className="flex items-center gap-2 px-4 py-2 rounded-lg glass hover:bg-secondary/50 transition-all text-foreground">
            <Filter className="h-4 w-4" />
            Filter
          </button>
          <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-all">
            <Download className="h-4 w-4" />
            Export
          </button>
        </div>
      </motion.div>

      {/* Top Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Predictions"
          value="25,510"
          icon={BarChart3}
          trend={{ value: 12.5, positive: true }}
          description="Predictions made this period"
          delay={0}
        />
        <StatCard
          title="High-Risk Patients"
          value="2,370"
          icon={AlertTriangle}
          trend={{ value: 3.2, positive: false }}
          description="Requiring immediate attention"
          delay={0.1}
        />
        <StatCard
          title="Network Hospitals"
          value="3"
          icon={Building2}
          description="Active in federated network"
          delay={0.2}
        />
        <StatCard
          title="Fairness Score"
          value="94.2"
          suffix="%"
          icon={Scale}
          trend={{ value: 1.8, positive: true }}
          description="Model bias assessment"
          delay={0.3}
        />
      </div>

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
              High-risk patient identification over time
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-chart-1" />
              <span className="text-xs text-muted-foreground">Diabetes</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-chart-2" />
              <span className="text-xs text-muted-foreground">Heart Disease</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full bg-chart-3" />
              <span className="text-xs text-muted-foreground">Hypertension</span>
            </div>
          </div>
        </div>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={diseaseTrendData}>
              <defs>
                <linearGradient id="diabetesGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="heartGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2dd4bf" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#2dd4bf" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="hyperGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis
                dataKey="month"
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
                dataKey="diabetes"
                stroke="#22d3ee"
                strokeWidth={2}
                fill="url(#diabetesGradient)"
                name="Diabetes"
              />
              <Area
                type="monotone"
                dataKey="heart"
                stroke="#2dd4bf"
                strokeWidth={2}
                fill="url(#heartGradient)"
                name="Heart Disease"
              />
              <Area
                type="monotone"
                dataKey="hypertension"
                stroke="#818cf8"
                strokeWidth={2}
                fill="url(#hyperGradient)"
                name="Hypertension"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Two Column Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* High-Risk Patient Distribution */}
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
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={highRiskStats}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {highRiskStats.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-2 gap-4 mt-4">
            {highRiskStats.map((item) => (
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
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hospitalContribution}>
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
        {/* Age Group Risk Analysis */}
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
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ageGroupRisk}>
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

        {/* Gender-Based Model Comparison */}
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
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={genderComparison}>
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
          {fairnessData.map((item, index) => (
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
    </div>
  )
}
