"use client"

import { motion, AnimatePresence } from "framer-motion"
import { useState } from "react"
import {
  AlertTriangle,
  AlertOctagon,
  Shield,
  ShieldCheck,
  Activity,
  Heart,
  Eye,
  ChevronDown,
  ChevronUp,
  Stethoscope,
  ClipboardList,
  Siren,
  ArrowUpRight,
  Network,
  Info,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
} from "lucide-react"
import type { ClinicalSupportResponse } from "@/lib/api"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ClinicalSupportPanelProps {
  data: ClinicalSupportResponse
  isLoading?: boolean
}

// ---------------------------------------------------------------------------
// Severity Config
// ---------------------------------------------------------------------------

type SeverityLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
type PriorityLevel = "ROUTINE" | "ELEVATED" | "URGENT" | "EMERGENCY"

const SEVERITY_CONFIG: Record<
  SeverityLevel,
  {
    label: string
    icon: React.ElementType
    headerGradient: string
    borderColor: string
    textColor: string
    badgeBg: string
    glowClass: string
    pulseClass: string
    priorityBg: string
  }
> = {
  LOW: {
    label: "LOW RISK",
    icon: ShieldCheck,
    headerGradient: "from-emerald-900/60 via-green-900/40 to-transparent",
    borderColor: "border-emerald-500/30",
    textColor: "text-emerald-400",
    badgeBg: "bg-emerald-500/15",
    glowClass: "shadow-[0_0_30px_rgba(52,211,153,0.15)]",
    pulseClass: "",
    priorityBg: "bg-emerald-500/20 text-emerald-300",
  },
  MEDIUM: {
    label: "MEDIUM RISK",
    icon: AlertTriangle,
    headerGradient: "from-amber-900/60 via-yellow-900/40 to-transparent",
    borderColor: "border-amber-500/30",
    textColor: "text-amber-400",
    badgeBg: "bg-amber-500/15",
    glowClass: "shadow-[0_0_30px_rgba(251,191,36,0.15)]",
    pulseClass: "",
    priorityBg: "bg-amber-500/20 text-amber-300",
  },
  HIGH: {
    label: "HIGH RISK",
    icon: AlertTriangle,
    headerGradient: "from-red-900/70 via-rose-900/50 to-transparent",
    borderColor: "border-red-500/40",
    textColor: "text-red-400",
    badgeBg: "bg-red-500/15",
    glowClass: "shadow-[0_0_40px_rgba(239,68,68,0.2)]",
    pulseClass: "animate-pulse-glow-red",
    priorityBg: "bg-red-500/20 text-red-300",
  },
  CRITICAL: {
    label: "⚡ CRITICAL — EMERGENCY",
    icon: AlertOctagon,
    headerGradient: "from-red-950/90 via-rose-900/70 to-transparent",
    borderColor: "border-red-500/60",
    textColor: "text-red-300",
    badgeBg: "bg-red-600/20",
    glowClass: "shadow-[0_0_60px_rgba(239,68,68,0.35),0_0_100px_rgba(239,68,68,0.15)]",
    pulseClass: "animate-pulse-critical",
    priorityBg: "bg-red-600/30 text-red-200",
  },
}

const PRIORITY_ICONS: Record<PriorityLevel, React.ElementType> = {
  ROUTINE:   Clock,
  ELEVATED:  Activity,
  URGENT:    Zap,
  EMERGENCY: Siren,
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Animated expandable accordion section */
function AccordionSection({
  title,
  icon: Icon,
  items,
  iconColor,
  itemPrefix,
  defaultOpen = false,
  accentColor,
}: {
  title: string
  icon: React.ElementType
  items: string[]
  iconColor: string
  itemPrefix?: React.ReactNode
  defaultOpen?: boolean
  accentColor: string
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="rounded-xl border border-white/8 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setOpen((o) => !o)}
        className={`w-full flex items-center justify-between px-5 py-4 
          bg-white/4 hover:bg-white/7 transition-all duration-200 group`}
      >
        <div className="flex items-center gap-3">
          <div className={`p-1.5 rounded-lg ${accentColor}`}>
            <Icon className={`h-4 w-4 ${iconColor}`} />
          </div>
          <span className="text-sm font-semibold text-foreground">{title}</span>
          <span className="text-xs text-muted-foreground bg-white/8 px-2 py-0.5 rounded-full">
            {items.length}
          </span>
        </div>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
        </motion.div>
      </button>

      {/* Content */}
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-4 pt-2 space-y-2">
              {items.map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.04 }}
                  className="flex items-start gap-3 py-1.5"
                >
                  {itemPrefix ?? (
                    <div className={`mt-0.5 h-1.5 w-1.5 rounded-full flex-shrink-0 ${iconColor.replace("text-", "bg-")}`} />
                  )}
                  <span className="text-sm text-muted-foreground leading-relaxed">{item}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/** Primary contributors tag list */
function ContributorBadges({ factors }: { factors: string[] }) {
  const colors = [
    "bg-rose-500/15 text-rose-300 border-rose-500/25",
    "bg-orange-500/15 text-orange-300 border-orange-500/25",
    "bg-yellow-500/15 text-yellow-300 border-yellow-500/25",
    "bg-purple-500/15 text-purple-300 border-purple-500/25",
  ]
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {factors.map((f, i) => (
        <span
          key={i}
          className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${colors[i % colors.length]}`}
        >
          <span className="h-1.5 w-1.5 rounded-full bg-current" />
          {f}
        </span>
      ))}
    </div>
  )
}

/** Timeline step for next steps */
function NextStepTimeline({ steps, severity }: { steps: string[]; severity: SeverityLevel }) {
  const cfg = SEVERITY_CONFIG[severity]
  return (
    <div className="relative">
      <div className="absolute left-[18px] top-4 bottom-4 w-px bg-white/10" />
      <div className="space-y-4">
        {steps.map((step, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.08 }}
            className="flex items-start gap-4"
          >
            <div
              className={`relative z-10 flex-shrink-0 h-9 w-9 rounded-full ${cfg.badgeBg} border ${cfg.borderColor}
                flex items-center justify-center text-xs font-bold ${cfg.textColor}`}
            >
              {i + 1}
            </div>
            <div className="flex-1 pt-1.5">
              <p className="text-sm text-foreground leading-relaxed">{step}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

function LoadingSkeleton() {
  return (
    <div className="glass-card rounded-2xl p-8 border border-primary/20 space-y-6 animate-pulse">
      <div className="flex items-center gap-4">
        <div className="h-12 w-12 rounded-xl bg-white/10" />
        <div className="space-y-2 flex-1">
          <div className="h-5 bg-white/10 rounded w-64" />
          <div className="h-3 bg-white/5 rounded w-40" />
        </div>
      </div>
      <div className="h-16 bg-white/5 rounded-xl" />
      <div className="grid grid-cols-2 gap-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-32 bg-white/5 rounded-xl" />
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ClinicalSupportPanel({ data, isLoading }: ClinicalSupportPanelProps) {
  if (isLoading) return <LoadingSkeleton />
  if (!data) return null

  const severity = data.severity_level as SeverityLevel
  const priority = data.emergency_priority as PriorityLevel
  const cfg = SEVERITY_CONFIG[severity]
  const SeverityIcon = cfg.icon
  const PriorityIcon = PRIORITY_ICONS[priority]
  const isCritical = severity === "CRITICAL"
  const isHigh = severity === "HIGH"

  return (
    <motion.div
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={`relative rounded-2xl overflow-hidden border ${cfg.borderColor} ${cfg.glowClass} ${isCritical ? cfg.pulseClass : ""}`}
    >
      {/* Critical pulse ring overlay */}
      {isCritical && (
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute inset-0 animate-ping-slow rounded-2xl border-2 border-red-500/20" />
        </div>
      )}

      {/* Background */}
      <div
        className={`absolute inset-0 bg-gradient-to-b ${cfg.headerGradient} pointer-events-none`}
      />
      <div className="absolute inset-0 glass-card pointer-events-none" />

      <div className="relative z-10 p-6 space-y-6">
        {/* ── Header ───────────────────────────────────────────────────────── */}
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-4">
            <div
              className={`relative p-3 rounded-xl ${cfg.badgeBg} border ${cfg.borderColor} flex-shrink-0`}
            >
              <Stethoscope className={`h-6 w-6 ${cfg.textColor}`} />
              {(isCritical || isHigh) && (
                <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-red-500 animate-ping" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <h2 className="text-xl font-bold text-foreground">
                  AI Clinical Emergency Support Assistant
                </h2>
              </div>
              <p className="text-xs text-muted-foreground">
                Nurse Assistance Intelligence Layer · Clinical Decision Support
              </p>
            </div>
          </div>

          {/* Severity badge */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 260, damping: 18, delay: 0.2 }}
            className={`flex-shrink-0 flex items-center gap-2 px-4 py-2 rounded-full 
              ${cfg.badgeBg} border ${cfg.borderColor}`}
          >
            <SeverityIcon className={`h-4 w-4 ${cfg.textColor}`} />
            <span className={`text-sm font-bold ${cfg.textColor}`}>{cfg.label}</span>
          </motion.div>
        </div>

        {/* ── Priority + Contributors row ───────────────────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Emergency priority */}
          <div className="glass rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <PriorityIcon className={`h-4 w-4 ${cfg.textColor}`} />
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Emergency Priority Level
              </span>
            </div>
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-bold ${cfg.priorityBg}`}>
              {priority}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {priority === "EMERGENCY" && "Activate rapid response team immediately"}
              {priority === "URGENT" && "Immediate physician evaluation required"}
              {priority === "ELEVATED" && "Physician review advised within 48–72 hrs"}
              {priority === "ROUTINE" && "Standard monitoring schedule applies"}
            </p>
          </div>

          {/* Primary contributors */}
          <div className="glass rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <Activity className={`h-4 w-4 ${cfg.textColor}`} />
              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                Primary Risk Contributors
              </span>
            </div>
            <ContributorBadges factors={data.primary_contributors} />
          </div>
        </div>

        {/* ── AI Summary ───────────────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className={`rounded-xl p-5 ${cfg.badgeBg} border ${cfg.borderColor}`}
        >
          <div className="flex items-start gap-3">
            <div className={`p-2 rounded-lg bg-white/8 flex-shrink-0 mt-0.5`}>
              <Stethoscope className={`h-4 w-4 ${cfg.textColor}`} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">
                AI Clinical Summary
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {data.ai_summary}
              </p>
            </div>
          </div>
        </motion.div>

        {/* ── Accordion sections ───────────────────────────────────────────── */}
        <div className="space-y-3">
          {/* Recommended Actions */}
          <AccordionSection
            title="Recommended Clinical Actions"
            icon={ClipboardList}
            items={data.recommendations}
            iconColor={cfg.textColor}
            accentColor={cfg.badgeBg}
            defaultOpen={true}
          />

          {/* Monitoring Checklist */}
          <AccordionSection
            title="Monitoring Checklist"
            icon={Activity}
            items={data.monitoring_checklist}
            iconColor="text-cyan-400"
            accentColor="bg-cyan-500/10"
            defaultOpen={isHigh || isCritical}
            itemPrefix={<CheckCircle2 className="h-4 w-4 text-cyan-500/70 flex-shrink-0 mt-0.5" />}
          />

          {/* Escalation Criteria */}
          <AccordionSection
            title="Escalation Warning Criteria"
            icon={Siren}
            items={data.escalation_criteria}
            iconColor="text-rose-400"
            accentColor="bg-rose-500/10"
            defaultOpen={isHigh || isCritical}
            itemPrefix={<XCircle className="h-4 w-4 text-rose-500/70 flex-shrink-0 mt-0.5" />}
          />
        </div>

        {/* ── Next Steps Timeline ──────────────────────────────────────────── */}
        <div className="glass rounded-xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <ArrowUpRight className={`h-4 w-4 ${cfg.textColor}`} />
            <h3 className="text-sm font-semibold text-foreground">
              Suggested Next Steps
            </h3>
          </div>
          <NextStepTimeline steps={data.next_steps} severity={severity} />
        </div>

        {/* ── Federated attribution footer ─────────────────────────────────── */}
        <div className="rounded-xl border border-primary/15 bg-primary/5 p-4">
          <div className="flex items-start gap-3">
            <Network className="h-4 w-4 text-primary flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-semibold text-primary mb-0.5">
                Federated Learning Intelligence
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {data.federated_note}
              </p>
            </div>
          </div>
        </div>

        {/* ── Disclaimer ───────────────────────────────────────────────────── */}
        <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
          <div className="flex items-start gap-3">
            <Info className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-amber-200/70 leading-relaxed">
              {data.disclaimer}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
