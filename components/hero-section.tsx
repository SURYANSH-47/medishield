"use client"

import { motion, useMotionValue, useTransform, animate, useMotionValueEvent } from "framer-motion"
import { useEffect, useRef, useState } from "react"
import Link from "next/link"
import { ArrowRight, Play, Shield, Sparkles } from "lucide-react"

// Animated neural network background
function NeuralNetworkBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let animationId: number
    const nodes: { x: number; y: number; vx: number; vy: number }[] = []
    const nodeCount = 50

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resize()
    window.addEventListener("resize", resize)

    // Initialize nodes
    for (let i = 0; i < nodeCount; i++) {
      nodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
      })
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update and draw nodes
      nodes.forEach((node, i) => {
        node.x += node.vx
        node.y += node.vy

        if (node.x < 0 || node.x > canvas.width) node.vx *= -1
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1

        // Draw node
        ctx.beginPath()
        ctx.arc(node.x, node.y, 2, 0, Math.PI * 2)
        ctx.fillStyle = "rgba(34, 211, 238, 0.5)"
        ctx.fill()

        // Draw connections
        nodes.forEach((other, j) => {
          if (i === j) return
          const dist = Math.hypot(node.x - other.x, node.y - other.y)
          if (dist < 150) {
            ctx.beginPath()
            ctx.moveTo(node.x, node.y)
            ctx.lineTo(other.x, other.y)
            ctx.strokeStyle = `rgba(34, 211, 238, ${0.1 * (1 - dist / 150)})`
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        })
      })

      animationId = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      window.removeEventListener("resize", resize)
      cancelAnimationFrame(animationId)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none opacity-60"
    />
  )
}

// Floating healthcare card component
function FloatingCard({
  children,
  delay,
  x,
  y,
}: {
  children: React.ReactNode
  delay: number
  x: string
  y: string
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{
        opacity: 1,
        scale: 1,
        y: [0, -10, 0],
      }}
      transition={{
        opacity: { duration: 0.8, delay },
        scale: { duration: 0.8, delay },
        y: { duration: 3, repeat: Infinity, ease: "easeInOut", delay },
      }}
      className="absolute glass-card rounded-xl p-4 hidden lg:block"
      style={{ left: x, top: y }}
    >
      {children}
    </motion.div>
  )
}

// Animated counter
function AnimatedCounter({ value, suffix = "" }: { value: number; suffix?: string }) {
  const count = useMotionValue(0)
  const rounded = useTransform(count, (latest) => Math.round(latest))
  const [displayValue, setDisplayValue] = useState(0)

  useMotionValueEvent(rounded, "change", (latest) => {
    setDisplayValue(latest)
  })

  useEffect(() => {
    const controls = animate(count, value, { duration: 2, ease: "easeOut" })
    return controls.stop
  }, [count, value])

  return (
    <span>
      {displayValue}
      {suffix}
    </span>
  )
}

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
      {/* Animated background */}
      <div className="absolute inset-0 neural-bg" />
      <NeuralNetworkBackground />
      
      {/* Gradient orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-accent/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />

      {/* Floating cards */}
      <FloatingCard delay={0.5} x="10%" y="20%">
        <div className="flex items-center gap-2 text-sm text-foreground">
          <div className="h-2 w-2 rounded-full bg-success animate-pulse" />
          <span>Hospital A Connected</span>
        </div>
      </FloatingCard>

      <FloatingCard delay={0.8} x="75%" y="25%">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-primary/20 flex items-center justify-center">
            <Shield className="h-5 w-5 text-primary" />
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Data Privacy</div>
            <div className="text-sm font-semibold text-foreground">100% Secure</div>
          </div>
        </div>
      </FloatingCard>

      <FloatingCard delay={1.1} x="5%" y="60%">
        <div className="text-sm">
          <div className="text-muted-foreground text-xs mb-1">Model Accuracy</div>
          <div className="text-2xl font-bold text-primary">
            <AnimatedCounter value={94} suffix="%" />
          </div>
        </div>
      </FloatingCard>

      <FloatingCard delay={1.4} x="80%" y="65%">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-warning" />
          <span className="text-sm text-foreground">AI Prediction Active</span>
        </div>
      </FloatingCard>

      {/* Main content */}
      <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-6"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm text-primary">
            <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
            Next-Gen Healthcare AI Platform
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold leading-tight mb-6 text-balance"
        >
          <span className="text-foreground">Privacy-Preserving AI for</span>
          <br />
          <span className="text-primary text-glow">Early Disease Detection</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="text-lg sm:text-xl text-muted-foreground max-w-3xl mx-auto mb-10 text-pretty"
        >
          Hospitals collaboratively train AI models without sharing sensitive patient data. 
          Powered by federated learning for secure, accurate disease risk prediction.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4"
        >
          <Link
            href="/dashboard"
            className="group flex items-center gap-2 px-8 py-4 rounded-xl bg-primary text-primary-foreground font-semibold hover:bg-primary/90 transition-all duration-300 glow-cyan"
          >
            View Dashboard
            <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
          </Link>
          <Link
            href="/prediction"
            className="group flex items-center gap-2 px-8 py-4 rounded-xl glass-card text-foreground font-semibold hover:bg-secondary/50 transition-all duration-300 glow-border"
          >
            <Play className="h-5 w-5" />
            Run Prediction
          </Link>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1 }}
          className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8"
        >
          {[
            { value: 50, suffix: "+", label: "Partner Hospitals" },
            { value: 2, suffix: "M+", label: "Patients Protected" },
            { value: 99.9, suffix: "%", label: "Data Privacy" },
            { value: 94, suffix: "%", label: "Prediction Accuracy" },
          ].map((stat, i) => (
            <div key={i} className="glass-card rounded-xl p-4">
              <div className="text-2xl sm:text-3xl font-bold text-primary">
                <AnimatedCounter value={stat.value} suffix={stat.suffix} />
              </div>
              <div className="text-sm text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
