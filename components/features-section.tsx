"use client"

import { motion } from "framer-motion"
import { 
  Brain, 
  Shield, 
  Activity, 
  Lock, 
  BarChart3, 
  Scale,
  Zap,
  Eye
} from "lucide-react"

const features = [
  {
    icon: Brain,
    title: "Federated Learning",
    description: "Train AI models across multiple hospitals without centralizing sensitive patient data.",
    color: "text-primary",
    bgColor: "bg-primary/10",
  },
  {
    icon: Activity,
    title: "Disease Risk Prediction",
    description: "Advanced algorithms predict diabetes, heart disease, and other conditions with high accuracy.",
    color: "text-accent",
    bgColor: "bg-accent/10",
  },
  {
    icon: Eye,
    title: "Explainable AI",
    description: "Understand exactly why predictions are made with SHAP-based feature importance analysis.",
    color: "text-info",
    bgColor: "bg-info/10",
  },
  {
    icon: Lock,
    title: "Privacy Protection",
    description: "Patient data never leaves the hospital. Only model updates are shared securely.",
    color: "text-success",
    bgColor: "bg-success/10",
  },
  {
    icon: BarChart3,
    title: "Real-Time Analytics",
    description: "Monitor model performance, hospital participation, and risk distributions in real-time.",
    color: "text-warning",
    bgColor: "bg-warning/10",
  },
  {
    icon: Scale,
    title: "Bias Monitoring",
    description: "Ensure fair predictions across demographics with continuous fairness assessment.",
    color: "text-destructive",
    bgColor: "bg-destructive/10",
  },
]

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: "easeOut" },
  },
}

export function FeaturesSection() {
  return (
    <section className="relative py-24 overflow-hidden">
      <div className="absolute inset-0 neural-bg opacity-30" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm text-primary mb-4">
            <Zap className="h-4 w-4" />
            Powerful Capabilities
          </span>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-foreground mb-4 text-balance">
            Enterprise-Grade <span className="text-primary">Healthcare AI</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto text-pretty">
            Built for hospitals and healthcare systems that prioritize both innovation and patient privacy.
          </p>
        </motion.div>

        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              variants={itemVariants}
              whileHover={{ scale: 1.02, y: -5 }}
              className="group glass-card rounded-2xl p-6 hover:glow-border transition-all duration-300"
            >
              <div className={`inline-flex p-3 rounded-xl ${feature.bgColor} mb-4 group-hover:scale-110 transition-transform duration-300`}>
                <feature.icon className={`h-6 w-6 ${feature.color}`} />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">
                {feature.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
