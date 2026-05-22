"use client"

import { motion } from "framer-motion"
import { CheckCircle, Shield, HeartPulse, AlertTriangle, Users } from "lucide-react"

const benefits = [
  {
    icon: Shield,
    title: "Secure Collaboration",
    description: "Multiple healthcare institutions can collaborate on AI development without exposing patient records to external parties.",
    stats: "100% Data Privacy",
  },
  {
    icon: HeartPulse,
    title: "Early Disease Detection",
    description: "Identify health risks before they become critical, enabling preventive care and better patient outcomes.",
    stats: "6 Months Earlier",
  },
  {
    icon: AlertTriangle,
    title: "Reduced Healthcare Risks",
    description: "Proactive risk assessment helps healthcare providers allocate resources efficiently and reduce adverse events.",
    stats: "40% Risk Reduction",
  },
  {
    icon: Users,
    title: "Privacy-Preserving AI",
    description: "Leverage collective medical knowledge while maintaining strict HIPAA compliance and patient confidentiality.",
    stats: "HIPAA Compliant",
  },
]

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.2 },
  },
}

const itemVariants = {
  hidden: { opacity: 0, x: -20 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { duration: 0.6, ease: "easeOut" },
  },
}

export function BenefitsSection() {
  return (
    <section className="relative py-24 overflow-hidden">
      <div className="absolute inset-0 neural-bg opacity-20" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left Content */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm text-primary mb-4">
              <CheckCircle className="h-4 w-4" />
              Key Benefits
            </span>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-foreground mb-6 text-balance">
              Transform Healthcare with <span className="text-primary">Federated AI</span>
            </h2>
            <p className="text-lg text-muted-foreground mb-8 text-pretty">
              MediShield AI brings the power of collaborative machine learning to healthcare 
              while ensuring that sensitive patient information never leaves its secure environment.
            </p>

            {/* Mini Stats */}
            <div className="grid grid-cols-2 gap-4">
              <div className="glass-card rounded-xl p-4">
                <div className="text-3xl font-bold text-primary mb-1">94%</div>
                <div className="text-sm text-muted-foreground">Prediction Accuracy</div>
              </div>
              <div className="glass-card rounded-xl p-4">
                <div className="text-3xl font-bold text-accent mb-1">50+</div>
                <div className="text-sm text-muted-foreground">Partner Hospitals</div>
              </div>
            </div>
          </motion.div>

          {/* Right Content - Benefits List */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="space-y-6"
          >
            {benefits.map((benefit, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ x: 10 }}
                className="glass-card rounded-xl p-6 hover:glow-border transition-all duration-300"
              >
                <div className="flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="p-3 rounded-xl bg-primary/20">
                      <benefit.icon className="h-6 w-6 text-primary" />
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-lg font-semibold text-foreground">
                        {benefit.title}
                      </h3>
                      <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-1 rounded-full">
                        {benefit.stats}
                      </span>
                    </div>
                    <p className="text-muted-foreground text-sm leading-relaxed">
                      {benefit.description}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  )
}
