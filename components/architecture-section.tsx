"use client"

import { motion } from "framer-motion"
import { Building2, Server, ArrowRight, RefreshCw } from "lucide-react"

const hospitals = [
  { name: "Hospital A", position: "left" },
  { name: "Hospital B", position: "right-top" },
  { name: "Hospital C", position: "right-bottom" },
]

export function ArchitectureSection() {
  return (
    <section className="relative py-24 overflow-hidden">
      <div className="absolute inset-0 gradient-animate" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm text-primary mb-4">
            <RefreshCw className="h-4 w-4" />
            How It Works
          </span>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-foreground mb-4 text-balance">
            Federated Learning <span className="text-primary">Architecture</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto text-pretty">
            Secure model synchronization across healthcare institutions while keeping patient data local.
          </p>
        </motion.div>

        {/* Architecture Diagram */}
        <div className="relative max-w-4xl mx-auto">
          {/* Central Server */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="relative mx-auto w-fit mb-16"
          >
            <div className="absolute inset-0 bg-primary/30 rounded-2xl blur-xl animate-pulse" />
            <div className="relative glass-card rounded-2xl p-8 glow-cyan">
              <div className="flex flex-col items-center gap-4">
                <div className="p-4 rounded-xl bg-primary/20">
                  <Server className="h-12 w-12 text-primary" />
                </div>
                <div className="text-center">
                  <h3 className="text-xl font-bold text-foreground">Central AI Server</h3>
                  <p className="text-sm text-muted-foreground">Model Aggregation</p>
                </div>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                  className="absolute -top-2 -right-2 p-2 rounded-full bg-success/20"
                >
                  <RefreshCw className="h-4 w-4 text-success" />
                </motion.div>
              </div>
            </div>
          </motion.div>

          {/* Connection Lines & Hospital Nodes */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {hospitals.map((hospital, index) => (
              <motion.div
                key={hospital.name}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: 0.4 + index * 0.2 }}
                className="relative"
              >
                {/* Connection Arrow */}
                <div className="absolute left-1/2 -top-8 transform -translate-x-1/2 hidden md:block">
                  <motion.div
                    animate={{ y: [0, -5, 0] }}
                    transition={{ duration: 1.5, repeat: Infinity, delay: index * 0.3 }}
                  >
                    <ArrowRight className="h-6 w-6 text-primary rotate-[-90deg]" />
                  </motion.div>
                </div>

                <div className="glass-card rounded-2xl p-6 hover:glow-border transition-all duration-300">
                  <div className="flex flex-col items-center gap-4">
                    <div className="p-3 rounded-xl bg-accent/20">
                      <Building2 className="h-8 w-8 text-accent" />
                    </div>
                    <div className="text-center">
                      <h4 className="text-lg font-semibold text-foreground">{hospital.name}</h4>
                      <p className="text-sm text-muted-foreground">Local Training</p>
                    </div>
                    
                    {/* Status Indicators */}
                    <div className="flex gap-2">
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-success/20 text-xs text-success">
                        <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
                        Connected
                      </span>
                      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-primary/20 text-xs text-primary">
                        <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                        Training
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full">
                      <div className="flex justify-between text-xs text-muted-foreground mb-1">
                        <span>Model Update</span>
                        <motion.span
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                        >
                          {85 + index * 5}%
                        </motion.span>
                      </div>
                      <div className="h-2 bg-secondary rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          whileInView={{ width: `${85 + index * 5}%` }}
                          viewport={{ once: true }}
                          transition={{ duration: 1, delay: 0.5 + index * 0.2 }}
                          className="h-full bg-gradient-to-r from-primary to-accent rounded-full"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Data Flow Explanation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 1 }}
            className="mt-12 glass-card rounded-xl p-6 text-center"
          >
            <p className="text-muted-foreground">
              <span className="text-primary font-semibold">Step 1:</span> Each hospital trains the model locally on their patient data
              <span className="mx-4 text-border">|</span>
              <span className="text-accent font-semibold">Step 2:</span> Only model weights are sent to the central server
              <span className="mx-4 text-border">|</span>
              <span className="text-success font-semibold">Step 3:</span> Aggregated model is distributed back to all hospitals
            </p>
          </motion.div>
        </div>
      </div>
    </section>
  )
}
