"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { Shield, Github, Twitter, Linkedin, Mail } from "lucide-react"

const footerLinks = {
  product: [
    { name: "Dashboard", href: "/dashboard" },
    { name: "Prediction", href: "/prediction" },
    { name: "Explainable AI", href: "/explainable" },
    { name: "Analytics", href: "/analytics" },
  ],
  company: [
    { name: "About Us", href: "#" },
    { name: "Careers", href: "#" },
    { name: "Blog", href: "#" },
    { name: "Press", href: "#" },
  ],
  resources: [
    { name: "Documentation", href: "#" },
    { name: "API Reference", href: "#" },
    { name: "Support", href: "#" },
    { name: "Status", href: "#" },
  ],
  legal: [
    { name: "Privacy Policy", href: "#" },
    { name: "Terms of Service", href: "#" },
    { name: "HIPAA Compliance", href: "#" },
    { name: "Security", href: "#" },
  ],
}

const socialLinks = [
  { name: "GitHub", icon: Github, href: "#" },
  { name: "Twitter", icon: Twitter, href: "#" },
  { name: "LinkedIn", icon: Linkedin, href: "#" },
  { name: "Email", icon: Mail, href: "#" },
]

export function Footer() {
  return (
    <footer className="relative pt-24 pb-8 overflow-hidden">
      <div className="absolute inset-0 gradient-animate opacity-30" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
      
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* CTA Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="glass-card rounded-2xl p-8 md:p-12 mb-16 text-center glow-border"
        >
          <h2 className="text-2xl md:text-4xl font-bold text-foreground mb-4 text-balance">
            Ready to Transform Your Healthcare AI?
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto mb-8 text-pretty">
            Join leading healthcare institutions using MediShield AI for privacy-preserving disease prediction.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/dashboard"
              className="px-8 py-4 rounded-xl bg-primary text-primary-foreground font-semibold hover:bg-primary/90 transition-all duration-300 glow-cyan"
            >
              Get Started Free
            </Link>
            <Link
              href="#"
              className="px-8 py-4 rounded-xl glass text-foreground font-semibold hover:bg-secondary/50 transition-all duration-300"
            >
              Schedule Demo
            </Link>
          </div>
        </motion.div>

        {/* Footer Links */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-8 mb-12">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <Link href="/" className="flex items-center gap-2 mb-4">
              <div className="relative">
                <div className="absolute inset-0 blur-lg bg-primary/50" />
                <Shield className="relative h-8 w-8 text-primary" />
              </div>
              <span className="text-lg font-bold text-foreground">
                Medi<span className="text-primary">Shield</span>
              </span>
            </Link>
            <p className="text-sm text-muted-foreground mb-4">
              Privacy-preserving AI for healthcare intelligence.
            </p>
            <div className="flex gap-3">
              {socialLinks.map((social) => (
                <a
                  key={social.name}
                  href={social.href}
                  className="p-2 rounded-lg glass hover:bg-primary/20 transition-colors"
                  aria-label={social.name}
                >
                  <social.icon className="h-4 w-4 text-muted-foreground hover:text-primary transition-colors" />
                </a>
              ))}
            </div>
          </div>

          {/* Links */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h3 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-4">
                {category}
              </h3>
              <ul className="space-y-2">
                {links.map((link) => (
                  <li key={link.name}>
                    <Link
                      href={link.href}
                      className="text-sm text-muted-foreground hover:text-primary transition-colors"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-muted-foreground">
            &copy; {new Date().getFullYear()} MediShield AI. All rights reserved.
          </p>
          <div className="flex items-center gap-6">
            <span className="text-xs text-muted-foreground flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-success animate-pulse" />
              All systems operational
            </span>
            <span className="text-xs text-muted-foreground">
              HIPAA Compliant
            </span>
            <span className="text-xs text-muted-foreground">
              SOC 2 Certified
            </span>
          </div>
        </div>
      </div>
    </footer>
  )
}
