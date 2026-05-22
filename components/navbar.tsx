"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { Menu, X, Shield, Activity } from "lucide-react"
import { useState } from "react"

const navLinks = [
  { name: "Dashboard", href: "/dashboard" },
  { name: "Prediction", href: "/prediction" },
  { name: "Explainable AI", href: "/explainable" },
  { name: "Analytics", href: "/analytics" },
]

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="fixed top-0 left-0 right-0 z-50 glass"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="relative">
              <div className="absolute inset-0 blur-lg bg-primary/50" />
              <Shield className="relative h-8 w-8 text-primary" />
            </div>
            <span className="text-xl font-bold text-foreground">
              Medi<span className="text-primary">Shield</span> AI
            </span>
          </Link>

          <div className="hidden md:flex items-center gap-8">
            {navLinks.map((link) => (
              <Link
                key={link.name}
                href={link.href}
                className="text-sm text-muted-foreground hover:text-primary transition-colors duration-300"
              >
                {link.name}
              </Link>
            ))}
            <Link
              href="/dashboard"
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-all duration-300 glow-cyan"
            >
              Launch App
            </Link>
          </div>

          <button
            onClick={() => setIsOpen(!isOpen)}
            className="md:hidden text-foreground"
          >
            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="md:hidden glass"
        >
          <div className="px-4 py-4 space-y-3">
            {navLinks.map((link) => (
              <Link
                key={link.name}
                href={link.href}
                className="block text-sm text-muted-foreground hover:text-primary transition-colors"
                onClick={() => setIsOpen(false)}
              >
                {link.name}
              </Link>
            ))}
            <Link
              href="/dashboard"
              className="block w-full px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium text-center"
              onClick={() => setIsOpen(false)}
            >
              Launch App
            </Link>
          </div>
        </motion.div>
      )}
    </motion.nav>
  )
}
