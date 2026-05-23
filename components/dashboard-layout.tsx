"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { 
  Shield, 
  LayoutDashboard, 
  Activity, 
  Brain, 
  BarChart3,
  Settings,
  ChevronRight
} from "lucide-react"
import { usePathname } from "next/navigation"

const sidebarLinks = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "Prediction", href: "/prediction", icon: Activity },
  { name: "Explainable AI", href: "/explainable", icon: Brain },
  { name: "Analytics", href: "/analytics", icon: BarChart3 },
]

export function DashboardSidebar() {
  const pathname = usePathname()

  return (
    <motion.aside
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="fixed left-0 top-0 h-screen w-64 glass border-r border-border z-40 hidden lg:block"
    >
      <div className="flex flex-col h-full">
        {/* Logo */}
        <div className="p-6 border-b border-border">
          <Link href="/" className="flex items-center gap-2">
            <div className="relative">
              <div className="absolute inset-0 blur-lg bg-primary/50" />
              <Shield className="relative h-8 w-8 text-primary" />
            </div>
            <span className="text-lg font-bold text-foreground">
              Medi<span className="text-primary">Shield</span>
            </span>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {sidebarLinks.map((link) => {
            const isActive = pathname === link.href
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group ${
                  isActive
                    ? "bg-primary/20 text-primary glow-border"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                }`}
              >
                <link.icon className={`h-5 w-5 ${isActive ? "text-primary" : ""}`} />
                <span className="font-medium">{link.name}</span>
                {isActive && (
                  <ChevronRight className="h-4 w-4 ml-auto" />
                )}
              </Link>
            )
          })}
        </nav>

        {/* Bottom Section */}
        <div className="p-4 border-t border-border space-y-2">
          <Link
            href="/settings"
            className="flex items-center gap-3 px-4 py-3 rounded-xl text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-all duration-300"
          >
            <Settings className="h-5 w-5" />
            <span className="font-medium">Settings</span>
          </Link>
        </div>
      </div>
    </motion.aside>
  )
}

export function DashboardHeader() {
  return (
    <motion.header
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 right-0 left-0 lg:left-64 h-16 glass border-b border-border z-30"
    >
      <div className="flex items-center justify-between h-full px-6">
        <div className="flex items-center gap-4">
          <Link href="/" className="lg:hidden flex items-center gap-2">
            <Shield className="h-6 w-6 text-primary" />
            <span className="font-bold text-foreground">MediShield</span>
          </Link>
          <div className="hidden sm:block">
            <h1 className="text-lg font-semibold text-foreground">Federated Learning Dashboard</h1>
            <p className="text-xs text-muted-foreground">Real-time model training insights</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
        </div>
      </div>
    </motion.header>
  )
}
