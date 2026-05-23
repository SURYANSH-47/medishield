"use client"

import { motion } from "framer-motion"
import { useState } from "react"
import { toast } from "sonner"
import {
  Bell,
  Lock,
  Database,
  Eye,
  Moon,
  Sun,
  Save,
  ChevronRight,
} from "lucide-react"

interface Setting {
  id: string
  label: string
  description: string
  icon: React.ElementType
  type: "toggle" | "select" | "text"
  value: any
  options?: { label: string; value: any }[]
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<Record<string, any>>({
    notifications_enabled: true,
    email_alerts: true,
    predictions_threshold: "75",
    theme: "dark",
    data_retention: "90",
    privacy_mode: false,
    auto_refresh: true,
  })

  const [hasChanges, setHasChanges] = useState(false)

  const settingsList: Setting[] = [
    {
      id: "notifications_enabled",
      label: "Push Notifications",
      description: "Receive notifications for model training updates",
      icon: Bell,
      type: "toggle",
      value: settings.notifications_enabled,
    },
    {
      id: "email_alerts",
      label: "Email Alerts",
      description: "Get email alerts for high-risk patient predictions",
      icon: Bell,
      type: "toggle",
      value: settings.email_alerts,
    },
    {
      id: "predictions_threshold",
      label: "Risk Threshold",
      description: "Alert threshold for high-risk predictions (%)",
      icon: Eye,
      type: "select",
      value: settings.predictions_threshold,
      options: [
        { label: "50%", value: "50" },
        { label: "65%", value: "65" },
        { label: "75%", value: "75" },
        { label: "85%", value: "85" },
      ],
    },
    {
      id: "theme",
      label: "Theme",
      description: "Choose your preferred theme",
      icon: Moon,
      type: "select",
      value: settings.theme,
      options: [
        { label: "Dark", value: "dark" },
        { label: "Light", value: "light" },
        { label: "Auto", value: "auto" },
      ],
    },
    {
      id: "data_retention",
      label: "Data Retention Period",
      description: "How long to keep historical analytics data (days)",
      icon: Database,
      type: "select",
      value: settings.data_retention,
      options: [
        { label: "30 days", value: "30" },
        { label: "60 days", value: "60" },
        { label: "90 days", value: "90" },
        { label: "180 days", value: "180" },
      ],
    },
    {
      id: "privacy_mode",
      label: "Privacy Mode",
      description: "Anonymize patient data in analytics",
      icon: Lock,
      type: "toggle",
      value: settings.privacy_mode,
    },
    {
      id: "auto_refresh",
      label: "Auto-Refresh Dashboard",
      description: "Automatically refresh analytics every 10 seconds",
      icon: Bell,
      type: "toggle",
      value: settings.auto_refresh,
    },
  ]

  const handleToggle = (id: string) => {
    setSettings((prev) => ({
      ...prev,
      [id]: !prev[id],
    }))
    setHasChanges(true)
  }

  const handleSelectChange = (id: string, value: string) => {
    setSettings((prev) => ({
      ...prev,
      [id]: value,
    }))
    setHasChanges(true)
  }

  const handleSave = () => {
    // Save to localStorage
    localStorage.setItem("medishield_settings", JSON.stringify(settings))
    toast.success("Settings saved successfully!")
    setHasChanges(false)
  }

  return (
    <div className="pt-20 px-6 min-h-screen">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold text-foreground mb-2">Settings</h1>
          <p className="text-muted-foreground">
            Customize your MediShield experience
          </p>
        </motion.div>

        {/* Settings Sections */}
        <div className="space-y-6">
          {/* Notifications Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card rounded-2xl p-6"
          >
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Bell className="h-5 w-5 text-primary" />
              Notifications & Alerts
            </h2>
            <div className="space-y-4">
              {settingsList.slice(0, 3).map((setting) => (
                <div
                  key={setting.id}
                  className="flex items-center justify-between p-4 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors"
                >
                  <div className="flex-1">
                    <p className="font-medium text-foreground">{setting.label}</p>
                    <p className="text-sm text-muted-foreground">
                      {setting.description}
                    </p>
                  </div>
                  <div className="ml-4">
                    {setting.type === "toggle" ? (
                      <button
                        onClick={() => handleToggle(setting.id)}
                        className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                          settings[setting.id]
                            ? "bg-primary"
                            : "bg-secondary border border-border"
                        }`}
                      >
                        <span
                          className={`inline-block h-6 w-6 transform rounded-full bg-background transition-transform ${
                            settings[setting.id]
                              ? "translate-x-7"
                              : "translate-x-1"
                          }`}
                        />
                      </button>
                    ) : (
                      <select
                        value={settings[setting.id]}
                        onChange={(e) =>
                          handleSelectChange(setting.id, e.target.value)
                        }
                        className="bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                      >
                        {setting.options?.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Appearance Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-card rounded-2xl p-6"
          >
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Sun className="h-5 w-5 text-primary" />
              Appearance
            </h2>
            <div className="flex items-center justify-between p-4 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors">
              <div className="flex-1">
                <p className="font-medium text-foreground">
                  {settingsList.find((s) => s.id === "theme")?.label}
                </p>
                <p className="text-sm text-muted-foreground">
                  {settingsList.find((s) => s.id === "theme")?.description}
                </p>
              </div>
              <div className="ml-4">
                <select
                  value={settings.theme}
                  onChange={(e) => handleSelectChange("theme", e.target.value)}
                  className="bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                >
                  <option value="dark">Dark</option>
                  <option value="light">Light</option>
                  <option value="auto">Auto</option>
                </select>
              </div>
            </div>
          </motion.div>

          {/* Data & Privacy Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-card rounded-2xl p-6"
          >
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Lock className="h-5 w-5 text-primary" />
              Data & Privacy
            </h2>
            <div className="space-y-4">
              {settingsList.slice(4).map((setting) => (
                <div
                  key={setting.id}
                  className="flex items-center justify-between p-4 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors"
                >
                  <div className="flex-1">
                    <p className="font-medium text-foreground">{setting.label}</p>
                    <p className="text-sm text-muted-foreground">
                      {setting.description}
                    </p>
                  </div>
                  <div className="ml-4">
                    {setting.type === "toggle" ? (
                      <button
                        onClick={() => handleToggle(setting.id)}
                        className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                          settings[setting.id]
                            ? "bg-primary"
                            : "bg-secondary border border-border"
                        }`}
                      >
                        <span
                          className={`inline-block h-6 w-6 transform rounded-full bg-background transition-transform ${
                            settings[setting.id]
                              ? "translate-x-7"
                              : "translate-x-1"
                          }`}
                        />
                      </button>
                    ) : (
                      <select
                        value={settings[setting.id]}
                        onChange={(e) =>
                          handleSelectChange(setting.id, e.target.value)
                        }
                        className="bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                      >
                        {setting.options?.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Account Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="glass-card rounded-2xl p-6"
          >
            <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Database className="h-5 w-5 text-primary" />
              Account
            </h2>
            <div className="space-y-3">
              <button className="w-full flex items-center justify-between p-4 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors text-left">
                <div>
                  <p className="font-medium text-foreground">
                    Change Password
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Update your account password
                  </p>
                </div>
                <ChevronRight className="h-5 w-5 text-muted-foreground" />
              </button>
              <button className="w-full flex items-center justify-between p-4 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors text-left">
                <div>
                  <p className="font-medium text-foreground">
                    Download Your Data
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Export your analytics and settings
                  </p>
                </div>
                <ChevronRight className="h-5 w-5 text-muted-foreground" />
              </button>
              <button className="w-full flex items-center justify-between p-4 rounded-xl bg-destructive/10 hover:bg-destructive/20 transition-colors text-left">
                <div>
                  <p className="font-medium text-destructive">
                    Delete Account
                  </p>
                  <p className="text-sm text-destructive/70">
                    Permanently delete your account and data
                  </p>
                </div>
                <ChevronRight className="h-5 w-5 text-destructive" />
              </button>
            </div>
          </motion.div>
        </div>

        {/* Save Button */}
        {hasChanges && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="fixed bottom-6 right-6"
          >
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-all shadow-lg hover:shadow-xl glow-cyan"
            >
              <Save className="h-4 w-4" />
              Save Settings
            </button>
          </motion.div>
        )}

        {/* Padding for button */}
        <div className="pb-20" />
      </div>
    </div>
  )
}
