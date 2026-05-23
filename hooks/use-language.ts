"use client"

/**
 * Language Context & Hook
 * =======================
 *
 * Manages multilingual state for clinical support recommendations.
 * Persists language selection in localStorage across sessions.
 * Defaults to "en" on first render to avoid hydration mismatch.
 *
 * NOTE: Written with React.createElement (no JSX) so this file can
 * stay as .ts rather than .tsx.
 */

import React, { createContext, useContext, useState, useEffect } from "react"
import type { Language } from "@/lib/translations"

// ============================================================================
// Types
// ============================================================================

interface LanguageContextType {
  language: Language
  setLanguage: (lang: Language) => void
}

// ============================================================================
// Context
// ============================================================================

const LanguageContext = createContext<LanguageContextType | undefined>(undefined)

// ============================================================================
// Provider Component
// ============================================================================

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  // Default to "en" — safe for SSR, no hydration mismatch
  const [language, setLanguageState] = useState<Language>("en")

  // Restore persisted language on the client after first paint
  useEffect(() => {
    const saved = localStorage.getItem("medishield-language") as Language | null
    if (saved && (["en", "hi", "kn"] as Language[]).includes(saved)) {
      setLanguageState(saved)
    }
  }, [])

  const setLanguage = (lang: Language) => {
    setLanguageState(lang)
    localStorage.setItem("medishield-language", lang)
  }

  // Use React.createElement so this file remains .ts (no JSX extension needed)
  return React.createElement(
    LanguageContext.Provider,
    { value: { language, setLanguage } },
    children
  )
}

// ============================================================================
// Hook
// ============================================================================

export function useLanguage() {
  const context = useContext(LanguageContext)
  if (!context) {
    throw new Error("useLanguage must be used within LanguageProvider")
  }
  return context
}
