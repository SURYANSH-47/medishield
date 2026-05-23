/**
 * Language Context & Hook
 * =======================
 * 
 * Manages multilingual state for clinical support recommendations
 * Persists language selection in localStorage
 */

"use client"

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
  const [language, setLanguageState] = useState<Language>("en")
  const [mounted, setMounted] = useState(false)

  // Load language from localStorage on mount
  useEffect(() => {
    const savedLanguage = localStorage.getItem("medishield-language") as Language | null
    if (savedLanguage && ["en", "hi", "kn"].includes(savedLanguage)) {
      setLanguageState(savedLanguage)
    }
    setMounted(true)
  }, [])

  const setLanguage = (lang: Language) => {
    setLanguageState(lang)
    localStorage.setItem("medishield-language", lang)
  }

  // Prevent hydration mismatch
  if (!mounted) {
    return <>{children}</>
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage }}>
      {children}
    </LanguageContext.Provider>
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
