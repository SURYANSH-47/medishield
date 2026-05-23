/**
 * Language Selector Component
 * ============================
 *
 * Button group for selecting clinical recommendation language.
 * Supports: English, Hindi (हिन्दी), Kannada (ಕನ್ನಡ)
 *
 * Positioned above the Clinical Support Panel — does NOT translate the
 * dashboard, charts, or admin UI; only the prediction + clinical output.
 */

"use client"

import { motion } from "framer-motion"
import { Globe } from "lucide-react"
import { useLanguage } from "@/hooks/use-language"
import type { Language } from "@/lib/translations"
import { AVAILABLE_LANGUAGES } from "@/lib/translations"

// ============================================================================
// Component
// ============================================================================

export function LanguageSelector() {
  const { language, setLanguage } = useLanguage()

  return (
    <motion.div
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="flex items-center gap-3 flex-wrap"
    >
      {/* Label */}
      <div className="flex items-center gap-1.5">
        <Globe className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider whitespace-nowrap">
          Clinical Language:
        </span>
      </div>

      {/* Buttons — always show native script so nurses can identify their language */}
      <div className="flex gap-2 flex-wrap">
        {AVAILABLE_LANGUAGES.map((lang) => {
          const isActive = language === lang.code
          return (
            <motion.button
              key={lang.code}
              onClick={() => setLanguage(lang.code as Language)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
              aria-label={`Switch to ${lang.name}`}
              aria-pressed={isActive}
              className={`px-4 py-1.5 rounded-lg font-medium text-sm transition-all duration-200 ${
                isActive
                  ? "bg-primary/30 text-primary border border-primary/60 shadow-[0_0_16px_rgba(99,102,241,0.25)]"
                  : "bg-white/5 text-muted-foreground border border-white/10 hover:bg-white/10 hover:text-foreground"
              }`}
            >
              {lang.nativeName}
            </motion.button>
          )
        })}
      </div>
    </motion.div>
  )
}
