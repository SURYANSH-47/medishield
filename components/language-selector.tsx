/**
 * Language Selector Component
 * ============================
 * 
 * Button group for selecting clinical recommendation language
 * Supports: English, Hindi, Kannada
 * 
 * Features:
 * - Smooth transitions on language change
 * - Persistent selection via localStorage
 * - Minimalist design matching MediShield UI
 */

"use client"

import { motion } from "framer-motion"
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
      className="flex items-center gap-2 flex-wrap"
    >
      <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
        Recommendation Language:
      </span>
      <div className="flex gap-2 flex-wrap">
        {AVAILABLE_LANGUAGES.map((lang) => (
          <motion.button
            key={lang.code}
            onClick={() => setLanguage(lang.code as Language)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
            className={`px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 ${
              language === lang.code
                ? "bg-primary/40 text-primary border border-primary/60 shadow-[0_0_16px_rgba(99,102,241,0.2)]"
                : "bg-white/5 text-muted-foreground border border-white/10 hover:bg-white/10 hover:text-foreground"
            }`}
          >
            <span className="hidden sm:inline">{lang.name}</span>
            <span className="sm:hidden">{lang.nativeName}</span>
          </motion.button>
        ))}
      </div>
    </motion.div>
  )
}
