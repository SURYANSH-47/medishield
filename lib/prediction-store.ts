/**
 * prediction-store.ts
 * ===================
 * Lightweight localStorage-based store for sharing prediction results
 * between the Prediction page and the Explainable AI page.
 *
 * No external state library required — uses localStorage + React hooks.
 * The store persists across page navigations and browser refreshes.
 *
 * Usage
 * -----
 *   // In prediction-form.tsx (after a successful API call):
 *   import { savePrediction } from "@/lib/prediction-store"
 *   savePrediction({ inputs, response, timestamp: Date.now() })
 *
 *   // In explainable-ai.tsx (to read the last prediction):
 *   import { usePredictionStore } from "@/lib/prediction-store"
 *   const { snapshot, secondsSince, isEmpty } = usePredictionStore()
 */

import { useState, useEffect, useCallback } from "react"
import type { PredictionRequest, PredictionResponse } from "./api"

// ---------------------------------------------------------------------------
// Storage key
// ---------------------------------------------------------------------------

const STORAGE_KEY = "medishield_last_prediction"

// Custom DOM event name for same-tab broadcasts
const UPDATE_EVENT = "medishield-prediction-updated"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Everything persisted after a successful prediction */
export interface PredictionSnapshot {
  /** Patient input values sent to the API */
  inputs: PredictionRequest
  /** Full API response (risk_score, shap_values, etc.) */
  response: PredictionResponse
  /** Unix timestamp (ms) when the prediction was made */
  timestamp: number
}

// ---------------------------------------------------------------------------
// Persist / load helpers (safe for SSR — guard with typeof window check)
// ---------------------------------------------------------------------------

/**
 * Persist a prediction snapshot to localStorage and notify any open
 * components that data has changed.
 */
export function savePrediction(snapshot: PredictionSnapshot): void {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot))
    // Broadcast to all listeners in the same tab immediately
    window.dispatchEvent(new CustomEvent(UPDATE_EVENT))
  } catch {
    // localStorage unavailable (e.g. private mode, quota exceeded) — ignore
  }
}

/**
 * Read the most recent prediction snapshot from localStorage.
 * Returns null if no prediction has been saved yet.
 */
export function loadPrediction(): PredictionSnapshot | null {
  if (typeof window === "undefined") return null
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    return JSON.parse(raw) as PredictionSnapshot
  } catch {
    return null
  }
}

/**
 * Remove the stored prediction from localStorage.
 */
export function clearPrediction(): void {
  if (typeof window === "undefined") return
  try {
    localStorage.removeItem(STORAGE_KEY)
    window.dispatchEvent(new CustomEvent(UPDATE_EVENT))
  } catch {}
}

// ---------------------------------------------------------------------------
// React hook
// ---------------------------------------------------------------------------

export interface UsePredictionStore {
  /** The most recent stored prediction, or null if none exists */
  snapshot: PredictionSnapshot | null
  /**
   * Seconds elapsed since the stored prediction was made.
   * Updates every second while a snapshot is present.
   * null if no snapshot.
   */
  secondsSince: number | null
  /** True when no prediction has been stored yet */
  isEmpty: boolean
}

/**
 * React hook that provides reactive access to the prediction store.
 *
 * Automatically re-renders when:
 *   • A new prediction is saved in the same tab
 *   • Another tab saves a prediction (via the storage event)
 *   • The elapsed time ticks over a new second
 */
export function usePredictionStore(): UsePredictionStore {
  const [snapshot, setSnapshot] = useState<PredictionSnapshot | null>(null)
  const [now, setNow] = useState<number>(Date.now())

  // Reload from storage (called on mount and on any update event)
  const refresh = useCallback(() => {
    setSnapshot(loadPrediction())
  }, [])

  // Hydrate from localStorage on mount (client-side only)
  useEffect(() => {
    refresh()
  }, [refresh])

  // Listen for updates from prediction-form in the same tab,
  // and from other tabs via the native storage event
  useEffect(() => {
    const handle = () => refresh()
    window.addEventListener(UPDATE_EVENT, handle)
    window.addEventListener("storage", handle)
    return () => {
      window.removeEventListener(UPDATE_EVENT, handle)
      window.removeEventListener("storage", handle)
    }
  }, [refresh])

  // Tick every second so "X seconds ago" label stays current
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1_000)
    return () => clearInterval(id)
  }, [])

  const secondsSince = snapshot
    ? Math.floor((now - snapshot.timestamp) / 1_000)
    : null

  return {
    snapshot,
    secondsSince,
    isEmpty: snapshot === null,
  }
}
