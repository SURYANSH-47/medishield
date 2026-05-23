/**
 * Multilingual Healthcare Support Translations
 * =============================================
 * 
 * Translation dictionaries for Clinical Support recommendations
 * Supported languages: English, Hindi, Kannada
 * 
 * Designed for frontline healthcare workers in regional healthcare environments
 */

export type Language = "en" | "hi" | "kn"

export interface TranslationDict {
  // Risk levels
  riskLevelLow: string
  riskLevelMedium: string
  riskLevelHigh: string
  riskLevelCritical: string

  // Section titles
  clinicalSummaryTitle: string
  recommendedActionsTitle: string
  monitoringChecklistTitle: string
  escalationWarningTitle: string
  suggestedNextStepsTitle: string
  primaryRiskContributorsTitle: string
  emergencyPriorityTitle: string
  aiClinicalsummaryTitle: string

  // Clinical Recommendations
  immediatePhysicianReview: string
  monitorBloodGlucoseImmediately: string
  observeBloodPressureClosely: string
  continuousMonitoringAdvised: string
  emergencyEscalationRequired: string
  prepareDiabeticManagementWorkflow: string
  watchForDehydrationSymptoms: string
  monitorOxygenSaturation: string
  schedulePhysicianConsultation: string

  // Monitoring Checklist
  checkGlucoseEvery30Minutes: string
  monitorBloodPressure: string
  observePatientVitals: string
  watchForConfusionDizziness: string

  // Escalation Warnings
  chestPain: string
  severeDizziness: string
  oxygenDrop: string
  lossOfConsciousness: string

  // Emergency Priority Levels
  routinePriority: string
  elevatedPriority: string
  urgentPriority: string
  emergencyPriority: string

  // Priority level descriptions
  routineDescription: string
  elevatedDescription: string
  urgentDescription: string
  emergencyDescription: string

  // UI Labels
  primaryRiskDrivers: string
  clinicalSummaryLabel: string
  suggestedNextStepsLabel: string
  emergencyPriorityLabel: string
  federated: string

  // Next steps (common ones)
  nextStepImmediate: string
  nextStepPhysician: string
  nextStepMonitoring: string
  nextStepRest: string
  nextStepDiet: string
  nextStepExercise: string
}

// ============================================================================
// English Translations
// ============================================================================
const EN: TranslationDict = {
  // Risk levels
  riskLevelLow: "Low Risk",
  riskLevelMedium: "Medium Risk",
  riskLevelHigh: "High Risk",
  riskLevelCritical: "Critical Risk",

  // Section titles
  clinicalSummaryTitle: "AI Clinical Summary",
  recommendedActionsTitle: "Recommended Clinical Actions",
  monitoringChecklistTitle: "Monitoring Checklist",
  escalationWarningTitle: "Escalation Warning Criteria",
  suggestedNextStepsTitle: "Suggested Next Steps",
  primaryRiskContributorsTitle: "Primary Risk Contributors",
  emergencyPriorityTitle: "Emergency Priority Level",
  aiClinicalsummaryTitle: "AI Clinical Summary",

  // Clinical Recommendations
  immediatePhysicianReview: "Immediate physician review recommended",
  monitorBloodGlucoseImmediately: "Monitor blood glucose immediately",
  observeBloodPressureClosely: "Observe blood pressure closely",
  continuousMonitoringAdvised: "Continuous monitoring advised",
  emergencyEscalationRequired: "Emergency escalation required",
  prepareDiabeticManagementWorkflow: "Prepare diabetic management workflow",
  watchForDehydrationSymptoms: "Watch for dehydration symptoms",
  monitorOxygenSaturation: "Monitor oxygen saturation",
  schedulePhysicianConsultation: "Schedule physician consultation",

  // Monitoring Checklist
  checkGlucoseEvery30Minutes: "Check glucose every 30 minutes",
  monitorBloodPressure: "Monitor blood pressure",
  observePatientVitals: "Observe patient vitals",
  watchForConfusionDizziness: "Watch for confusion or dizziness",

  // Escalation Warnings
  chestPain: "Chest pain",
  severeDizziness: "Severe dizziness",
  oxygenDrop: "Oxygen drop",
  lossOfConsciousness: "Loss of consciousness",

  // Emergency Priority Levels
  routinePriority: "ROUTINE",
  elevatedPriority: "ELEVATED",
  urgentPriority: "URGENT",
  emergencyPriority: "EMERGENCY",

  // Priority level descriptions
  routineDescription: "Standard monitoring schedule applies",
  elevatedDescription: "Physician review advised within 48–72 hrs",
  urgentDescription: "Immediate physician evaluation required",
  emergencyDescription: "Activate rapid response team immediately",

  // UI Labels
  primaryRiskDrivers: "Primary Risk Drivers",
  clinicalSummaryLabel: "AI Clinical Summary",
  suggestedNextStepsLabel: "Suggested Next Steps",
  emergencyPriorityLabel: "Emergency Priority",
  federated: "Federated Learning Intelligence",

  // Next steps (common ones)
  nextStepImmediate: "Immediate action required",
  nextStepPhysician: "Consult with physician",
  nextStepMonitoring: "Continue close monitoring",
  nextStepRest: "Ensure adequate rest",
  nextStepDiet: "Review dietary habits",
  nextStepExercise: "Increase physical activity",
}

// ============================================================================
// Hindi Translations
// ============================================================================
const HI: TranslationDict = {
  // Risk levels
  riskLevelLow: "कम जोखिम",
  riskLevelMedium: "माध्यम जोखिम",
  riskLevelHigh: "उच्च जोखिम",
  riskLevelCritical: "गंभीर जोखिम",

  // Section titles
  clinicalSummaryTitle: "AI नैदानिक सारांश",
  recommendedActionsTitle: "अनुशंसित नैदानिक कार्रवाई",
  monitoringChecklistTitle: "निगरानी चेकलिस्ट",
  escalationWarningTitle: "संवर्धन चेतावनी मानदंड",
  suggestedNextStepsTitle: "सुझाए गए अगले कदम",
  primaryRiskContributorsTitle: "प्राथमिक जोखिम कारक",
  emergencyPriorityTitle: "आपातकालीन प्राथमिकता स्तर",
  aiClinicalsummaryTitle: "AI नैदानिक सारांश",

  // Clinical Recommendations
  immediatePhysicianReview: "तत्काल चिकित्सकीय समीक्षा की सलाह दी जाती है",
  monitorBloodGlucoseImmediately: "तुरंत रक्त ग्लूकोज की निगरानी करें",
  observeBloodPressureClosely: "रक्तचाप की बारीकी से निगरानी करें",
  continuousMonitoringAdvised: "निरंतर निगरानी की सलाह दी जाती है",
  emergencyEscalationRequired: "आपातकालीन संवर्धन आवश्यक है",
  prepareDiabeticManagementWorkflow: "मधुमेह प्रबंधन वर्कफ़्लो तैयार करें",
  watchForDehydrationSymptoms: "निर्जलीकरण के लक्षणों के लिए देखें",
  monitorOxygenSaturation: "ऑक्सीजन संतृप्ति की निगरानी करें",
  schedulePhysicianConsultation: "चिकित्सक परामर्श शेड्यूल करें",

  // Monitoring Checklist
  checkGlucoseEvery30Minutes: "हर 30 मिनट में ग्लूकोज की जांच करें",
  monitorBloodPressure: "रक्तचाप की निगरानी करें",
  observePatientVitals: "रोगी के महत्वपूर्ण संकेतों का अवलोकन करें",
  watchForConfusionDizziness: "भ्रम या चक्कर आने के लिए देखें",

  // Escalation Warnings
  chestPain: "सीने में दर्द",
  severeDizziness: "गंभीर चक्कर आना",
  oxygenDrop: "ऑक्सीजन में गिरावट",
  lossOfConsciousness: "चेतना की हानि",

  // Emergency Priority Levels
  routinePriority: "नियमित",
  elevatedPriority: "उन्नत",
  urgentPriority: "तुरंत",
  emergencyPriority: "आपातकाल",

  // Priority level descriptions
  routineDescription: "मानक निगरानी अनुसूची लागू होती है",
  elevatedDescription: "चिकित्सक समीक्षा 48-72 घंटों के भीतर सलाह दी जाती है",
  urgentDescription: "तत्काल चिकित्सक मूल्यांकन आवश्यक है",
  emergencyDescription: "तुरंत तेजी से प्रतिक्रिया दल को सक्रिय करें",

  // UI Labels
  primaryRiskDrivers: "प्राथमिक जोखिम कारक",
  clinicalSummaryLabel: "AI नैदानिक सारांश",
  suggestedNextStepsLabel: "सुझाए गए अगले कदम",
  emergencyPriorityLabel: "आपातकालीन प्राथमिकता",
  federated: "संघीय शिक्षा बुद्धिमत्ता",

  // Next steps (common ones)
  nextStepImmediate: "तत्काल कार्रवाई आवश्यक है",
  nextStepPhysician: "चिकित्सक से परामर्श करें",
  nextStepMonitoring: "निकट निगरानी जारी रखें",
  nextStepRest: "पर्याप्त आराम सुनिश्चित करें",
  nextStepDiet: "आहार की आदतों की समीक्षा करें",
  nextStepExercise: "शारीरिक गतिविधि बढ़ाएं",
}

// ============================================================================
// Kannada Translations
// ============================================================================
const KN: TranslationDict = {
  // Risk levels
  riskLevelLow: "ಕಡಿಮೆ ಅಪಾಯ",
  riskLevelMedium: "ಮಧ್ಯಮ ಅಪಾಯ",
  riskLevelHigh: "ಹೆಚ್ಚಿನ ಅಪಾಯ",
  riskLevelCritical: "ನಿರ್ಣಾಯಕ ಅಪಾಯ",

  // Section titles
  clinicalSummaryTitle: "AI ಕ್ಲಿನಿಕಲ್ ಸಾರಾಂಶ",
  recommendedActionsTitle: "ಶಿಫಾರಸು ಮಾಡಿದ ಕ್ಲಿನಿಕಲ್ ಕ್ರಿಯೆಗಳು",
  monitoringChecklistTitle: "ಮೇಲ್ವಿಚಾರಣೆ ಚೆಕ್‌ಲಿಸ್ಟ್",
  escalationWarningTitle: "ಏರಿಕೆ ಎಚ್ಚರಿಕೆ ಮಾನದಂಡ",
  suggestedNextStepsTitle: "ಸೂಚಿತ ಮುಂದಿನ ಹಂತಗಳು",
  primaryRiskContributorsTitle: "ಪ್ರಾಥಮಿಕ ಅಪಾಯ ಅಂಶಗಳು",
  emergencyPriorityTitle: "ತುರ್ತು ಆದ್ಯತೆ ಮಟ್ಟ",
  aiClinicalsummaryTitle: "AI ಕ್ಲಿನಿಕಲ್ ಸಾರಾಂಶ",

  // Clinical Recommendations
  immediatePhysicianReview: "ತಕ್ಷಣ ವೈದ್ಯಕೀಯ ಪರಿಶೀಲನೆ ಅಗತ್ಯವಿದೆ",
  monitorBloodGlucoseImmediately: "ತಕ್ಷಣ ರಕ್ತದ ಗ್ಲುಕೋಸ್ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ",
  observeBloodPressureClosely: "ರಕ್ತದ ಒತ್ತಡವನ್ನು ಎಚ್ಚೆತ್ತು ಗಮನಿಸಿ",
  continuousMonitoringAdvised: "ನಿರಂತರ ಮೇಲ್ವಿಚಾರಣೆ ಸೂಚಿಸಲಾಗುತ್ತದೆ",
  emergencyEscalationRequired: "ತುರ್ತು ಏರಿಕೆ ಅಗತ್ಯವಿದೆ",
  prepareDiabeticManagementWorkflow: "ಮಧುಮೇಹ ನಿರ್ವಹಣೆ ಕೆಲಸದ ಹರಿವು ಸಿದ್ಧಪಡಿಸಿ",
  watchForDehydrationSymptoms: "ನಿರ್ಜಲೀಕರಣ ರೋಗಲಕ್ಷಣಗಳನ್ನು ಗಮನಿಸಿ",
  monitorOxygenSaturation: "ಆಮ್ಲಜನಕ ನಿರ್ಬಂಧಕ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ",
  schedulePhysicianConsultation: "ವೈದ್ಯ ಸಮಾಲೋಚನೆ ನಿಗದಿಪಡಿಸಿ",

  // Monitoring Checklist
  checkGlucoseEvery30Minutes: "ಪ್ರತಿ 30 ನಿಮಿಷಕ್ಕೆ ಗ್ಲುಕೋಸ್ ಪರಿಶೀಲಿಸಿ",
  monitorBloodPressure: "ರಕ್ತದ ಒತ್ತಡವನ್ನು ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ",
  observePatientVitals: "ರೋಗಿಯ ಮೂಲ ಚಿಹ್ನೆಗಳನ್ನು ಗಮನಿಸಿ",
  watchForConfusionDizziness: "ಗೊಂದಲ ಅಥವಾ ತಲೆತಿರುಗುವಿಕೆಯನ್ನು ಗಮನಿಸಿ",

  // Escalation Warnings
  chestPain: "ಛಾತಿ ನೋವು",
  severeDizziness: "ತೀವ್ರ ತಲೆತಿರುಗುವಿಕೆ",
  oxygenDrop: "ಆಮ್ಲಜನಕ ಕುಸಿತ",
  lossOfConsciousness: "ಚೇತನ್ಯ ನಷ್ಟ",

  // Emergency Priority Levels
  routinePriority: "ನೈಯಮಿಕ",
  elevatedPriority: "ಎತ್ತರದ",
  urgentPriority: "ತುರ್ತು",
  emergencyPriority: "ತುರ್ತುಸ್ಥಿತಿ",

  // Priority level descriptions
  routineDescription: "ಪ್ರಮಾಣಿತ ಮೇಲ್ವಿಚಾರಣೆ ವೇಳಾಪಟ್ಟಿ ಅನ್ವಯಿಸುತ್ತದೆ",
  elevatedDescription: "ವೈದ್ಯ ಪರಿಶೀಲನೆ 48-72 ಗಂಟೆಗಳ ಬಳಿಕ ಸೂಚಿಸಲಾಗುತ್ತದೆ",
  urgentDescription: "ತಕ್ಷಣ ವೈದ್ಯ ಮೂಲ್ಯಮಾಪನ ಅಗತ್ಯವಿದೆ",
  emergencyDescription: "ತಕ್ಷಣ ವೇಗದ ಪ್ರತಿಕ್ರಿಯೆ ತಂಡವನ್ನು ಸಕ್ರಿಯ ಮಾಡಿ",

  // UI Labels
  primaryRiskDrivers: "ಪ್ರಾಥಮಿಕ ಅಪಾಯ ಚಾಲಕಗಳು",
  clinicalSummaryLabel: "AI ಕ್ಲಿನಿಕಲ್ ಸಾರಾಂಶ",
  suggestedNextStepsLabel: "ಸೂಚಿತ ಮುಂದಿನ ಹಂತಗಳು",
  emergencyPriorityLabel: "ತುರ್ತು ಆದ್ಯತೆ",
  federated: "ಸಾಂಘಿಕ ಕಲಿಕೆ ಬುದ್ಧಿಮತ್ತೆ",

  // Next steps (common ones)
  nextStepImmediate: "ತಕ್ಷಣ ಕ್ರಿಯೆ ಅಗತ್ಯವಿದೆ",
  nextStepPhysician: "ವೈದ್ಯನೊಂದಿಗೆ ಸಮಾಲೋಚನೆ ಮಾಡಿ",
  nextStepMonitoring: "ಹತ್ತಿರದ ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ",
  nextStepRest: "ಸಾಕಷ್ಟು ವಿಶ್ರಾಮ ಖಾತ್ರಿಪಡಿಸಿ",
  nextStepDiet: "ಆಹಾರ ಅಭ್ಯಾಸಗಳ ಪರಿಶೀಲನೆ ಮಾಡಿ",
  nextStepExercise: "ದೈಹಿಕ ಚಟುವಟಿಕೆ ಹೆಚ್ಚಿಸಿ",
}

// ============================================================================
// Master translations object
// ============================================================================

export const TRANSLATIONS: Record<Language, TranslationDict> = {
  en: EN,
  hi: HI,
  kn: KN,
}

/**
 * Get translation for a given language and key
 * @param language - Language code ('en', 'hi', 'kn')
 * @param key - Translation key
 * @returns Translated string or the key itself if not found
 */
export function getTranslation(language: Language, key: keyof TranslationDict): string {
  return TRANSLATIONS[language]?.[key] ?? key
}

/**
 * Helper to get all available languages
 */
export const AVAILABLE_LANGUAGES: Array<{ code: Language; name: string; nativeName: string }> = [
  { code: "en", name: "English", nativeName: "English" },
  { code: "hi", name: "Hindi", nativeName: "हिन्दी" },
  { code: "kn", name: "Kannada", nativeName: "ಕನ್ನಡ" },
]
