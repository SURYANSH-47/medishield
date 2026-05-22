import { DashboardSidebar, DashboardHeader } from "@/components/dashboard-layout"
import { HealthcareAnalytics } from "@/components/healthcare-analytics"

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen">
      <DashboardSidebar />
      <DashboardHeader />
      <main className="lg:ml-64 pt-16">
        <div className="p-6">
          <HealthcareAnalytics />
        </div>
      </main>
    </div>
  )
}
