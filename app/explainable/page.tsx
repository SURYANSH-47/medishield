import { DashboardSidebar, DashboardHeader } from "@/components/dashboard-layout"
import { ExplainableAIDashboard } from "@/components/explainable-ai"

export default function ExplainablePage() {
  return (
    <div className="min-h-screen">
      <DashboardSidebar />
      <DashboardHeader />
      <main className="lg:ml-64 pt-16">
        <div className="p-6">
          <ExplainableAIDashboard />
        </div>
      </main>
    </div>
  )
}
