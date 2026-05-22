import { DashboardSidebar, DashboardHeader } from "@/components/dashboard-layout"
import { PredictionForm } from "@/components/prediction-form"

export default function PredictionPage() {
  return (
    <div className="min-h-screen">
      <DashboardSidebar />
      <DashboardHeader />
      <main className="lg:ml-64 pt-16">
        <div className="p-6">
          <PredictionForm />
        </div>
      </main>
    </div>
  )
}
