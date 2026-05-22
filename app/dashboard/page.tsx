import { DashboardSidebar, DashboardHeader } from "@/components/dashboard-layout"
import { FederatedDashboard } from "@/components/federated-dashboard"

export default function DashboardPage() {
  return (
    <div className="min-h-screen">
      <DashboardSidebar />
      <DashboardHeader />
      <main className="lg:ml-64 pt-16">
        <div className="p-6">
          <FederatedDashboard />
        </div>
      </main>
    </div>
  )
}
