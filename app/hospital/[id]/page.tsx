import { DashboardSidebar, DashboardHeader } from "@/components/dashboard-layout"
import { HospitalPage } from "@/components/hospital-page"
import { notFound } from "next/navigation"

const VALID_IDS = ["a", "b", "c"] as const
type HospitalId = (typeof VALID_IDS)[number]

export default async function HospitalDetailPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params

  if (!VALID_IDS.includes(id as HospitalId)) {
    notFound()
  }

  return (
    <div className="min-h-screen">
      <DashboardSidebar />
      <DashboardHeader />
      <main className="lg:ml-64 pt-16">
        <div className="p-6">
          <HospitalPage hospitalId={id as HospitalId} />
        </div>
      </main>
    </div>
  )
}
