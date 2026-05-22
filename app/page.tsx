import { Navbar } from "@/components/navbar"
import { HeroSection } from "@/components/hero-section"
import { FeaturesSection } from "@/components/features-section"
import { ArchitectureSection } from "@/components/architecture-section"
import { BenefitsSection } from "@/components/benefits-section"
import { Footer } from "@/components/footer"

export default function Home() {
  return (
    <main className="min-h-screen">
      <Navbar />
      <HeroSection />
      <FeaturesSection />
      <ArchitectureSection />
      <BenefitsSection />
      <Footer />
    </main>
  )
}
