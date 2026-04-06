import Link from "next/link";
import { ArrowRight, FileScan, ShieldCheck, Sparkles, Stethoscope } from "lucide-react";
import MarketingNav from "@/components/marketing-nav";
import FeatureCard from "@/components/feature-card";
import Footer from "@/components/footer";
import StarBorder from "@/components/StarBorder";
import LaserFlow from "@/components/LaserFlow";

const features = [
  {
    icon: FileScan,
    title: "OCR-powered extraction",
    description: "Upload reports or prescriptions as image/PDF and extract medical text in seconds."
  },
  {
    icon: Sparkles,
    title: "Easy-language simplification",
    description: "Convert difficult terms into clear explanations a normal user can understand."
  },
  {
    icon: ShieldCheck,
    title: "Private report history",
    description: "Securely access your previous uploads and simplifications from one dashboard."
  }
];

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-hero-grid">
      <MarketingNav />

      <section className="mx-auto grid max-w-7xl items-center gap-10 px-6 py-24 md:grid-cols-2">
        <div>
          <span className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/70 px-4 py-2 text-xs text-slate-300">
            <Stethoscope size={14} /> Healthcare clarity for everyone
          </span>
          <StarBorder
            as="h1"
            color="#3b82f6"
            speed="8s"
            className="mt-6 text-3xl font-bold tracking-tight sm:text-4xl"
          >
            Medical Report <span className="gradient-text">Simplifier</span>
          </StarBorder>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-slate-300">
            Understand reports without medical jargon. Upload any report, extract text, and get
            patient-friendly explanations instantly.
          </p>
          <div className="mt-10 flex flex-wrap items-center gap-4">
            <Link
              href="/signup"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-500 px-6 py-3 font-medium shadow-soft transition hover:bg-blue-400"
            >
              Get Started <ArrowRight size={16} />
            </Link>
            <Link
              href="/login"
              className="rounded-xl border border-slate-700 px-6 py-3 text-slate-200 transition hover:border-slate-500"
            >
              Try Demo
            </Link>
          </div>
        </div>

        <div className="relative">
          <div className="pointer-events-none absolute -top-40 left-1/2 z-0 h-[360px] w-[440px] -translate-x-1/2 opacity-90">
            <LaserFlow
              color="#3b82f6"
              horizontalBeamOffset={0}
              verticalBeamOffset={0}
              horizontalSizing={0.42}
              verticalSizing={6.6}
              wispDensity={1.35}
              wispSpeed={12.5}
              wispIntensity={5.6}
              flowSpeed={0.35}
              flowStrength={0.25}
              fogIntensity={0.5}
              fogScale={0.3}
              fogFallSpeed={0.6}
              decay={1.1}
              falloffStart={1.2}
            />
          </div>
          <div className="pointer-events-none absolute -top-3 left-1/2 z-20 h-8 w-56 -translate-x-1/2 rounded-full bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.55)_0%,rgba(59,130,246,0.2)_45%,rgba(59,130,246,0)_75%)] blur-sm" />

          <StarBorder
            as="div"
            id="how"
            className="relative z-10 w-full"
            color="#3b82f6"
            speed="8s"
            innerClassName="glass-card !rounded-2xl !border-slate-700/60 !bg-slate-900/70 !p-8 !text-left"
          >
            <h2 className="text-xl font-semibold">How it works</h2>
            <ol className="mt-6 space-y-5 text-slate-300">
              <li><strong className="text-white">1.</strong> Upload an image or PDF report.</li>
              <li><strong className="text-white">2.</strong> OCR reads and extracts medical text.</li>
              <li><strong className="text-white">3.</strong> AI simplifies text into plain language.</li>
              <li><strong className="text-white">4.</strong> Save and review from dashboard anytime.</li>
            </ol>
          </StarBorder>
        </div>
      </section>

      <section id="features" className="mx-auto max-w-7xl px-6 py-10">
        <StarBorder
          as="h2"
          color="#06b6d4"
          speed="7s"
          className="mx-auto block w-fit text-3xl font-semibold"
        >
          Built for patients, students, and families
        </StarBorder>
        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {features.map((feature) => (
            <FeatureCard key={feature.title} {...feature} />
          ))}
        </div>
      </section>

      <section id="benefits" className="mx-auto max-w-7xl px-6 py-20">
        <StarBorder
          as="div"
          className="w-full"
          color="#06b6d4"
          speed="7s"
          innerClassName="glass-card !grid !gap-8 !rounded-2xl !p-10 !text-left md:!grid-cols-3"
        >
          <div>
            <p className="text-3xl font-bold gradient-text">95%</p>
            <p className="mt-2 text-sm text-slate-300">Reports become easier to read with simplification.</p>
          </div>
          <div>
            <p className="text-3xl font-bold gradient-text">1-click</p>
            <p className="mt-2 text-sm text-slate-300">Upload flow for image or PDF with clean UX.</p>
          </div>
          <div>
            <p className="text-3xl font-bold gradient-text">24/7</p>
            <p className="mt-2 text-sm text-slate-300">Access your report history whenever you need it.</p>
          </div>
        </StarBorder>
      </section>

      <Footer />
    </main>
  );
}
