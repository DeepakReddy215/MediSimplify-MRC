import Link from "next/link";

export default function MarketingNav() {
  return (
    <header className="sticky top-0 z-30 border-b border-slate-800/60 bg-background/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <Link href="/" className="text-lg font-semibold tracking-tight">
          Medi<span className="gradient-text">Simplify</span>
        </Link>
        <nav className="hidden gap-8 text-sm text-slate-300 md:flex">
          <a href="#features" className="transition hover:text-white">Features</a>
          <a href="#how" className="transition hover:text-white">How it works</a>
          <a href="#benefits" className="transition hover:text-white">Benefits</a>
        </nav>
        <div className="flex items-center gap-3">
          <Link
            href="/login"
            className="rounded-xl border border-slate-700 px-4 py-2 text-sm transition hover:border-slate-500"
          >
            Login
          </Link>
          <Link
            href="/signup"
            className="rounded-xl bg-blue-500 px-4 py-2 text-sm font-medium text-white shadow-soft transition hover:bg-blue-400"
          >
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}
