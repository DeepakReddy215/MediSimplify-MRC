export default function Footer() {
  return (
    <footer className="border-t border-slate-800 py-10">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-6 text-sm text-slate-400 md:flex-row md:items-center md:justify-between">
        <p>© {new Date().getFullYear()} MediSimplify. Built for patient readability.</p>
        <p>Medical guidance only. Not a replacement for doctors.</p>
      </div>
    </footer>
  );
}
