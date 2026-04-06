"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Upload, FilesIcon, Clock, ArrowRight, Sparkles } from "lucide-react";
import DashboardShell from "@/components/dashboard-shell";
import { getReports } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { Report } from "@/lib/types";

export default function DashboardPage() {
  const router = useRouter();
  const { token, user, ready } = useAuth();
  const [reports, setReports] = useState<Report[]>([]);

  useEffect(() => {
    if (!ready) return;
    if (!token) {
      router.push("/login");
      return;
    }

    const loadReports = async () => {
      try {
        const data = await getReports(token);
        setReports(data);
      } catch {
        setReports([]);
      }
    };

    void loadReports();
  }, [ready, token, router]);

  const stats = useMemo(
    () => [
      {
        label: "Total uploads",
        value: reports.length.toString(),
        icon: Upload,
        gradient: "from-blue-500 to-blue-600"
      },
      {
        label: "Latest upload",
        value: reports[0] ? new Date(reports[0].created_at).toLocaleDateString() : "-",
        icon: Clock,
        gradient: "from-purple-500 to-purple-600"
      },
      {
        label: "Saved reports",
        value: reports.filter((r) => !!r.id).length.toString(),
        icon: FilesIcon,
        gradient: "from-emerald-500 to-emerald-600"
      }
    ],
    [reports]
  );

  if (!ready || (!token && !user)) {
    return <div className="grid min-h-screen place-items-center text-slate-400">Loading...</div>;
  }

  return (
    <DashboardShell
      activeTab="overview"
      title={`Welcome, ${user?.full_name?.split(" ")[0] || "User"}`}
      subtitle="Manage your medical reports with AI-powered simplification and secure history tracking."
    >
      <section className="mb-8 grid gap-5 md:grid-cols-3">
        {stats.map((card) => {
          const Icon = card.icon;
          return (
            <div
              key={card.label}
              className="group relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur transition hover:border-slate-700 hover:bg-slate-900/70"
            >
              <div className={`absolute inset-0 opacity-0 bg-gradient-to-br ${card.gradient} transition group-hover:opacity-5`} />
              <div className="relative space-y-3">
                <div className={`inline-flex rounded-lg bg-gradient-to-br ${card.gradient} p-3`}>
                  <Icon size={20} className="text-white" />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-400">{card.label}</p>
                  <p className="mt-2 text-3xl font-bold text-white">{card.value}</p>
                </div>
              </div>
            </div>
          );
        })}
      </section>

      <section className="grid gap-5 md:grid-cols-3">
        <Link
          href="/dashboard/upload"
          className="group rounded-2xl border border-slate-800 bg-slate-900/50 p-6 transition hover:border-indigo-500/60 hover:bg-slate-900/70"
        >
          <div className="mb-4 inline-flex rounded-lg bg-indigo-500/20 p-3">
            <Upload size={20} className="text-indigo-300" />
          </div>
          <h3 className="text-lg font-semibold text-white">Upload & OCR</h3>
          <p className="mt-2 text-sm text-slate-400">Upload report files and run OCR extraction in a dedicated workspace.</p>
          <p className="mt-4 inline-flex items-center gap-1 text-sm text-indigo-300">
            Open module <ArrowRight size={14} className="transition group-hover:translate-x-1" />
          </p>
        </Link>

        <Link
          href="/dashboard/result"
          className="group rounded-2xl border border-slate-800 bg-slate-900/50 p-6 transition hover:border-cyan-500/60 hover:bg-slate-900/70"
        >
          <div className="mb-4 inline-flex rounded-lg bg-cyan-500/20 p-3">
            <Sparkles size={20} className="text-cyan-300" />
          </div>
          <h3 className="text-lg font-semibold text-white">Simplified Results</h3>
          <p className="mt-2 text-sm text-slate-400">Review your latest simplified output and key medical terms on a focused page.</p>
          <p className="mt-4 inline-flex items-center gap-1 text-sm text-cyan-300">
            Open module <ArrowRight size={14} className="transition group-hover:translate-x-1" />
          </p>
        </Link>

        <Link
          href="/dashboard/history"
          className="group rounded-2xl border border-slate-800 bg-slate-900/50 p-6 transition hover:border-emerald-500/60 hover:bg-slate-900/70"
        >
          <div className="mb-4 inline-flex rounded-lg bg-emerald-500/20 p-3">
            <FilesIcon size={20} className="text-emerald-300" />
          </div>
          <h3 className="text-lg font-semibold text-white">Report History</h3>
          <p className="mt-2 text-sm text-slate-400">Browse saved reports and open individual records from your timeline.</p>
          <p className="mt-4 inline-flex items-center gap-1 text-sm text-emerald-300">
            Open module <ArrowRight size={14} className="transition group-hover:translate-x-1" />
          </p>
        </Link>
      </section>

      <section className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
        <h3 className="text-lg font-semibold text-white">How to use</h3>
        <p className="mt-2 text-sm text-slate-400">
          Start with <span className="text-indigo-300">Upload & OCR</span>, then move to
          <span className="text-cyan-300"> Simplified Results</span>, and manage everything in
          <span className="text-emerald-300"> Report History</span>.
        </p>
      </section>
    </DashboardShell>
  );
}
