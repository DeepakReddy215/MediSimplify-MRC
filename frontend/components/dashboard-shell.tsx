"use client";

import Link from "next/link";
import type { Route } from "next";
import dynamic from "next/dynamic";
import { usePathname, useRouter } from "next/navigation";
import { LogOut, BarChart3, LayoutDashboard, UploadCloud, Sparkles, History } from "lucide-react";
import { useMemo, type ReactNode } from "react";
import { useAuth } from "@/lib/auth";
import StarBorder from "./StarBorder";

const Silk = dynamic(() => import("@/components/ui/silk"), { ssr: false });

export default function DashboardShell({
  title,
  subtitle,
  children,
  activeTab = "overview"
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  activeTab?: "overview" | "upload" | "result" | "history";
}) {
  const router = useRouter();
  const pathname = usePathname();
  const { user, logout } = useAuth();

  const navItems = useMemo(
    () => [
      {
        id: "overview",
        label: "Overview",
        href: "/dashboard" as Route,
        icon: LayoutDashboard,
        iconClass: "text-blue-400"
      },
      {
        id: "upload",
        label: "Upload & OCR",
        href: "/dashboard/upload" as Route,
        icon: UploadCloud,
        iconClass: "text-indigo-400"
      },
      {
        id: "result",
        label: "Simplified Result",
        href: "/dashboard/result" as Route,
        icon: Sparkles,
        iconClass: "text-cyan-400"
      },
      {
        id: "history",
        label: "Report History",
        href: "/dashboard/history" as Route,
        icon: History,
        iconClass: "text-emerald-400"
      }
    ],
    []
  );

  const handleLogout = () => {
    logout();
    router.push("/");
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950">
      <div className="pointer-events-none fixed inset-0 z-0">
        <div className="absolute inset-0 opacity-40">
          <Silk speed={8.7} scale={1.2} color="#7B7481" noiseIntensity={1.2} rotation={0} />
        </div>
        <div className="absolute inset-0 bg-gradient-to-b from-slate-950/70 via-slate-950/50 to-slate-950/70" />
      </div>

      <header className="relative z-20 border-b border-slate-800/50 bg-slate-950/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <Link href="/dashboard" className="group flex items-center gap-2 transition">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-600">
              <BarChart3 size={18} className="text-white" />
            </div>
            <span className="text-lg font-bold tracking-tight">
              Medi<span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">Simplify</span>
            </span>
          </Link>

          <div className="flex items-center gap-6">
            <div className="hidden items-center gap-2 rounded-lg bg-slate-900/50 px-4 py-2 md:flex">
              <div className="h-2 w-2 rounded-full bg-emerald-500" />
              <p className="text-sm font-medium text-slate-200">{user?.full_name}</p>
            </div>
            <button
              onClick={handleLogout}
              className="group flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/50 px-4 py-2 text-sm font-medium text-slate-300 transition hover:border-slate-600 hover:bg-slate-800 hover:text-white"
            >
              <LogOut size={16} className="transition group-hover:scale-110" />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-7xl px-6 py-12">
        <div className="mb-8 space-y-3">
          <StarBorder as="h1" color="#3b82f6" speed="8s" className="text-4xl font-bold tracking-tight">
            {title}
          </StarBorder>
          <p className="max-w-2xl text-slate-400">{subtitle}</p>
        </div>

        <div className="mb-6 flex gap-2 overflow-x-auto pb-1 lg:hidden">
          {navItems.map((item) => {
            const isActive = activeTab === item.id || pathname === item.href;
            return (
              <Link
                key={item.id}
                href={item.href}
                className={`rounded-lg border px-3 py-2 text-xs transition ${
                  isActive
                    ? "border-blue-500/60 bg-blue-500/15 text-blue-200"
                    : "border-slate-700 bg-slate-900/60 text-slate-200"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>

        <div className="grid gap-6 lg:grid-cols-[260px_minmax(0,1fr)] lg:items-start">
          <aside className="hidden lg:block">
            <div className="sticky top-24 rounded-2xl border border-slate-800 bg-slate-900/60 p-4 backdrop-blur">
              <p className="mb-3 px-2 text-xs font-semibold uppercase tracking-wider text-slate-400">Workspace</p>
              <nav className="space-y-1">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = activeTab === item.id || pathname === item.href;

                  return (
                    <Link
                      key={item.id}
                      href={item.href}
                      className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition ${
                        isActive
                          ? "bg-blue-500/15 text-blue-100 ring-1 ring-blue-500/40"
                          : "text-slate-200 hover:bg-slate-800/80"
                      }`}
                    >
                      <Icon size={16} className={item.iconClass} />
                      {item.label}
                    </Link>
                  );
                })}
              </nav>

              <div className="mt-4 rounded-xl border border-slate-700 bg-slate-900/60 p-3">
                <p className="text-xs font-medium text-slate-300">Startup flow</p>
                <p className="mt-1 text-xs text-slate-400">
                  Upload file → Extract text → Simplify with AI → Save to timeline.
                </p>
              </div>
            </div>
          </aside>

          <section>{children}</section>
        </div>
      </main>
    </div>
  );
}
