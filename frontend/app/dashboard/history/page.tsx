"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import DashboardShell from "@/components/dashboard-shell";
import ReportHistory from "@/components/report-history";
import { getReports } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { Report } from "@/lib/types";

export default function DashboardHistoryPage() {
  const router = useRouter();
  const { token, ready } = useAuth();
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!ready) return;
    if (!token) {
      router.push("/login");
      return;
    }

    const loadReports = async () => {
      setLoading(true);
      try {
        const data = await getReports(token);
        setReports(data);
      } finally {
        setLoading(false);
      }
    };

    void loadReports();
  }, [ready, token, router]);

  return (
    <DashboardShell
      activeTab="history"
      title="Report History"
      subtitle="Browse your saved reports and open any entry for full detail."
    >
      {loading ? (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-8 text-center text-sm text-slate-400">
          Loading report history...
        </div>
      ) : (
        <ReportHistory reports={reports} />
      )}
    </DashboardShell>
  );
}
