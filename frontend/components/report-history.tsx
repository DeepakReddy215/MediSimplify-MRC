import Link from "next/link";
import { Calendar, FileText, ArrowRight } from "lucide-react";
import StarBorder from "./StarBorder";
import type { Report } from "@/lib/types";

export default function ReportHistory({ reports }: { reports: Report[] }) {
  if (!reports.length) {
    return (
      <StarBorder
        as="div"
        className="w-full"
        color="#06b6d4"
        speed="7s"
        innerClassName="!rounded-2xl !border-slate-700/70 !bg-gradient-to-br !from-slate-900/50 !to-slate-900/30 !p-8 !text-left !backdrop-blur"
      >
        <div className="text-center">
          <div className="mb-4 flex justify-center">
            <div className="rounded-lg bg-slate-800/50 p-3">
              <FileText className="text-slate-500" size={32} />
            </div>
          </div>
          <StarBorder as="h3" color="#06b6d4" speed="7s" className="text-lg font-semibold">
            Report history
          </StarBorder>
          <p className="mt-3 text-sm text-slate-400">No reports yet. Upload your first file to get started.</p>
        </div>
      </StarBorder>
    );
  }

  return (
    <StarBorder
      as="div"
      className="w-full"
      color="#06b6d4"
      speed="7s"
      innerClassName="!space-y-4 !rounded-2xl !border-slate-700/70 !bg-gradient-to-br !from-slate-900/50 !to-slate-900/30 !p-8 !text-left !backdrop-blur"
    >
      <StarBorder as="h3" color="#06b6d4" speed="7s" className="text-lg font-semibold">
        Report history
      </StarBorder>
      <div className="space-y-3">
        {reports.map((report) => (
          <Link
            key={report.id}
            href={`/reports/${report.id}`}
            className="group block rounded-xl border border-slate-700 bg-slate-900/30 p-4 transition hover:border-blue-400 hover:bg-slate-900/60"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex gap-3 flex-1 min-w-0">
                <div className="mt-1 rounded-lg bg-slate-800/50 p-2">
                  <FileText size={16} className="text-blue-400" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium text-slate-200 group-hover:text-white transition">{report.file_name}</p>
                  <p className="mt-1 line-clamp-2 text-xs text-slate-400 group-hover:text-slate-300 transition">
                    {report.simplified_text}
                  </p>
                </div>
              </div>
              <div className="flex flex-col items-end gap-2">
                <div className="flex items-center gap-1 text-xs text-slate-500 whitespace-nowrap">
                  <Calendar size={12} />
                  {new Date(report.created_at).toLocaleDateString()}
                </div>
                <ArrowRight size={16} className="text-slate-600 group-hover:text-blue-400 transition translate-x-0 group-hover:translate-x-1" />
              </div>
            </div>
          </Link>
        ))}
      </div>
    </StarBorder>
  );
}
