"use client";

import { useState } from "react";
import { BookOpen, Sparkles } from "lucide-react";
import ClickSpark from "./ClickSpark";
import StarBorder from "./StarBorder";
import type { UploadResponse } from "@/lib/types";

export default function ResultPanel({ result }: { result: UploadResponse | null }) {
  const [audienceMode, setAudienceMode] = useState<"patient" | "caregiver">("patient");

  if (!result) {
    return (
      <StarBorder
        as="div"
        className="w-full"
        color="#06b6d4"
        speed="7s"
        innerClassName="!rounded-2xl !border-slate-700/70 !bg-gradient-to-br !from-slate-900/50 !to-slate-900/30 !p-8 !text-left !backdrop-blur"
      >
        <div className="flex flex-col items-center justify-center space-y-4 py-12">
          <div className="rounded-lg bg-slate-800/50 p-3">
            <BookOpen className="text-slate-500" size={32} />
          </div>
          <div className="text-center">
            <StarBorder as="h3" color="#06b6d4" speed="7s" className="text-lg font-semibold">
              Latest result
            </StarBorder>
            <p className="mt-3 text-sm text-slate-400">Upload a report to see extracted and simplified text.</p>
          </div>
        </div>
      </StarBorder>
    );
  }

  const explanationText = audienceMode === "patient"
    ? result.simplified_text
    : (result.caregiver_text || result.simplified_text);

  return (
    <ClickSpark sparkColor="#06b6d4" sparkSize={6} sparkCount={10} duration={500}>
      <StarBorder
        as="div"
        className="w-full"
        color="#06b6d4"
        speed="7s"
        innerClassName="!space-y-5 !rounded-2xl !border-slate-700/70 !bg-gradient-to-br !from-slate-900/50 !to-slate-900/30 !p-8 !text-left !backdrop-blur"
      >
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-3">
            <StarBorder as="h3" color="#06b6d4" speed="7s" className="text-lg font-semibold">
              Simplified output
            </StarBorder>
            <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-300">
              <div className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              {result.saved ? "Saved" : "Preview"}
            </span>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <BookOpen size={14} className="text-slate-500" />
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Original text</p>
          </div>
          <div className="max-h-32 overflow-y-auto rounded-lg border border-slate-700 bg-slate-900/50 p-4 text-sm leading-relaxed text-slate-300">
            {result.extracted_text || "No text could be extracted."}
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Sparkles size={14} className="text-blue-400" />
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Simplified explanation</p>
          </div>
          <div className="flex items-center gap-2 pb-1">
            <button
              onClick={() => setAudienceMode("patient")}
              className={`rounded-full px-3 py-1 text-xs transition ${
                audienceMode === "patient"
                  ? "border border-blue-400/50 bg-blue-500/20 text-blue-100"
                  : "border border-slate-700 bg-slate-900/50 text-slate-300"
              }`}
            >
              Patient mode
            </button>
            <button
              onClick={() => setAudienceMode("caregiver")}
              className={`rounded-full px-3 py-1 text-xs transition ${
                audienceMode === "caregiver"
                  ? "border border-cyan-400/50 bg-cyan-500/20 text-cyan-100"
                  : "border border-slate-700 bg-slate-900/50 text-slate-300"
              }`}
            >
              Caregiver mode
            </button>
          </div>
          <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-4 text-sm leading-relaxed text-blue-100">
            {explanationText}
          </div>
        </div>

        {result.glossary_entries.length > 0 && (
          <div className="space-y-3 rounded-lg border border-fuchsia-500/25 bg-fuchsia-500/10 p-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-fuchsia-200">Medical glossary</p>
            <div className="space-y-2">
              {result.glossary_entries.map((entry) => (
                <div key={entry.term} className="rounded-md border border-fuchsia-500/20 bg-slate-900/30 p-3">
                  <p className="text-sm font-semibold text-fuchsia-100">{entry.term}</p>
                  <p className="mt-1 text-xs text-slate-200">{entry.plain_meaning}</p>
                  {entry.source_snippet && <p className="mt-1 text-xs text-slate-400">Context: {entry.source_snippet}</p>}
                </div>
              ))}
            </div>
          </div>
        )}

        {result.safety_alerts.length > 0 && (
          <div className="space-y-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-amber-200">Clinical safety alerts</p>
            <div className="space-y-2">
              {result.safety_alerts.map((alert) => (
                <div key={alert.code} className="rounded-md border border-amber-500/20 bg-slate-900/30 p-3">
                  <p className="text-sm font-semibold text-amber-100">{alert.title}</p>
                  <p className="mt-1 text-xs text-amber-200/90">
                    Match: <span className="font-medium">{alert.matched_text}</span>
                  </p>
                  <p className="mt-1 text-xs text-slate-200">{alert.recommendation}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.grounded_points.length > 0 && (
          <div className="space-y-3 rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-cyan-200">Source links</p>
            <div className="space-y-2">
              {result.grounded_points.map((point, index) => (
                <div key={`${point.evidence_start}-${index}`} className="rounded-md border border-cyan-500/20 bg-slate-900/30 p-3">
                  <p className="text-sm text-cyan-100">{point.statement}</p>
                  <p className="mt-1 text-xs text-slate-300">Evidence: {point.evidence_text}</p>
                  <p className="mt-1 text-[11px] text-slate-400">Confidence: {(point.confidence * 100).toFixed(0)}%</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {result.important_terms.length > 0 && (
          <div className="space-y-3 border-t border-slate-700 pt-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Key medical terms</p>
            <div className="flex flex-wrap gap-2">
              {result.important_terms.map((term) => (
                <span
                  key={term}
                  className="inline-flex items-center rounded-full border border-slate-600 bg-slate-800/50 px-3 py-1 text-xs font-medium text-slate-200 backdrop-blur"
                >
                  {term}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center justify-between border-t border-slate-700 pt-4 text-xs text-slate-500">
          <span>Processed</span>
          <span>{new Date(result.created_at).toLocaleString()}</span>
        </div>
      </StarBorder>
    </ClickSpark>
  );
}
