"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import DashboardShell from "@/components/dashboard-shell";
import { getReportById, submitReportFeedback } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import type { Report } from "@/lib/types";

export default function ReportDetailsPage() {
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const { token, ready } = useAuth();
  const [report, setReport] = useState<Report | null>(null);
  const [audienceMode, setAudienceMode] = useState<"patient" | "caregiver">("patient");
  const [clarityRating, setClarityRating] = useState(4);
  const [accuracyRating, setAccuracyRating] = useState(4);
  const [correctedText, setCorrectedText] = useState("");
  const [comment, setComment] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<string | null>(null);
  const [savingFeedback, setSavingFeedback] = useState(false);

  useEffect(() => {
    if (!ready) return;
    if (!token) {
      router.push("/login");
      return;
    }

    const reportId = params.id;
    if (!reportId) return;

    const load = async () => {
      const data = await getReportById(token, reportId);
      setReport(data);
    };

    void load();
  }, [ready, token, params.id, router]);

  if (!report) {
    return <div className="grid min-h-screen place-items-center text-slate-400">Loading report...</div>;
  }

  const explanationText = audienceMode === "patient"
    ? report.simplified_text
    : (report.caregiver_text || report.simplified_text);

  const handleFeedbackSubmit = async () => {
    if (!token) return;
    setSavingFeedback(true);
    setFeedbackStatus(null);
    try {
      await submitReportFeedback(token, report.id, {
        clarity_rating: clarityRating,
        accuracy_rating: accuracyRating,
        corrected_text: correctedText,
        comment,
      });
      setFeedbackStatus("Feedback saved. Thank you.");
      setCorrectedText("");
      setComment("");
    } catch (error) {
      setFeedbackStatus(error instanceof Error ? error.message : "Failed to save feedback");
    } finally {
      setSavingFeedback(false);
    }
  };

  return (
    <DashboardShell
      activeTab="history"
      title={report.file_name}
      subtitle="Detailed report simplification output"
    >
      <div className="space-y-6">
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold">Original extracted text</h2>
          <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-slate-200">
            {report.extracted_text}
          </p>
        </div>

        <div className="glass-card border-blue-500/40 p-6">
          <h2 className="text-lg font-semibold">Simplified explanation</h2>
          <div className="mt-3 flex items-center gap-2">
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
          <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-blue-100">
            {explanationText}
          </p>
        </div>

        {report.glossary_entries.length > 0 && (
          <div className="glass-card border-fuchsia-500/30 bg-fuchsia-500/10 p-6">
            <h2 className="text-lg font-semibold text-fuchsia-100">Medical glossary</h2>
            <div className="mt-3 space-y-3">
              {report.glossary_entries.map((entry) => (
                <div key={entry.term} className="rounded-lg border border-fuchsia-500/20 bg-slate-900/40 p-4">
                  <p className="text-sm font-semibold text-fuchsia-100">{entry.term}</p>
                  <p className="mt-1 text-sm text-slate-100">{entry.plain_meaning}</p>
                  {entry.source_snippet && <p className="mt-1 text-xs text-slate-400">Context: {entry.source_snippet}</p>}
                </div>
              ))}
            </div>
          </div>
        )}

        {report.safety_alerts.length > 0 && (
          <div className="glass-card border-amber-500/30 bg-amber-500/10 p-6">
            <h2 className="text-lg font-semibold text-amber-100">Clinical safety alerts</h2>
            <div className="mt-3 space-y-3">
              {report.safety_alerts.map((alert) => (
                <div key={alert.code} className="rounded-lg border border-amber-500/20 bg-slate-900/40 p-4">
                  <p className="text-sm font-semibold text-amber-100">{alert.title}</p>
                  <p className="mt-1 text-xs text-amber-200">Match: {alert.matched_text}</p>
                  <p className="mt-1 text-sm text-slate-200">{alert.recommendation}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {report.grounded_points.length > 0 && (
          <div className="glass-card border-cyan-500/30 bg-cyan-500/10 p-6">
            <h2 className="text-lg font-semibold text-cyan-100">Source links</h2>
            <div className="mt-3 space-y-3">
              {report.grounded_points.map((point, index) => (
                <div key={`${point.evidence_start}-${index}`} className="rounded-lg border border-cyan-500/20 bg-slate-900/40 p-4">
                  <p className="text-sm text-cyan-100">{point.statement}</p>
                  <p className="mt-1 text-xs text-slate-300">Evidence: {point.evidence_text}</p>
                  <p className="mt-1 text-xs text-slate-400">Confidence: {(point.confidence * 100).toFixed(0)}%</p>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold">Important medical terms</h2>
          <div className="mt-3 flex flex-wrap gap-2">
            {report.important_terms.length ? (
              report.important_terms.map((term) => (
                <span
                  key={term}
                  className="rounded-full border border-slate-600 bg-slate-800 px-3 py-1 text-xs"
                >
                  {term}
                </span>
              ))
            ) : (
              <p className="text-sm text-slate-300">No key terms found.</p>
            )}
          </div>
        </div>

        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold">Help improve explanations</h2>
          <p className="mt-2 text-sm text-slate-300">Rate this result and optionally submit a corrected version.</p>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <label className="text-sm text-slate-200">
              Clarity rating (1-5)
              <select
                value={clarityRating}
                onChange={(e) => setClarityRating(Number(e.target.value))}
                className="mt-1 w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
              >
                {[1, 2, 3, 4, 5].map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>

            <label className="text-sm text-slate-200">
              Accuracy rating (1-5)
              <select
                value={accuracyRating}
                onChange={(e) => setAccuracyRating(Number(e.target.value))}
                className="mt-1 w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2"
              >
                {[1, 2, 3, 4, 5].map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>
          </div>

          <label className="mt-4 block text-sm text-slate-200">
            Corrected explanation (optional)
            <textarea
              value={correctedText}
              onChange={(e) => setCorrectedText(e.target.value)}
              rows={4}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-sm"
              placeholder="Write a better simplified version here"
            />
          </label>

          <label className="mt-4 block text-sm text-slate-200">
            Comment (optional)
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              rows={3}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-900 px-3 py-2 text-sm"
              placeholder="Tell us what was unclear"
            />
          </label>

          {feedbackStatus && <p className="mt-3 text-sm text-slate-300">{feedbackStatus}</p>}

          <button
            onClick={handleFeedbackSubmit}
            disabled={savingFeedback}
            className="mt-4 rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {savingFeedback ? "Saving..." : "Submit feedback"}
          </button>
        </div>
      </div>
    </DashboardShell>
  );
}
