"use client";

import { useState } from "react";
import DashboardShell from "@/components/dashboard-shell";
import UploadPanel from "@/components/upload-panel";
import ResultPanel from "@/components/result-panel";
import type { UploadResponse } from "@/lib/types";

export default function DashboardUploadPage() {
  const [latestResult, setLatestResult] = useState<UploadResponse | null>(null);

  return (
    <DashboardShell
      activeTab="upload"
      title="Upload & OCR Workspace"
      subtitle="Upload medical files, run OCR extraction, and generate simplified explanations."
    >
      <section className="grid gap-6 lg:grid-cols-2">
        <UploadPanel onUploaded={setLatestResult} />
        <ResultPanel result={latestResult} />
      </section>
    </DashboardShell>
  );
}
