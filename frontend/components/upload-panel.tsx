"use client";

import { useRef, useState } from "react";
import { Upload, CheckCircle } from "lucide-react";
import { uploadReport } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import ClickSpark from "./ClickSpark";
import StarBorder from "./StarBorder";
import type { UploadResponse } from "@/lib/types";

export default function UploadPanel({
  onUploaded
}: {
  onUploaded: (result: UploadResponse) => void;
}) {
  const { token } = useAuth();
  const [fileName, setFileName] = useState<string>("");
  const [saveResult, setSaveResult] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleUpload = async () => {
    if (!token || !fileInputRef.current?.files?.[0]) {
      setError("Please choose a file first.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await uploadReport(token, fileInputRef.current.files[0], saveResult);
      onUploaded(result);
      setFileName("");
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <StarBorder
      as="section"
      className="w-full"
      color="#3b82f6"
      speed="7s"
      innerClassName="!overflow-hidden !rounded-2xl !border-slate-700/70 !bg-gradient-to-br !from-slate-900/50 !to-slate-900/30 !p-0 !text-left !backdrop-blur"
    >
      <div className="space-y-6 p-8">
        <div className="space-y-2">
          <StarBorder as="h2" color="#3b82f6" speed="7s" className="text-2xl font-bold">
            Upload medical report
          </StarBorder>
          <p className="text-sm text-slate-400">Supported: PNG, JPG, JPEG, PDF</p>
        </div>

        <label className="group relative block cursor-pointer rounded-2xl border-2 border-dashed border-slate-600 bg-slate-900/30 px-6 py-12 text-center transition hover:border-blue-400 hover:bg-slate-900/50">
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.pdf"
            className="hidden"
            onChange={(e) => setFileName(e.target.files?.[0]?.name || "")}
          />
          <div className="space-y-3">
            <div className="flex justify-center">
              <div className="rounded-lg bg-blue-500/20 p-3">
                <Upload className="text-blue-400" size={24} />
              </div>
            </div>
            <div>
              <p className="font-medium text-slate-200">Drop file here or click to browse</p>
              <p className="mt-1 text-xs text-slate-500">{fileName || "No file selected"}</p>
            </div>
          </div>
        </label>

        <div className="flex items-center gap-3 rounded-lg bg-slate-800/30 px-4 py-3">
          <input
            id="save_result"
            checked={saveResult}
            onChange={(e) => setSaveResult(e.target.checked)}
            type="checkbox"
            className="h-4 w-4 rounded border-slate-600 bg-slate-800 accent-blue-500"
          />
          <label htmlFor="save_result" className="flex items-center gap-2 text-sm text-slate-300">
            <CheckCircle size={16} className="text-emerald-500" />
            Save this report in my history
          </label>
        </div>

        {error && (
          <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        <ClickSpark sparkColor="#3b82f6" sparkSize={8} sparkCount={12} duration={600}>
          <button
            onClick={handleUpload}
            disabled={loading}
            className="w-full rounded-xl bg-gradient-to-r from-blue-500 to-blue-600 px-6 py-3 font-medium text-white transition disabled:opacity-50 hover:from-blue-600 hover:to-blue-700 disabled:cursor-not-allowed"
          >
            {loading ? "Processing..." : "Extract and simplify"}
          </button>
        </ClickSpark>
      </div>
    </StarBorder>
  );
}
