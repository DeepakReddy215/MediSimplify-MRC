import type { AuthResponse, Report, ReportFeedbackRequest, UploadResponse } from "@/lib/types";

const normalizeBase = (base: string) => (base.endsWith("/") ? base.slice(0, -1) : base);
const configuredBase = process.env.NEXT_PUBLIC_API_URL?.trim();

const API_BASES = ["/api", configuredBase]
  .filter((value): value is string => Boolean(value && value.trim()))
  .map((value) => normalizeBase(value))
  .filter((value, index, arr) => arr.indexOf(value) === index);

const buildUrl = (base: string, path: string) => `${base}${path}`;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response | null = null;

  for (const base of API_BASES) {
    try {
      response = await fetch(buildUrl(base, path), {
        ...init,
        headers: {
          "Content-Type": "application/json",
          ...(init?.headers || {})
        }
      });
      break;
    } catch (error) {
      void error;
    }
  }

  if (!response) {
    throw new Error("Backend is unreachable. Please ensure the API server is running on port 8000.");
  }

  if (!response.ok) {
    if (response.status === 502 || response.status === 503 || response.status === 504) {
      throw new Error("Backend is temporarily unavailable. Please start/restart the API server and try again.");
    }
    const body = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(body.detail || "Request failed");
  }

  return response.json() as Promise<T>;
}

export async function signup(payload: {
  email: string;
  password: string;
  full_name: string;
}) {
  return request<AuthResponse>("/auth/signup", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function login(payload: { email: string; password: string }) {
  return request<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function getReports(token: string) {
  return request<Report[]>("/reports", {
    headers: {
      Authorization: `Bearer ${token}`
    }
  });
}

export async function getReportById(token: string, reportId: string) {
  return request<Report>(`/reports/${reportId}`, {
    headers: {
      Authorization: `Bearer ${token}`
    }
  });
}

export async function uploadReport(token: string, file: File, saveResult = true) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("save_result", String(saveResult));

  let response: Response | null = null;

  for (const base of API_BASES) {
    try {
      response = await fetch(buildUrl(base, "/reports/upload"), {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`
        },
        body: formData
      });
      break;
    } catch {
      // try next base
    }
  }

  if (!response) {
    throw new Error("Backend is unreachable. Please ensure the API server is running on port 8000.");
  }

  if (!response.ok) {
    if (response.status === 502 || response.status === 503 || response.status === 504) {
      throw new Error("Backend is temporarily unavailable. Please start/restart the API server and try again.");
    }
    const body = await response.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(body.detail || "Upload failed");
  }

  return response.json() as Promise<UploadResponse>;
}

export async function submitReportFeedback(
  token: string,
  reportId: string,
  payload: ReportFeedbackRequest
) {
  return request<{ message: string }>(`/reports/${reportId}/feedback`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify(payload)
  });
}
