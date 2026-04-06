export type User = {
  id: string;
  email: string;
  full_name: string;
  created_at: string;
};

export type AuthResponse = {
  access_token: string;
  token_type: string;
  user: User;
};

export type Report = {
  id: string;
  user_id: string;
  file_name: string;
  extracted_text: string;
  simplified_text: string;
  caregiver_text: string;
  important_terms: string[];
  glossary_entries: GlossaryEntry[];
  safety_alerts: SafetyAlert[];
  grounded_points: GroundedPoint[];
  created_at: string;
};

export type SafetyAlert = {
  code: string;
  title: string;
  severity: "urgent" | "warning" | string;
  recommendation: string;
  matched_text: string;
};

export type GroundedPoint = {
  statement: string;
  evidence_text: string;
  evidence_start: number;
  evidence_end: number;
  confidence: number;
};

export type GlossaryEntry = {
  term: string;
  plain_meaning: string;
  source_snippet: string;
};

export type ReportFeedbackRequest = {
  clarity_rating: number;
  accuracy_rating: number;
  corrected_text?: string;
  comment?: string;
};

export type UploadResponse = {
  saved: boolean;
  report: Report | null;
  file_name: string;
  extracted_text: string;
  simplified_text: string;
  caregiver_text: string;
  important_terms: string[];
  glossary_entries: GlossaryEntry[];
  safety_alerts: SafetyAlert[];
  grounded_points: GroundedPoint[];
  created_at: string;
};
