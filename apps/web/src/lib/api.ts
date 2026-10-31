export type CheckStatus = "pass" | "warning" | "fail" | "pending_model";
export type QualityStatus = "pass" | "warning" | "fail";
export type ExpectedView =
  | "left_front"
  | "right_front"
  | "left_rear"
  | "right_rear"
  | "interior_front"
  | "interior_rear"
  | "damage_detail";

export type InspectionStage = "pickup" | "return";
export type InspectionSessionStatus =
  | "draft"
  | "in_progress"
  | "capture_completed"
  | "cancelled";
export type InspectionPhotoStatus = "accepted" | "retry_required";
export type InspectionAnalysisStatus =
  | "not_requested"
  | "queued"
  | "processing"
  | "completed"
  | "needs_review"
  | "failed";

export interface InspectionPhoto {
  id: string;
  session_id: string;
  slot: ExpectedView;
  /** Exterior angle a damage close-up stands in for; it replaces that standard photo. */
  damage_view?: ExpectedView | null;
  filename: string;
  quality_request_id: string;
  quality_status: QualityStatus;
  quality_score: number;
  capture_status: InspectionPhotoStatus;
  analysis_status: InspectionAnalysisStatus;
  image_available: boolean;
  created_at: string;
  updated_at: string;
}

export interface InspectionSession {
  id: string;
  rental_id: string;
  vehicle_id: string;
  stage: InspectionStage;
  status: InspectionSessionStatus;
  required_slots: ExpectedView[];
  completed_slots: ExpectedView[];
  missing_slots: ExpectedView[];
  photos: InspectionPhoto[];
  created_at: string;
  updated_at: string;
  capture_completed_at: string | null;
}

export interface ActiveRentalContext {
  rental_id: string;
  vehicle_id: string;
  vehicle_label: string;
  license_plate: string | null;
  source: string;
  pickup_completed: boolean;
  return_allowed: boolean;
  completed_return_session_id: string | null;
  active_session_id: string | null;
}

export interface QualityCheck {
  key: string;
  label: string;
  status: CheckStatus;
  measured_value: number | string | null;
  threshold: string | null;
  message: string;
}

export interface ImageMetrics {
  width: number;
  height: number;
  megapixels: number;
  blur_variance: number;
  brightness: number;
  dynamic_range: number;
  dark_pixel_ratio: number;
  bright_pixel_ratio: number;
  orientation: string;
}

export interface ViewPrediction {
  predicted_view: ExpectedView;
  confidence: number;
  confidence_threshold: number;
  is_confident: boolean;
  expected_view: ExpectedView | null;
  matches_expected: boolean | null;
  latency_ms: number;
  model_version: string;
}

export interface CoveragePrediction {
  car_detected: boolean;
  confidence: number;
  bounding_box: number[];
  area_ratio: number;
  width_ratio: number;
  height_ratio: number;
  minimum_edge_margin: number;
  status: CheckStatus;
  latency_ms: number;
  model_version: string;
}

export interface InteriorCapturePrediction {
  scene_label: string;
  scene_confidence: number;
  scene_confidence_threshold: number;
  scene_confident: boolean;
  matches_expected: boolean;
  specialist_probability?: number | null;
  specialist_threshold?: number | null;
  rotation_probability: number | null;
  rotation_threshold: number | null;
  rotation_vetoed: boolean;
  latency_ms: number;
  model_version: string;
}

export interface ExteriorScenePrediction {
  front_interior_probability: number;
  rear_interior_probability: number;
  interior_probability: number;
  veto_threshold: number;
  is_interior: boolean;
  latency_ms: number;
  model_version: string;
}

export interface PlatePrediction {
  plate_detected: boolean;
  plate_text: string | null;
  normalized_plate: string | null;
  confidence: number;
  is_confident: boolean;
  bounding_box: number[] | null;
  image_usable: boolean;
  message: string;
  model_version: string;
  latency_ms: number;
  model_uncertain: boolean;
  timed_out: boolean;
  fast_matched?: boolean;
  matched_characters?: string | null;
}

export interface CaptureTestAnalysis {
  task: string;
  result: {
    assessment?: Partial<InteriorAnalysisResponse["assessment"]>;
    model?: string | null;
    model_pending?: boolean;
    summary?: string;
    comparison_status?: string;
    matched_pickup_detail?: number | null;
    risk_score?: number;
    automation?: { effective_route: string };
  };
  needs_review: boolean;
  error_message: string;
}

export interface QualityResponse {
  quality_skipped?: boolean;
  recognition_abandoned?: boolean;
  recognition_ms?: number | null;
  analyses?: CaptureTestAnalysis[];
  /** Client-side only: names of the pickup photos sent as comparison candidates. */
  pickup_reference_labels?: string[];
  request_id: string;
  filename: string;
  expected_view: ExpectedView | null;
  status: QualityStatus;
  score: number | null;
  summary: string;
  retake_required: boolean;
  checks: QualityCheck[];
  metrics: ImageMetrics | null;
  model_checks_ready: boolean;
  view_prediction: ViewPrediction | null;
  coverage_prediction: CoveragePrediction | null;
  interior_capture_prediction: InteriorCapturePrediction | null;
  exterior_scene_prediction: ExteriorScenePrediction | null;
  plate_prediction: PlatePrediction | null;
}

export interface CapturePreviewResponse {
  coverage_prediction: CoveragePrediction | null;
  messages: string[];
}

export interface InspectionCaptureResponse {
  quality: QualityResponse;
  session: InspectionSession;
}

export interface ReferenceSample {
  id: string;
  expected_label: string | null;
  corrected_label: string | null;
  score: number | null;
  confidence: number | null;
  lighting: string;
}

export interface AuthPrincipal {
  subject: string;
  audience: "user" | "ops";
  role: "user" | "staff" | "admin";
  display_name: string;
  rental_id: string | null;
  vehicle_id: string | null;
  vehicle_label: string | null;
  license_plate: string | null;
  session_id?: string | null;
}

export interface AuthSession {
  access_token: string;
  token_type: "bearer";
  expires_at: string;
  principal: AuthPrincipal;
}

export interface ManagedReferenceGuide {
  view: ExpectedView;
  title: string;
  description: string;
  image_available: boolean;
  image_url: string | null;
  filename: string | null;
  updated_at: string | null;
}

export type InteriorView = "front" | "rear" | "unknown";
export type CleanlinessLevel = "dirty" | "ordinary" | "clean" | "cannot_assess";
export type CleanlinessEvidenceCategory =
  | "trash"
  | "stain"
  | "liquid"
  | "food"
  | "pet_hair"
  | "clutter"
  | "floor"
  | "seat"
  | "other";

export interface CleanlinessEvidence {
  category: CleanlinessEvidenceCategory;
  severity: number;
  description: string;
  visible: boolean;
  bounding_box?: { x1: number; y1: number; x2: number; y2: number } | null;
}

export interface InteriorAnalysisResponse {
  request_id: string;
  rental_id: string;
  vehicle_id: string;
  filename: string;
  assessment: {
    is_vehicle_interior: boolean;
    image_usable: boolean;
    view: InteriorView;
    level: CleanlinessLevel;
    confidence: number;
    immediate_cleaning_required: boolean;
    specialist?: { model_version: string; score: number | null; threshold: number; suspected_dirty: boolean; human_review_required: boolean; score_is_probability: boolean } | null;
    summary: string;
    evidence: CleanlinessEvidence[];
    localization_status?: "not_requested" | "completed" | "unavailable";
    localization_model?: string | null;
  };
  model: string;
  latency_ms: number;
  alert_id: string | null;
  human_review_required: boolean;
}

export type DamageComparisonStatus = "no_change" | "review_required" | "incompatible";
export type DamagePairDecision = "confirmed_new_damage" | "no_new_damage" | "cannot_assess";

export interface DamageCandidate {
  id: string;
  bounding_box: number[];
  area_ratio: number;
  change_score: number;
  description: string;
}

export interface DamagePairVerification {
  decision: DamagePairDecision;
  confidence: number;
  same_vehicle_and_view: boolean;
  confirmed_region_ids: string[];
  summary: string;
  evidence: string;
  model: string;
  latency_ms: number;
}

export interface DamageCase {
  id: string;
  rental_id: string;
  vehicle_id: string;
  expected_view: ExpectedView;
  pickup_filename: string;
  return_filename: string;
  comparison: {
    status: DamageComparisonStatus;
    alignment: {
      success: boolean;
      matched_features: number;
      inlier_ratio: number;
      overlap_ratio: number;
      score: number;
      message: string;
    };
    candidates: DamageCandidate[];
    change_area_ratio: number;
    risk_score: number;
    summary: string;
    annotated_return_image: string | null;
    difference_mask_image: string | null;
    model_version: string;
    latency_ms: number;
    verification: DamagePairVerification | null;
    specialist?: {
      detected: boolean;
      score: number;
      threshold: number;
      pickup_match_score: number;
      model_version: string;
      latency_ms: number;
    } | null;
  };
  review_decision: "pending" | "confirmed_damage" | "dismissed" | "retry_photos";
  reviewer_notes: string;
  alert_id: string | null;
  created_at: string;
  reviewed_at: string | null;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

export async function getMobileAccess(): Promise<{ url: string | null; qr_data_url: string | null; message: string }> {
  return parseResponse(await fetch(`${API_BASE}/mobile/access`, { cache: "no-store" }));
}
const AUTH_STORAGE_KEY = "guardian-user-auth";

export function getStoredUserSession(): AuthSession | null {
  const raw = window.sessionStorage.getItem(AUTH_STORAGE_KEY)
    ?? window.localStorage.getItem(AUTH_STORAGE_KEY);
  if (!raw) return null;
  try {
    const session = JSON.parse(raw) as AuthSession;
    if (!session.access_token || new Date(session.expires_at).getTime() <= Date.now()) {
      window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
      return null;
    }
    window.sessionStorage.setItem(AUTH_STORAGE_KEY, raw);
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return session;
  } catch {
    window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return null;
  }
}

export function clearStoredUserSession(): void {
  window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
}

function authHeaders(): Record<string, string> {
  const session = getStoredUserSession();
  return session ? { Authorization: `Bearer ${session.access_token}` } : {};
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as {
      detail?: string | { message?: string };
    } | null;
    const detail = typeof payload?.detail === "string"
      ? payload.detail
      : payload?.detail?.message;
    throw new Error(detail ?? `伺服器回傳錯誤 (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function checkImageQuality(
  sessionId: string | null,
  file: File,
  expectedView: ExpectedView,
  captureFlowId?: string,
): Promise<QualityResponse> {
  const form = new FormData();
  form.append("image", file);
  form.append(sessionId ? "slot" : "expected_view", expectedView);
  if (captureFlowId) form.append("capture_flow_id", captureFlowId);
  return parseResponse<QualityResponse>(await fetch(
    sessionId ? `${API_BASE}/inspections/${encodeURIComponent(sessionId)}/photos/check` : `${API_BASE}/quality/check`,
    { method: "POST", headers: authHeaders(), body: form },
  ));
}

export async function detectCapturePreview(
  frame: Blob,
  signal: AbortSignal,
): Promise<CapturePreviewResponse> {
  const form = new FormData();
  form.append("image", frame, "preview.jpg");
  return parseResponse<CapturePreviewResponse>(await fetch(`${API_BASE}/capture/preview`, {
    method: "POST", headers: authHeaders(), body: form, signal,
  }));
}

export async function testCaptureQuality(
  file: File,
  expectedView: ExpectedView,
  stage: InspectionStage,
  pickupImage?: File,
  pickupDetails: File[] = [],
  damageView?: ExpectedView | null,
  captureFlowId?: string,
): Promise<QualityResponse> {
  const form = new FormData();
  form.append("image", file);
  form.append("expected_view", expectedView);
  form.append("stage", stage);
  if (captureFlowId) form.append("capture_flow_id", captureFlowId);
  if (damageView) form.append("damage_view", damageView);
  if (pickupImage) form.append("pickup_image", pickupImage);
  pickupDetails.forEach((detail) => form.append("pickup_details", detail));
  return parseResponse<QualityResponse>(await fetch(`${API_BASE}/capture/test`, {
    method: "POST", headers: authHeaders(), body: form,
  }));
}

export async function saveInspectionPhoto(
  sessionId: string,
  file: File,
  expectedView: ExpectedView,
  damageView?: ExpectedView | null,
): Promise<InspectionCaptureResponse> {
  const form = new FormData();
  form.append("image", file);
  form.append("slot", expectedView);
  if (damageView) form.append("damage_view", damageView);
  return parseResponse<InspectionCaptureResponse>(await fetch(
    `${API_BASE}/inspections/${encodeURIComponent(sessionId)}/photos`,
    { method: "POST", headers: authHeaders(), body: form },
  ));
}

export async function loginUser(values: {
  rental_id: string;
  password: string;
}): Promise<AuthSession> {
  const session = await parseResponse<AuthSession>(await fetch(`${API_BASE}/auth/user/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  }));
  window.sessionStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
  return session;
}

export async function logoutUser(): Promise<void> {
  const session = getStoredUserSession();
  try {
    if (session) {
      await fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        headers: { Authorization: `Bearer ${session.access_token}` },
      });
    }
  } finally {
    clearStoredUserSession();
  }
}

export async function createInspectionSession(
  stage: InspectionStage,
  restart = false,
): Promise<InspectionSession> {
  return parseResponse<InspectionSession>(await fetch(`${API_BASE}/inspections`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ stage, restart }),
  }));
}

export async function getInspectionRentalContext(): Promise<ActiveRentalContext> {
  return parseResponse<ActiveRentalContext>(await fetch(
    `${API_BASE}/inspections/context`, { headers: authHeaders() },
  ));
}

export async function getInspectionSession(sessionId: string): Promise<InspectionSession> {
  return parseResponse<InspectionSession>(await fetch(
    `${API_BASE}/inspections/${encodeURIComponent(sessionId)}`,
    { headers: authHeaders() },
  ));
}

export async function completeInspectionSession(
  sessionId: string,
): Promise<InspectionSession> {
  return parseResponse<InspectionSession>(await fetch(
    `${API_BASE}/inspections/${encodeURIComponent(sessionId)}/complete`,
    { method: "POST", headers: authHeaders() },
  ));
}

export async function getReferenceSamples(view: ExpectedView): Promise<ReferenceSample[]> {
  return parseResponse<ReferenceSample[]>(await fetch(
    `${API_BASE}/learning/references?view=${encodeURIComponent(view)}&limit=4`,
    { headers: authHeaders() },
  ));
}

export function learningImageUrl(sampleId: string): string {
  const token = getStoredUserSession()?.access_token;
  const query = new URLSearchParams();
  if (token) query.set("access_token", token);
  const suffix = query.size ? `?${query}` : "";
  return `${API_BASE}/learning/references/${encodeURIComponent(sampleId)}/image${suffix}`;
}

export async function getManagedReferenceGuides(): Promise<ManagedReferenceGuide[]> {
  return parseResponse<ManagedReferenceGuide[]>(await fetch(`${API_BASE}/admin/reference-guides`));
}

export async function compareDamagePair(
  pickupImage: File,
  returnImage: File,
  expectedView: ExpectedView,
): Promise<DamageCase> {
  const form = new FormData();
  form.append("pickup_image", pickupImage);
  form.append("return_image", returnImage);
  form.append("expected_view", expectedView);
  return parseResponse<DamageCase>(await fetch(`${API_BASE}/damage/compare-user`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  }));
}

export async function analyzeInteriorCleanliness(
  image: File,
  expectedView: Exclude<InteriorView, "unknown">,
): Promise<InteriorAnalysisResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("expected_view", expectedView);
  form.append("test_mode", "true");
  return parseResponse<InteriorAnalysisResponse>(await fetch(`${API_BASE}/interior/analyze`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  }));
}

export function managedReferenceGuideImageUrl(view: ExpectedView): string {
  return `${API_BASE}/admin/reference-guides/${encodeURIComponent(view)}/image`;
}
