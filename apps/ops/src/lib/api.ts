export type ExpectedView =
  | "left_front"
  | "right_front"
  | "left_rear"
  | "right_rear"
  | "interior_front"
  | "interior_rear"
  | "damage_detail";

export interface ReadinessReport {
  overall: "ready" | "degraded" | "blocked";
  checked_at: string;
  checks: Array<{
    key: string;
    label: string;
    status: "ready" | "warning" | "error";
    required: boolean;
    message: string;
    detail: string | null;
  }>;
  ready_count: number;
  total_count: number;
}

export type AlertStatus = "open" | "acknowledged" | "resolved" | "cancelled";
export interface OperationsAlert {
  id: string;
  type: string;
  severity: string;
  status: AlertStatus;
  rental_id: string;
  vehicle_id: string;
  title: string;
  summary: string;
  evidence: Array<{
    category: string;
    severity: number;
    description: string;
    visible: boolean;
    score_is_probability?: boolean;
    bounding_box?: { x1: number; y1: number; x2: number; y2: number } | number[] | null;
  }>;
  confidence: number;
  image_filename: string;
  image_url: string | null;
  created_at: string;
  acknowledged_at: string | null;
  resolved_at: string | null;
  cancelled_at: string | null;
  review_decision: "confirmed" | "dismissed" | null;
  reviewer_notes: string;
  reviewed_at: string | null;
}

export interface InspectionPhotoRecord {
  analysis_jobs?: RecognitionJobProgress[];
  id: string;
  session_id: string;
  rental_id: string;
  vehicle_id: string;
  stage: "pickup" | "return";
  session_status: string;
  slot: ExpectedView;
  /** Exterior angle an angle-tagged damage close-up stands in for. */
  damage_view?: ExpectedView | null;
  filename: string;
  quality_status: string;
  quality_score: number;
  capture_status: string;
  analysis_status: string;
  image_available: boolean;
  image_url: string | null;
  expected_plate: string | null;
  recognized_plate: string | null;
  human_verified_plate: string | null;
  plate_model_uncertain: boolean;
  created_at: string;
}

export interface RecognitionJobProgress {
  heartbeat_at?: string | null;
  id: string;
  photo_id: string;
  task: string;
  status: string;
  filename: string;
  slot: string;
  damage_view?: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  model_pending?: boolean;
}

export interface RecognitionProgress {
  jobs?: RecognitionJobProgress[];
  session_id: string;
  rental_id: string;
  vehicle_id: string;
  stage: "pickup" | "return" | "damage_pair";
  total: number;
  queued: number;
  processing: number;
  completed: number;
  failed: number;
  started_at: string;
  finished_at: string | null;
}

export interface DeleteInspectionPhotosResponse {
  deleted_records: number;
  deleted_images: number;
  message: string;
}

export type WorkOrderStatus = "queued" | "assigned" | "in_progress" | "completed" | "cancelled";
export interface WorkOrderStep {
  action: string;
  actor: string;
  summary: string;
  created_at: string;
}
export interface WorkOrder {
  id: string;
  alert_id: string;
  kind: string;
  status: WorkOrderStatus;
  priority: string;
  assignee: string | null;
  vehicle_id: string;
  notes: string;
  due_at: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
  history: WorkOrderStep[];
}

export interface OperationsActivity {
  id: string;
  entity_type: string;
  entity_id: string;
  action: string;
  actor: string;
  summary: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface ClearOperationsActivityResponse {
  deleted_activity_events: number;
  older_than_days: number;
  message: string;
}

export type AutomationRoute = "auto_clear" | "auto_case" | "manual_review";

export interface AutomationDecision {
  id: string;
  job_id: string;
  session_id: string;
  user_id: string;
  rental_id: string;
  vehicle_id: string;
  task: string;
  recommended_route: AutomationRoute;
  effective_route: AutomationRoute;
  shadow_mode: boolean;
  confidence: number | null;
  reason_codes: string[];
  evidence: Record<string, unknown>;
  policy_version: string;
  created_at: string;
}

export interface AutomationSummary {
  enabled: boolean;
  shadow_mode: boolean;
  policy_version: string;
  total: number;
  recommended: Record<string, number>;
  effective: Record<string, number>;
  simulated_review_reduction_rate: number;
  effective_manual_review_rate: number;
  decisions: AutomationDecision[];
}

export interface RentalEvent {
  id: string;
  user_id: string;
  rental_id: string;
  vehicle_id: string;
  session_id: string | null;
  event_type: string;
  stage: string | null;
  source_type: string;
  source_id: string;
  title: string;
  summary: string;
  outcome: string | null;
  severity: string | null;
  confidence: number | null;
  verified: boolean;
  disputed: boolean;
  metadata: Record<string, unknown>;
  created_at: string;
  occurred_on: string | null;
}

export interface UserHistoryListItem {
  user_id: string;
  display_name: string;
  rental_ids: string[];
  vehicle_ids: string[];
  rental_count: number;
  event_count: number;
  confirmed_damage_count: number;
  confirmed_cleanliness_count: number;
  pending_review_count: number;
  last_event_at: string | null;
  caused_damage_count: number;
  compensation_count: number;
  compensation_amount: string;
  paid_amount: string;
  outstanding_amount: string;
  blacklisted: boolean;
  blacklist_reason: string;
  litter_count: number;
  pending_litter_count: number;
  disputed_litter_count: number;
  has_litter_history: boolean;
  dirty_count: number;
  damage_compensation_count: number;
  damage_compensation_amount: string;
  damage_paid_amount: string;
  damage_outstanding_amount: string;
}

export interface UserHistoryProfile extends UserHistoryListItem {
  disputed_count: number;
  risk_level: "insufficient_data" | "low" | "watch" | "high" | string;
  risk_reasons: string[];
  events: RentalEvent[];
}

export interface HistoryChatContext {
  scope: "user" | "users" | "vehicle" | "vehicles" | "all";
  user_ids: string[];
  license_plates: string[];
  rental_ids: string[];
  topics: string[];
  date_from: string | null;
  date_to: string | null;
  intent?: HistoryIntent | null;
  focus_event_ids?: string[];
}
export type HistoryIntent = "aggregate" | "list_users" | "event_detail" | "timeline" | "blacklist" | "risk"
  | "prior_damage" | "clarify" | "capabilities";
export interface EvidenceRow {
  ref: number;
  event_id: string | null;
  rental_id: string | null;
  occurred_on: string | null;
  return_date: string | null;
  renter_name: string;
  license_plate: string | null;
  kind: string;
  kind_label: string;
  title: string;
  summary: string;
  responsibility: "confirmed" | "pending" | "disputed" | "ai_only" | "not_applicable";
  counted: boolean;
  present_at_pickup: boolean | null;
  present_at_return: boolean | null;
  compensation_amount: string | null;
  paid_amount: string | null;
  photo_slots?: ("pickup" | "return")[];
  photo_source: "rag_seed" | "damage_case" | null;
  damage_case_id: string | null;
  cited: boolean;
}
export interface CardMetric { label: string; value: string; tone: "neutral" | "good" | "warn" | "bad" }
export interface ChipOption { label: string; question: string }
export interface ScopeChip { kind: "scope" | "period" | "topic"; label: string; options: ChipOption[] }
export interface PersonRow {
  name: string;
  user_id?: string;
  license_plates: string[];
  caused_damage_count: number;
  unconfirmed_damage_count: number;
  disputed_count: number;
  litter_count: number;
  compensation_amount: string;
  outstanding_amount: string;
}
export interface RiskRow {
  rank: number;
  name: string;
  license_plates: string[];
  level: string;
  level_label: string;
  reasons: string[];
  rental_count: number;
  caused_damage_count: number;
  litter_count: number;
  disputed_count: number;
  outstanding_amount: string;
  blacklisted: boolean;
  score: number;
}
export interface PriorRental {
  renter_name: string;
  rental_date: string | null;
  return_date: string | null;
  records: string[];
  same_location: boolean;
  visual_score: number | null;
}
export interface PriorDamageCheck {
  focus_event_id: string | null;
  renter_name: string | null;
  license_plate: string | null;
  location: string | null;
  damage_type: string | null;
  verdict: "already_present" | "new_damage" | "unknown";
  verdict_text: string;
  visual_checked: boolean;
  earlier_rentals: PriorRental[];
}
/** One thing the assistant can do now (answer to 「你可以做什麼？」), with live status and example questions. */
export interface CapabilityItem {
  key: string;
  title: string;
  description: string;
  status: "ready" | "limited" | "off";
  status_label: string;
  examples: string[];
}
export interface HistoryAnswerCard {
  intent: HistoryIntent;
  headline: string;
  scope_label: string;
  period_label: string;
  topic_label: string;
  metrics: CardMetric[];
  rows: EvidenceRow[];
  total_rows: number;
  people: PersonRow[];
  people_columns?: Array<"damage" | "litter">;
  risk: RiskRow[];
  chips: ScopeChip[];
  followups: string[];
  detail_event_ids: string[];
  prior_damage: PriorDamageCheck | null;
  photo_limit: number;
  photo_total: number;
  capabilities?: CapabilityItem[];
}
export interface HistoryEventDetail {
  event_id: string;
  renter_name: string;
  license_plate: string | null;
  occurred_on: string | null;
  title: string;
  summary: string;
  responsibility: string;
  disputed: boolean;
  source: "damage_case" | "record_text" | "none";
  view_label: string | null;
  location: string | null;
  damage_type: string | null;
  present_at_pickup: boolean | null;
  present_at_return: boolean | null;
  conclusion: string;
  ai_summary: string | null;
  ai_score: number | null;
  pickup_match_score: number | null;
  review_decision: string | null;
  reviewer_notes: string | null;
  regions: Array<{ description: string; score: number | null; bounding_box: number[] | null }>;
  photo_slots?: ("pickup" | "return")[];
  photo_source: "rag_seed" | "damage_case" | null;
  damage_case_id: string | null;
}
export interface HistoryScopeOption {
  scope: "user" | "vehicle";
  value: string;
  label: string;
}
export interface HistoryQueryResponse {
  /** Set when an AI summary is shown; a 👎 sends it back so the summary becomes a regression case. */
  answer_id?: string | null;
  conversation_title?: string | null;
  agent_steps?: Array<Record<string, unknown>>;
  answer_card?: HistoryAnswerCard | null;
  license_plate?: string | null;
  context?: HistoryChatContext | null;
  clarification_options?: HistoryScopeOption[];
  question: string;
  user_id: string | null;
  query_mode: string;
  answer: string;
  generated_by: string;
  interpreted_filters: Record<string, unknown>;
  citations: Array<{
    event_id: string;
    rental_id: string;
    source_type: string;
    source_id: string;
    created_at: string;
  }>;
  matched_events: RentalEvent[];
  warnings: string[];
  statistics: Record<string, unknown>;
}

export type NotificationDeliveryStatus =
  | "pending"
  | "processing"
  | "power_automate_queued"
  | "power_automate_failed"
  | "sent"
  | "failed";

export interface NotificationOutboxItem {
  id: string;
  event_key: string;
  channel: "email";
  provider: string;
  recipient: string;
  subject: string;
  body_text: string;
  status: NotificationDeliveryStatus;
  attempts: number;
  max_attempts: number;
  last_error: string | null;
  next_attempt_at: string | null;
  handed_off_at: string | null;
  sent_at: string | null;
  source_type: string;
  source_id: string;
  created_at: string;
  updated_at: string;
}

export interface NotificationDispatchResponse {
  processed: number;
  handed_off: number;
  failed: number;
  message: string;
}

export type LearningTask =
  | "exterior_view"
  | "vehicle_coverage"
  | "interior_capture"
  | "interior_cleanliness"
  | "visible_damage"
  | "damage_pair"
  | "license_plate"
  | "capture_sharpness";
export type LearningStatus = "pending_review" | "approved" | "rejected" | "trained";
export type LearningAnnotationImageSlot = "primary" | "secondary";
export type LearningAnnotationKind = "issue" | "hard_negative" | "reference";
export interface LearningAnnotationPoint { x: number; y: number }
export interface LearningRegionAnnotation {
  image_slot: LearningAnnotationImageSlot;
  kind: LearningAnnotationKind;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  notes: string;
  shape?: "rectangle" | "polygon";
  points?: LearningAnnotationPoint[];
  normal_reference?: boolean;
  auto_refinement?: {
    original: LearningRegionGeometry;
    method: string;
    removed_area_ratio: number;
  } | null;
}
export interface LearningRegionGeometry {
  x: number; y: number; width: number; height: number;
  shape?: "rectangle" | "polygon";
  points?: LearningAnnotationPoint[];
}
export interface LearningVehicleIdentity {
  make: string;
  vehicle_type: "passenger_car" | "suv" | "van" | null;
  model?: string;
  generation?: string;
  seat_material?: string;
  seat_style?: string;
  seat_pattern?: string;
}

export interface LearningSample {
  id: string;
  task: LearningTask;
  status: LearningStatus;
  trigger_reasons: string[];
  primary_filename: string;
  secondary_filename: string | null;
  has_secondary_image: boolean;
  rental_id: string | null;
  vehicle_id: string | null;
  source_id: string | null;
  inspection_stage: "pickup" | "return" | null;
  inspection_slot: string | null;
  expected_label: string | null;
  predicted_label: string | null;
  corrected_label: string | null;
  confidence: number | null;
  score: number | null;
  lighting: string;
  auto_category: string;
  model_version: string | null;
  sample_kind: "hard_case" | "good_reference" | string;
  training_eligible: boolean;
  reference_enabled: boolean;
  metadata: Record<string, unknown>;
  reviewer_notes: string;
  occurrence_count: number;
  created_at: string;
  last_seen_at: string;
  reviewed_at: string | null;
  trained_at: string | null;
}


export type TrainingRunStatus = "queued" | "running" | "promoted" | "candidate" | "rejected" | "failed" | "skipped";
export type TrainingRunner = "deploy" | "candidate" | "none";
export interface TrainingRun {
  id: string;
  task: LearningTask;
  status: TrainingRunStatus;
  trigger: string;
  approved_samples: number;
  baseline_metrics: Record<string, number>;
  candidate_metrics: Record<string, number>;
  message: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface LearningStats {
  total: number;
  pending_review: number;
  approved: number;
  rejected: number;
  trained: number;
  by_task: Record<string, number>;
  by_lighting: Record<string, number>;
  approved_exterior_untrained: number;
  next_training_in: number;
  auto_train_enabled: boolean;
  min_approved_for_training: number;
  last_training_run: TrainingRun | null;
  approved_untrained: number;
  next_training_task: LearningTask | null;
  datasets: DatasetTrainingStatus[];
}

export interface DatasetTrainingStatus {
  task: LearningTask;
  runner: TrainingRunner;
  approved_untrained: number;
  trained: number;
  threshold: number;
  remaining: number;
  last_run: TrainingRun | null;
}

export interface AuthPrincipal {
  subject: string;
  audience: "user" | "ops";
  role: "user" | "staff" | "admin";
  display_name: string;
  customer_id?: string | null;
  session_id?: string | null;
}

export interface AuthSession {
  access_token: string;
  token_type: "bearer";
  expires_at: string;
  principal: AuthPrincipal;
}

export type ManagedAccountKind = "rental" | "staff" | "admin";

export interface ManagedAccount {
  subject: string;
  kind: ManagedAccountKind;
  username: string;
  display_name: string;
  customer_id: string | null;
  rental_date: string | null;
  return_date: string | null;
  rental_id: string | null;
  vehicle_id: string | null;
  vehicle_label: string | null;
  license_plate: string | null;
  created_by: string;
  created_by_name: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
  is_current: boolean;
}

export interface AccountManagementResponse {
  accounts: ManagedAccount[];
  active_account_count: number;
  active_session_count: number;
  active_device_count: number;
  active_device_window_minutes: number;
  active_inspection_count: number;
  queued_analysis_job_count: number;
  processing_analysis_job_count: number;
  analysis_queue_capacity: number;
  analysis_queue_at_capacity: boolean;
  generated_at: string;
}

export interface DeleteAccountResponse {
  deleted_subject: string;
  message: string;
  management: AccountManagementResponse;
}

export interface RecognitionSettings {
  quality_min_short_edge: number;
  quality_blur_fail: number;
  quality_blur_warning: number;
  blur_model_enabled: boolean;
  blur_model_fail_probability: number;
  blur_model_warning_probability: number;
  quality_brightness_low_fail: number;
  quality_brightness_low_warning: number;
  quality_brightness_high_warning: number;
  quality_brightness_high_fail: number;
  quality_min_dynamic_range: number;
  quality_max_clipped_ratio: number;
  view_confidence_threshold: number;
  vehicle_detection_confidence_threshold: number;
  vehicle_coverage_low_fail: number;
  vehicle_coverage_low_warning: number;
  vehicle_coverage_high_warning: number;
  vehicle_coverage_high_fail: number;
  vehicle_edge_margin_warning: number;
  vehicle_edge_margin_fail: number;
  interior_review_confidence_threshold: number;
  interior_alert_confidence_threshold: number;
  damage_alert_confidence_threshold: number;
  damage_review_confidence_threshold: number;
  damage_pair_confirm_confidence_threshold: number;
  damage_pair_no_change_confidence_threshold: number;
  plate_recognition_confidence_threshold: number;
  inspection_plate_check_enabled: boolean;
  automation_auto_clear_confidence: number;
  automation_auto_case_confidence: number;
  interior_scene_veto_confidence: number;
  cleanliness_fast_lane_enabled: boolean;
  cleanliness_fast_lane_color_threshold: number;
  cleanliness_fast_lane_min_area_ratio: number;
  cleanliness_vlm_only_when_suspected: boolean;
  plate_hint_min_width_px: number;
  plate_hint_min_height_px: number;
  plate_hint_min_sharpness: number;
  timeout_view_seconds: number;
  timeout_vehicle_detection_seconds: number;
  timeout_interior_capture_seconds: number;
  timeout_plate_seconds: number;
  timeout_front_cleanliness_seconds: number;
  timeout_rear_cleanliness_seconds: number;
  timeout_damage_pair_seconds: number;
  active_learning_enabled: boolean;
  active_learning_score_threshold: number;
  active_learning_confidence_threshold: number;
  active_learning_min_training_score: number;
  active_learning_collect_good_samples: boolean;
  active_learning_auto_train: boolean;
  active_learning_min_approved: number;
  auto_train_min_vehicle_coverage: number;
  auto_train_min_interior_capture: number;
  auto_train_min_damage_pair: number;
  active_learning_max_metric_drop: number;
  active_learning_min_test_macro_f1: number;
  active_learning_min_accepted_accuracy: number;
  active_learning_max_coverage_drop: number;
}

export interface RecognitionSettingsResponse {
  settings: RecognitionSettings;
  updated_at: string | null;
  updated_by: string | null;
}

export type ManagedModelSource = "deployed" | "auto_trained" | "manual_trained" | "history";
export type ManagedModelKind =
  | "exterior_view"
  | "interior_front"
  | "interior_rear"
  | "interior_front_cleanliness"
  | "interior_rear_cleanliness"
  | "damage_pair"
  | "vehicle_coverage"
  | "interior_capture";
export type ManagedModelStage = "deployed" | "first_version" | "trial" | "history";

export interface ManagedModelMetric {
  key: string;
  label: string;
  value: number;
  unit: "percent" | "seconds" | "score";
  target: number | null;
  higher_is_better: boolean;
  passed: boolean | null;
}

export interface ManagedViewModel {
  id: string;
  version: string;
  architecture: string;
  kind: ManagedModelKind;
  stage: ManagedModelStage;
  source: ManagedModelSource;
  is_current: boolean;
  quality_score: number | null;
  minimum_quality_score: number | null;
  quality_metric: string | null;
  label_source: string | null;
  production_approved: boolean;
  average_latency_ms?: number | null;
  latency_breakdown_ms?: Record<string, number> | null;
  evaluation?: ManagedModelMetric[];
  backbone_summary?: string | null;
  in_use?: boolean;
  can_activate?: boolean;
  recommendation_rank?: number | null;
  recommendation_note?: string | null;
  file_size_bytes: number;
  modified_at: string;
  sha256: string;
}

export interface ModelManagementResponse {
  models: ManagedViewModel[];
  queue: { model_ids: string[] };
  updated_at: string | null;
  updated_by: string | null;
}

export interface ActivateManagedModelResponse {
  activated_model_id: string;
  backup_model_id: string | null;
  message: string;
  management: ModelManagementResponse;
}

export interface DeleteManagedModelResponse {
  deleted_model_id: string;
  message: string;
  management: ModelManagementResponse;
}

export interface ManagedReferenceGuide {
  view: ExpectedView;
  title: string;
  description: string;
  image_available: boolean;
  image_url: string | null;
  filename: string | null;
  updated_at: string | null;
  updated_by: string | null;
}

export type DamageComparisonStatus = "no_change" | "review_required" | "incompatible";
export type DamageReviewDecision = "pending" | "confirmed_damage" | "dismissed" | "retry_photos";

export interface DamageCandidate {
  id: string;
  bounding_box: number[];
  area_ratio: number;
  change_score: number;
  description: string;
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
      method: string;
      matched_features: number;
      inlier_ratio: number;
      overlap_ratio: number;
      score: number;
      message: string;
    };
    candidates: DamageCandidate[];
    geometric_candidates?: DamageCandidate[];
    change_area_ratio: number;
    risk_score: number;
    summary: string;
    annotated_return_image: string | null;
    difference_mask_image: string | null;
    model_version: string;
    latency_ms: number;
    verification: {
      decision: "confirmed_new_damage" | "no_new_damage" | "cannot_assess";
      confidence: number;
      same_vehicle_and_view: boolean;
      confirmed_region_ids: string[];
      summary: string;
      evidence: string;
      model: string;
      latency_ms: number;
    } | null;
    specialist: {
      detected: boolean;
      score: number;
      threshold: number;
      pickup_match_score: number;
      model_version: string;
      latency_ms: number;
    } | null;
  };
  review_decision: DamageReviewDecision;
  reviewer_notes: string;
  alert_id: string | null;
  created_at: string;
  reviewed_at: string | null;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";
const AUTH_STORAGE_KEY = "guardian-ops-auth";

export function getStoredOpsSession(): AuthSession | null {
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

export function clearStoredOpsSession(): void {
  window.sessionStorage.removeItem(AUTH_STORAGE_KEY);
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
}

async function fetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const session = getStoredOpsSession();
  if (session) {
    headers.set("Authorization", `Bearer ${session.access_token}`);
  }
  return window.fetch(input, { ...init, headers });
}

export async function loginOps(username: string, password: string): Promise<AuthSession> {
  const session = await parseResponse<AuthSession>(await window.fetch(`${API_BASE}/auth/ops/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  }));
  window.sessionStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
  return session;
}

export async function logoutOps(): Promise<void> {
  const session = getStoredOpsSession();
  try {
    if (session) {
      await window.fetch(`${API_BASE}/auth/logout`, {
        method: "POST",
        headers: { Authorization: `Bearer ${session.access_token}` },
      });
    }
  } finally {
    clearStoredOpsSession();
  }
}

type ValidationIssue = { msg?: string };

function errorDetailMessage(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  // FastAPI validation errors (422) return a list of issues rather than a string.
  if (Array.isArray(detail)) {
    const messages = detail
      .map((issue: ValidationIssue) => issue?.msg?.replace(/^Value error, /, ""))
      .filter((message): message is string => Boolean(message));
    return messages.length ? messages.join("；") : null;
  }
  return null;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    throw new Error(errorDetailMessage(payload?.detail) ?? `伺服器回傳錯誤 (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function getAlerts(): Promise<OperationsAlert[]> {
  return parseResponse<OperationsAlert[]>(await fetch(`${API_BASE}/operations/alerts`, { cache: "no-store" }));
}

export async function getInspectionPhotoRecords(): Promise<InspectionPhotoRecord[]> {
  return parseResponse<InspectionPhotoRecord[]>(await fetch(
    `${API_BASE}/operations/inspection-photos?limit=500`,
    { cache: "no-store" },
  ));
}

export async function getRecognitionProgress(limit = 5): Promise<RecognitionProgress[]> {
  return parseResponse<RecognitionProgress[]>(await fetch(
    `${API_BASE}/operations/recognition-progress?limit=${limit}`, { cache: "no-store" },
  ));
}

export async function deleteInspectionPhotoRecord(
  photoId: string,
): Promise<DeleteInspectionPhotosResponse> {
  return parseResponse<DeleteInspectionPhotosResponse>(await fetch(
    `${API_BASE}/operations/inspection-photos/${encodeURIComponent(photoId)}`,
    { method: "DELETE" },
  ));
}

export async function clearInspectionPhotoRecords(): Promise<DeleteInspectionPhotosResponse> {
  return parseResponse<DeleteInspectionPhotosResponse>(await fetch(
    `${API_BASE}/operations/inspection-photos`,
    { method: "DELETE" },
  ));
}

export async function getReadiness(): Promise<ReadinessReport> {
  const report = await parseResponse<ReadinessReport>(
    await fetch(`${API_BASE}/readiness`, { cache: "no-store" }),
  );
  const checks = report.checks.filter((check) => check.key !== "cloud_fallback");
  return {
    ...report,
    checks,
    ready_count: checks.filter((check) => check.status === "ready").length,
    total_count: checks.length,
  };
}

export interface PlateModelInfo {
  name: string;
  version: string;
  engine: string;
  status: "active" | "unavailable";
  available: boolean;
  message: string;
  python: string;
  model_files: string[];
  source: string;
  license: string;
  requires_training: boolean;
  human_confirmation: boolean;
}

export async function getPlateModelInfo(): Promise<PlateModelInfo> {
  return parseResponse<PlateModelInfo>(await fetch(`${API_BASE}/plate/model-info`, { cache: "no-store" }));
}

export async function getWorkOrders(): Promise<WorkOrder[]> {
  return parseResponse<WorkOrder[]>(await fetch(`${API_BASE}/operations/work-orders`, { cache: "no-store" }));
}

export async function getOperationsActivity(offset = 0, includeArchived = false, before?: OperationsActivity): Promise<OperationsActivity[]> {
  const query = new URLSearchParams({ limit: "30", offset: String(before ? 0 : offset), include_archived: String(includeArchived) });
  if (before) { query.set("before_created_at", before.created_at); query.set("before_id", before.id); }
  return parseResponse<OperationsActivity[]>(await fetch(`${API_BASE}/operations/activity?${query}`, { cache: "no-store" }));
}

export async function downloadOperationsExcel(): Promise<void> {
  const response = await fetch(`${API_BASE}/operations/activity/export`, { cache: "no-store" });
  if (!response.ok) await parseResponse<unknown>(response);
  const url = URL.createObjectURL(await response.blob());
  const link = Object.assign(document.createElement("a"), { href: url, download: "全部操作紀錄.xlsx" });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function clearExpiredOperationsActivity(): Promise<ClearOperationsActivityResponse> {
  return parseResponse<ClearOperationsActivityResponse>(await fetch(
    `${API_BASE}/operations/activity/expired`,
    { method: "DELETE" },
  ));
}

export async function getAutomationSummary(): Promise<AutomationSummary> {
  return parseResponse<AutomationSummary>(await fetch(
    `${API_BASE}/history/automation?limit=30`,
    { cache: "no-store" },
  ));
}

export async function getHistoryUsers(search = ""): Promise<UserHistoryListItem[]> {
  const query = new URLSearchParams({ limit: "50" });
  if (search.trim()) query.set("search", search.trim());
  return parseResponse<UserHistoryListItem[]>(await fetch(
    `${API_BASE}/history/users?${query.toString()}`,
    { cache: "no-store" },
  ));
}

export async function getUserHistory(userId: string): Promise<UserHistoryProfile> {
  return parseResponse<UserHistoryProfile>(await fetch(
    `${API_BASE}/history/users/${encodeURIComponent(userId)}`,
    { cache: "no-store" },
  ));
}

export async function queryUserHistory(
  question: string,
  userId: string,
): Promise<HistoryQueryResponse> {
  return parseResponse<HistoryQueryResponse>(await fetch(`${API_BASE}/history/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(userId === "__all__"
      ? { question, all_users: true } : { question, user_id: userId }),
  }));
}

export async function getNotificationOutbox(): Promise<NotificationOutboxItem[]> {
  return parseResponse<NotificationOutboxItem[]>(await fetch(
    `${API_BASE}/operations/notifications?limit=50`,
    { cache: "no-store" },
  ));
}

export interface DamageHistoryRecordInput {
  rental_id: string;
  record_type: "damage";
  description: string;
  occurred_on: string | null;
  damage_case_id: string | null;
  compensation_amount: string;
  paid_amount: string;
  responsibility_confirmed: boolean;
  disputed: boolean;
}

export async function saveDamageHistoryRecord(
  values: DamageHistoryRecordInput, eventId?: string,
): Promise<RentalEvent> {
  return parseResponse<RentalEvent>(await fetch(
    `${API_BASE}/history/compensation-records${eventId ? `/${encodeURIComponent(eventId)}` : ""}`,
    { method: eventId ? "PUT" : "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values) },
  ));
}

export async function getDamageHistoryRecords(userId: string): Promise<RentalEvent[]> {
  return parseResponse<RentalEvent[]>(await fetch(
    `${API_BASE}/history/compensation-records?user_id=${encodeURIComponent(userId)}`,
    { cache: "no-store" },
  ));
}

export interface LitterHistoryRecordInput {
  claim_enabled?: boolean;
  compensation_amount?: string;
  paid_amount?: string;
  rental_id: string;
  description: string;
  occurred_on: string | null;
  cleanliness_alert_id: string | null;
  responsibility_confirmed: boolean;
  disputed: boolean;
}

export async function saveLitterHistoryRecord(
  values: LitterHistoryRecordInput, eventId?: string,
): Promise<RentalEvent> {
  return parseResponse<RentalEvent>(await fetch(
    `${API_BASE}/history/litter-records${eventId ? `/${encodeURIComponent(eventId)}` : ""}`,
    { method: eventId ? "PUT" : "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values) },
  ));
}

export async function getLitterHistoryRecords(userId: string): Promise<RentalEvent[]> {
  return parseResponse<RentalEvent[]>(await fetch(
    `${API_BASE}/history/litter-records?user_id=${encodeURIComponent(userId)}`,
    { cache: "no-store" },
  ));
}

export async function setHistoryBlacklist(
  userId: string, blacklisted: boolean, reason: string,
): Promise<UserHistoryProfile> {
  return parseResponse<UserHistoryProfile>(await fetch(
    `${API_BASE}/history/users/${encodeURIComponent(userId)}/blacklist`,
    { method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ blacklisted, reason }) },
  ));
}

export async function deleteHistoryRecord(event: RentalEvent): Promise<void> {
  const resource = event.event_type === "cleanliness_confirmed" ? "cleanliness-records"
    : event.event_type === "litter_recorded" ? "litter-records" : "compensation-records";
  const response = await fetch(`${API_BASE}/history/${resource}/${encodeURIComponent(event.id)}`, {
    method: "DELETE",
  });
  if (!response.ok) await parseResponse<unknown>(response);
}

export async function saveCleanlinessHistoryRecord(
  values: Omit<LitterHistoryRecordInput, "cleanliness_alert_id">, eventId: string,
): Promise<RentalEvent> {
  return parseResponse<RentalEvent>(await fetch(
    `${API_BASE}/history/cleanliness-records/${encodeURIComponent(eventId)}`,
    { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values) },
  ));
}

export async function dispatchNotificationOutbox(): Promise<NotificationDispatchResponse> {
  return parseResponse<NotificationDispatchResponse>(await fetch(
    `${API_BASE}/operations/notifications/dispatch`,
    { method: "POST" },
  ));
}

export async function retryNotification(
  notificationId: string,
): Promise<NotificationOutboxItem> {
  return parseResponse<NotificationOutboxItem>(await fetch(
    `${API_BASE}/operations/notifications/${encodeURIComponent(notificationId)}/retry`,
    { method: "POST" },
  ));
}

export interface ClearTestDataResponse {
  deleted_alerts: number;
  deleted_work_orders: number;
  deleted_damage_cases: number;
  deleted_activity_events: number;
  deleted_notification_events: number;
  message: string;
}

export interface ClearDamageReviewDuplicatesResponse {
  deleted_damage_cases: number;
  deleted_analysis_jobs: number;
  message: string;
}

export async function getLearningSamples(
  status: LearningStatus = "pending_review",
  task?: LearningTask,
): Promise<LearningSample[]> {
  const pageSize = 500;
  const samples: LearningSample[] = [];
  for (let offset = 0; ; offset += pageSize) {
    const params = new URLSearchParams({
      status,
      limit: String(pageSize),
      offset: String(offset),
    });
    if (task) params.set("task", task);
    if (status === "pending_review") params.set("pending_inbox_only", "true");
    const page = await parseResponse<LearningSample[]>(await fetch(
      `${API_BASE}/learning/samples?${params.toString()}`,
      { cache: "no-store" },
    ));
    samples.push(...page);
    if (page.length < pageSize) return samples;
  }
}

export async function getLearningStats(): Promise<LearningStats> {
  return parseResponse<LearningStats>(await fetch(`${API_BASE}/learning/stats`, { cache: "no-store" }));
}


export async function getRecentLearningApprovals(limit = 5): Promise<LearningSample[]> {
  return parseResponse<LearningSample[]>(await fetch(
    `${API_BASE}/learning/recent-approvals?limit=${limit}`,
    { cache: "no-store" },
  ));
}

export function learningImageUrl(sampleId: string, slot: "primary" | "secondary"): string {
  const query = new URLSearchParams();
  const token = getStoredOpsSession()?.access_token;
  if (token) query.set("access_token", token);
  const suffix = query.size ? `?${query}` : "";
  return `${API_BASE}/learning/samples/${sampleId}/images/${slot}${suffix}`;
}

export function damageCaseImageUrl(caseId: string, slot: "pickup" | "return"): string {
  return authenticatedAssetUrl(
    `/api/v1/damage/cases/${encodeURIComponent(caseId)}/images/${slot}`,
  );
}

export function authenticatedAssetUrl(path: string): string {
  if (path.startsWith("data:")) return path;
  const apiUrl = new URL(API_BASE, window.location.origin);
  const url = new URL(path, apiUrl.origin);
  const token = getStoredOpsSession()?.access_token;
  if (token) url.searchParams.set("access_token", token);
  return url.toString();
}

export async function getRagSeeds(): Promise<RentalEvent[]> {
  return parseResponse<RentalEvent[]>(await fetch(`${API_BASE}/admin/rag-seeds`, { cache: "no-store" }));
}

export interface VehicleHistorySummary {
  license_plate: string;
  vehicle_label: string;
  rental_count: number;
  renter_count: number;
  event_count: number;
  latest_return_date: string | null;
}

export interface VehicleRentalHistory {
  rental_id: string;
  renter_id: string;
  renter_name: string;
  rental_date: string | null;
  return_date: string | null;
  uploaded: boolean;
}

export interface VehicleHistoryProfile extends VehicleHistorySummary {
  rentals: VehicleRentalHistory[];
  events: RentalEvent[];
  statistics: Record<string, string | number | boolean>;
}

export async function getVehicleHistory(plate: string): Promise<VehicleHistoryProfile> {
  return parseResponse<VehicleHistoryProfile>(await fetch(`${API_BASE}/history/vehicles/${encodeURIComponent(plate)}`, { cache: "no-store", signal: AbortSignal.timeout(15000) }));
}

export async function getSavedVehicleHistories(): Promise<VehicleHistoryProfile[]> {
  const vehicles = await parseResponse<VehicleHistorySummary[]>(await fetch(`${API_BASE}/history/vehicles`, {
    cache: "no-store", signal: AbortSignal.timeout(15000),
  }));
  const profiles = await Promise.all(vehicles.map((vehicle) => getVehicleHistory(vehicle.license_plate)));
  return profiles.map((profile) => {
    const rentals = profile.rentals.filter((rental) => rental.uploaded);
    const rentalIds = new Set(rentals.map((rental) => rental.rental_id));
    const events = profile.events.filter((event) => rentalIds.has(event.rental_id));
    return { ...profile, rentals, events, rental_count: rentals.length,
      renter_count: new Set(rentals.map((rental) => rental.renter_id)).size, event_count: events.length };
  }).filter((profile) => profile.rental_count > 0);
}

export async function uploadVehicleHistory(records: unknown[], photos: { pickup: File[]; returned: File[]; evidence: File[] }[]): Promise<RentalEvent[]> {
  const form = new FormData();
  form.append("payload", JSON.stringify({ records }));
  form.append("photo_counts", JSON.stringify(photos.map((pair) => ({ pickup: pair.pickup.length, returned: pair.returned.length, evidence: pair.evidence.length }))));
  for (const pair of photos) {
    for (const file of pair.pickup) form.append("pickups", file);
    for (const file of pair.returned) form.append("returns", file);
    for (const file of pair.evidence) form.append("evidence", file);
  }
  return parseResponse<RentalEvent[]>(await fetch(`${API_BASE}/history/vehicles/import`, { method: "POST", body: form }));
}

export async function createRagSeed(payload: unknown, pickup: File | null, returned: File): Promise<RentalEvent[]> {
  const form = new FormData();
  form.append("payload", JSON.stringify(payload));
  if (pickup) form.append("pickup", pickup);
  form.append("returned", returned);
  return parseResponse<RentalEvent[]>(await fetch(`${API_BASE}/admin/rag-seeds`, { method: "POST", body: form }));
}

export function ragSeedImageUrl(eventId: string, slot: string): string {
  const url = new URL(`${API_BASE}/admin/rag-seeds/${encodeURIComponent(eventId)}/images/${encodeURIComponent(slot)}`, window.location.origin);
  const token = getStoredOpsSession()?.access_token;
  if (token) url.searchParams.set("access_token", token);
  return url.toString();
}

export async function getRecognitionSettings(): Promise<RecognitionSettingsResponse> {
  return parseResponse<RecognitionSettingsResponse>(await fetch(
    `${API_BASE}/admin/recognition-settings`, { cache: "no-store" },
  ));
}

export async function getManagedAccounts(): Promise<AccountManagementResponse> {
  return parseResponse<AccountManagementResponse>(await fetch(
    `${API_BASE}/admin/accounts`, { cache: "no-store" },
  ));
}

export async function createRentalAccount(values: {
  rental_id: string;
  customer_id?: string;
  display_name?: string;
  rental_date?: string | null;
  return_date?: string | null;
  vehicle_id?: string;
  vehicle_label: string;
  license_plate: string;
  password: string;
}): Promise<ManagedAccount> {
  return parseResponse<ManagedAccount>(await fetch(`${API_BASE}/admin/accounts/rentals`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  }));
}

export async function updateRentalAccount(
  rentalId: string,
  values: { customer_id?: string; display_name?: string; rental_date?: string | null; return_date?: string | null; vehicle_id?: string; vehicle_label: string; license_plate: string; password?: string },
): Promise<ManagedAccount> {
  return parseResponse<ManagedAccount>(await fetch(
    `${API_BASE}/admin/accounts/rentals/${encodeURIComponent(rentalId)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    },
  ));
}

export async function createOpsAccount(values: {
  username: string;
  display_name: string;
  kind: "staff" | "admin";
  password: string;
}): Promise<ManagedAccount> {
  return parseResponse<ManagedAccount>(await fetch(`${API_BASE}/admin/accounts/ops`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  }));
}

export async function updateOpsAccount(
  username: string,
  values: {
    display_name: string;
    kind: "staff" | "admin";
    password?: string;
  },
): Promise<ManagedAccount> {
  return parseResponse<ManagedAccount>(await fetch(
    `${API_BASE}/admin/accounts/ops/${encodeURIComponent(username)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    },
  ));
}

export async function deleteManagedAccount(
  account: ManagedAccount,
  confirmationIdentifier: string,
  adminPassword: string,
): Promise<DeleteAccountResponse> {
  const segment = account.kind === "rental" ? "rentals" : "ops";
  return parseResponse<DeleteAccountResponse>(await fetch(
    `${API_BASE}/admin/accounts/${segment}/${encodeURIComponent(account.username)}`,
    {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: account.kind === "rental" ? undefined : JSON.stringify({
        confirmation_identifier: confirmationIdentifier,
        admin_password: adminPassword,
      }),
    },
  ));
}

export async function updateRecognitionSettings(
  settings: RecognitionSettings,
): Promise<RecognitionSettingsResponse> {
  return parseResponse<RecognitionSettingsResponse>(await fetch(
    `${API_BASE}/admin/recognition-settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    },
  ));
}

export async function getModelManagement(): Promise<ModelManagementResponse> {
  return parseResponse<ModelManagementResponse>(await fetch(
    `${API_BASE}/admin/model-management`, { cache: "no-store" },
  ));
}

export async function updateModelManagement(
  modelIds: string[],
): Promise<ModelManagementResponse> {
  return parseResponse<ModelManagementResponse>(await fetch(
    `${API_BASE}/admin/model-management`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_ids: modelIds }),
    },
  ));
}

export async function activateDamagePairModel(
  modelId: string,
  adminPassword: string,
): Promise<ActivateManagedModelResponse> {
  return parseResponse<ActivateManagedModelResponse>(await fetch(
    `${API_BASE}/admin/model-management/damage-pair/activate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmation_model_id: modelId, admin_password: adminPassword }),
    },
  ));
}

export async function deleteManagedModel(
  modelId: string,
  adminPassword: string,
): Promise<DeleteManagedModelResponse> {
  return parseResponse<DeleteManagedModelResponse>(await fetch(
    `${API_BASE}/admin/model-management/${encodeURIComponent(modelId)}`,
    {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confirmation_model_id: modelId,
        admin_password: adminPassword,
      }),
    },
  ));
}

export async function getManagedReferenceGuides(): Promise<ManagedReferenceGuide[]> {
  return parseResponse<ManagedReferenceGuide[]>(await fetch(
    `${API_BASE}/admin/reference-guides`, { cache: "no-store" },
  ));
}

export function managedReferenceGuideImageUrl(view: ExpectedView): string {
  return `${API_BASE}/admin/reference-guides/${encodeURIComponent(view)}/image`;
}

export async function updateManagedReferenceGuide(
  view: ExpectedView,
  values: { title: string; description: string; image?: File | null },
): Promise<ManagedReferenceGuide> {
  const form = new FormData();
  form.append("title", values.title);
  form.append("description", values.description);
  if (values.image) form.append("image", values.image);
  return parseResponse<ManagedReferenceGuide>(await fetch(
    `${API_BASE}/admin/reference-guides/${encodeURIComponent(view)}`,
    { method: "PUT", body: form },
  ));
}

export async function addLearningSampleToTrainingSet(
  sampleId: string,
  approval: {
    corrected_label: string;
    reviewer_notes?: string;
    annotations?: LearningRegionAnnotation[];
    vehicle_identity?: LearningVehicleIdentity | null;
    review_method?: "region_annotation" | "manual_pair" | "manual_single";
    pair_angle_result?: "same_view" | "incompatible" | null;
  },
): Promise<LearningSample> {
  return parseResponse<LearningSample>(await fetch(`${API_BASE}/learning/samples/${sampleId}/training-set`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(approval),
  }));
}

export async function cancelLearningSampleApproval(
  sampleId: string,
): Promise<LearningSample> {
  return parseResponse<LearningSample>(await fetch(
    `${API_BASE}/learning/samples/${sampleId}/approval`,
    { method: "DELETE" },
  ));
}

export type ManualTruthDestination = "truth" | "pending";

export type ManualTruthUploadResponse = {
  task: LearningTask;
  corrected_label: string;
  destination: ManualTruthDestination;
  inspection_slot: string | null;
  created: number;
  results: {
    filename: string;
    status: "created" | "duplicate" | "failed";
    sample_id: string | null;
    message: string;
  }[];
};

export async function uploadManualTruths(values: {
  task: LearningTask;
  correctedLabel: string;
  destination: ManualTruthDestination;
  images: File[];
  secondaryImage?: File | null;
  inspectionSlot?: string | null;
  reviewerNotes?: string;
}): Promise<ManualTruthUploadResponse> {
  const form = new FormData();
  form.append("task", values.task);
  form.append("corrected_label", values.correctedLabel);
  form.append("destination", values.destination);
  values.images.forEach((file) => form.append("images", file));
  if (values.secondaryImage) form.append("secondary_image", values.secondaryImage);
  if (values.inspectionSlot) form.append("inspection_slot", values.inspectionSlot);
  if (values.reviewerNotes) form.append("reviewer_notes", values.reviewerNotes);
  const response = await fetch(`${API_BASE}/learning/manual-truths`, { method: "POST", body: form });
  if (response.status === 429 || response.status === 503) {
    const seconds = Number(response.headers.get("Retry-After"));
    throw new ManualTruthBusyError(Number.isFinite(seconds) && seconds > 0 ? seconds * 1000 : 2000);
  }
  return parseResponse<ManualTruthUploadResponse>(response);
}

// The server is throttling concurrent upload batches; retry after the given delay.
export class ManualTruthBusyError extends Error {
  readonly retryAfterMs: number;

  constructor(retryAfterMs: number) {
    super("伺服器忙碌中，稍後自動重試");
    this.retryAfterMs = retryAfterMs;
  }
}

export async function checkManualTruthDuplicates(task: LearningTask, hashes: string[]): Promise<string[]> {
  const payload = await parseResponse<{ duplicates: string[] }>(await fetch(
    `${API_BASE}/learning/manual-truths/duplicates`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, hashes }),
    },
  ));
  return payload.duplicates;
}

export interface DeleteLearningSampleResponse {
  sample_id: string;
  deleted_images: number;
  message: string;
}

export interface LearningRentalSummary {
  rental_id: string;
  vehicle_id: string;
  sample_count: number;
  oldest_created_at: string;
  newest_created_at: string;
}

export interface DeleteLearningRentalResponse {
  rental_id: string;
  vehicle_id: string;
  deleted_samples: number;
  deleted_images: number;
  message: string;
}

export async function getLearningRentals(): Promise<LearningRentalSummary[]> {
  return parseResponse<LearningRentalSummary[]>(await fetch(
    `${API_BASE}/learning/rentals`, { cache: "no-store" },
  ));
}

export async function deleteLearningRental(
  rental: LearningRentalSummary,
  sampleIds: string[],
): Promise<DeleteLearningRentalResponse> {
  const params = new URLSearchParams({
    rental_id: rental.rental_id,
    vehicle_id: rental.vehicle_id,
    expected_sample_count: String(rental.sample_count),
  });
  return parseResponse<DeleteLearningRentalResponse>(await fetch(
    `${API_BASE}/learning/rentals/pending?${params.toString()}`,
    { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify(sampleIds) },
  ));
}

export async function deleteLearningSample(
  sampleId: string,
): Promise<DeleteLearningSampleResponse> {
  return parseResponse<DeleteLearningSampleResponse>(await fetch(
    `${API_BASE}/learning/samples/${sampleId}`,
    { method: "DELETE" },
  ));
}

export async function clearTestData(): Promise<ClearTestDataResponse> {
  return parseResponse<ClearTestDataResponse>(await fetch(
    `${API_BASE}/operations/test-data`, { method: "DELETE" },
  ));
}

export async function clearDamageReviewDuplicates(): Promise<ClearDamageReviewDuplicatesResponse> {
  return parseResponse<ClearDamageReviewDuplicatesResponse>(await fetch(
    `${API_BASE}/operations/damage-review/duplicates`, { method: "DELETE" },
  ));
}

export async function clearPendingDamageReview(): Promise<ClearDamageReviewDuplicatesResponse> {
  return parseResponse<ClearDamageReviewDuplicatesResponse>(await fetch(
    `${API_BASE}/operations/damage-review/pending`, { method: "DELETE" },
  ));
}

export async function acknowledgeAlert(alertId: string): Promise<OperationsAlert> {
  return parseResponse<OperationsAlert>(await fetch(
    `${API_BASE}/operations/alerts/${alertId}/acknowledge`, {
      method: "PATCH",
    },
  ));
}

export async function cancelAlert(alertId: string): Promise<OperationsAlert> {
  return parseResponse<OperationsAlert>(await fetch(
    `${API_BASE}/operations/alerts/${alertId}/cancel`, {
      method: "PATCH",
    },
  ));
}

export async function reviewCleanlinessAlert(
  alertId: string,
  decision: "confirmed" | "dismissed",
): Promise<OperationsAlert> {
  return parseResponse<OperationsAlert>(await fetch(
    `${API_BASE}/operations/alerts/${alertId}/cleanliness-review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision }),
    },
  ));
}

export async function createOperationsWorkOrder(alertId: string): Promise<WorkOrder> {
  return parseResponse<WorkOrder>(await fetch(
    `${API_BASE}/operations/alerts/${alertId}/work-order`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ assignee: null, notes: "由 AI 異常警示與人工覆核建立" }),
    },
  ));
}

export async function updateWorkOrder(
  workOrderId: string,
  changes: {
    status?: WorkOrderStatus;
    assignee?: string;
    notes?: string;
  },
): Promise<WorkOrder> {
  return parseResponse<WorkOrder>(await fetch(`${API_BASE}/operations/work-orders/${workOrderId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  }));
}

/** Administrator only: removes the order; the source alert stays so a new order can be opened. */
export async function deleteWorkOrder(workOrderId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/operations/work-orders/${workOrderId}`, { method: "DELETE" });
  if (!response.ok) await parseResponse<unknown>(response);
}

export async function getDamageCases(): Promise<DamageCase[]> {
  return parseResponse<DamageCase[]>(await fetch(`${API_BASE}/damage/cases`, { cache: "no-store" }));
}

/** Re-run the two-stage comparison on a pending case's stored photos. */
export async function reanalyzeDamageCase(caseId: string): Promise<DamageCase> {
  return parseResponse<DamageCase>(await fetch(`${API_BASE}/damage/cases/${caseId}/reanalyze`, {
    method: "POST",
  }));
}

export async function reviewDamageCase(
  caseId: string,
  decision: Exclude<DamageReviewDecision, "pending">,
  reviewerNotes = "",
): Promise<DamageCase> {
  return parseResponse<DamageCase>(await fetch(`${API_BASE}/damage/cases/${caseId}/review`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, reviewer_notes: reviewerNotes }),
  }));
}

export async function queryAssistantHistory(question: string, context: HistoryChatContext | null,
  selection?: HistoryScopeOption, messages: Array<{ role: "user" | "assistant"; content: string }> = []): Promise<HistoryQueryResponse> {
  return parseResponse<HistoryQueryResponse>(await fetch(`${API_BASE}/history/query`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, context, messages, ...(selection ? { selection } : {}) }),
  }));
}

export type HistoryStreamEvent =
  | { type: "card"; data: HistoryQueryResponse }
  | { type: "status"; data: { state: "queued" } }
  | { type: "delta"; data: { text: string } }
  | { type: "done"; data: HistoryQueryResponse }
  | { type: "error"; data: { message: string } };

/** Card first, then verified summary sentences; resolves with the final response from the `done` event. */
export async function streamAssistantHistory(question: string, context: HistoryChatContext | null,
  options: { selection?: HistoryScopeOption; messages?: Array<{ role: "user" | "assistant"; content: string }>;
    modelId?: string | null; signal?: AbortSignal; onEvent: (event: HistoryStreamEvent) => void }): Promise<HistoryQueryResponse> {
  const response = await fetch(`${API_BASE}/history/query/stream`, {
    method: "POST", signal: options.signal,
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, context, messages: options.messages ?? [],
      ...(options.selection ? { selection: options.selection } : {}),
      ...(options.modelId ? { model_id: options.modelId } : {}) }),
  });
  if (!response.ok || !response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    // Errors (401/403/422) and plain JSON replies, e.g. from a test double, take the regular path.
    const result = await parseResponse<HistoryQueryResponse>(response);
    options.onEvent({ type: "done", data: result });
    return result;
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let final: HistoryQueryResponse | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer = (buffer + value).replace(/\r\n/g, "\n");
    for (let index = buffer.indexOf("\n\n"); index >= 0; index = buffer.indexOf("\n\n")) {
      const block = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      let name = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue;
      const event = { type: name, data: JSON.parse(data.join("\n")) } as HistoryStreamEvent;
      options.onEvent(event);
      if (event.type === "done") final = event.data;
      if (event.type === "error") throw new Error(event.data.message);
    }
  }
  if (!final) throw new Error("連線中斷，請再試一次。");
  return final;
}

/** Every record behind an answer as .xlsx (the chat shows only the first photo groups). */
export async function downloadHistoryExcel(request: {
  question: string; context?: HistoryChatContext | null; license_plate?: string | null;
}): Promise<void> {
  const body = request.license_plate ? { question: request.question, license_plate: request.license_plate }
    : { question: request.question, context: request.context ?? null };
  const response = await fetch(`${API_BASE}/history/query/export`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) await parseResponse<unknown>(response);
  const disposition = response.headers.get("content-disposition") ?? "";
  const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/)?.[1];
  const filename = encoded ? decodeURIComponent(encoded) : "借還車AI助理.xlsx";
  const url = URL.createObjectURL(await response.blob());
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function getHistoryEventDetail(eventId: string, signal?: AbortSignal): Promise<HistoryEventDetail> {
  return parseResponse<HistoryEventDetail>(await fetch(
    `${API_BASE}/history/events/${encodeURIComponent(eventId)}/detail`, { signal },
  ));
}

export interface HistoryFeedbackInput {
  question: string;
  answer_id?: string | null;
  answer: string;
  rating: 1 | -1;
  comment?: string;
  expected?: string;
  context?: HistoryChatContext | null;
  intent?: HistoryIntent | null;
  generated_by?: string | null;
  warnings?: string[];
  cited_event_ids?: string[];
}

export async function sendHistoryFeedback(input: HistoryFeedbackInput): Promise<void> {
  await parseResponse<unknown>(await fetch(`${API_BASE}/history/feedback`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input),
  }));
}

export interface HistoryUserSuggestion {
  user_id: string;
  display_name: string;
}

/** A text model for the assistant's summary; endpoints and keys stay on the server. */
export interface HistoryModelOption {
  id: string;
  label: string;
  provider: "ollama" | "openai_compatible";
  tier: "local" | "paid" | "gpu";
  description: string;
  available: boolean;
}

export interface HistoryModelList {
  default_id: string;
  options: HistoryModelOption[];
}

/** The composer's 「Model」 picker; the default (current local model) is listed first. */
export async function getHistoryModels(): Promise<HistoryModelList> {
  return parseResponse<HistoryModelList>(await fetch(`${API_BASE}/history/models`, { cache: "no-store" }));
}

/** The composer's + menu: questions that answer on the current data, with real names and plates. */
export async function getHistoryExamples(): Promise<string[]> {
  return parseResponse<string[]>(await fetch(`${API_BASE}/history/examples`, { cache: "no-store" }));
}

export async function getHistoryUserSuggestions(search: string, signal?: AbortSignal): Promise<HistoryUserSuggestion[]> {
  return parseResponse<HistoryUserSuggestion[]>(await fetch(
    `${API_BASE}/history/user-suggestions?search=${encodeURIComponent(search)}&limit=8`, { signal },
  ));
}
