import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  addLearningSampleToTrainingSet,
  cancelLearningSampleApproval,
  deleteLearningRental,
  deleteLearningSample,
  getLearningSamples,
  getLearningStats,
  learningImageUrl,
  type LearningSample,
  type LearningStats,
  type LearningTask,
  type LearningRegionAnnotation,
  type LearningVehicleIdentity,
} from "../lib/api";
import { ZoomableImage, type ImageHighlight } from "./ImageViewer";
import { PhotoListControls, usePhotoListPager } from "./PhotoListPager";
import { ManualTruthUpload, type ManualTruthTaskOption } from "./ManualTruthUpload";
import { RegionAnnotationEditor, regionAnnotationLabel } from "./RegionAnnotationEditor";
import { learningModelHighlights } from "../lib/learningModelHighlights";
import { VehicleIdentityEditor } from "./VehicleIdentityEditor";
import { RegionAnnotationOverlay } from "./RegionAnnotationOverlay";
import { evidenceRegions } from "../lib/regionGeometry";
import { seatDescription, seatMaterials, seatPatterns, seatStyles } from "../lib/vehicleCatalog";

const taskOrder: LearningTask[] = [
  "exterior_view",
  "vehicle_coverage",
  "capture_sharpness",
  "interior_capture",
  "interior_cleanliness",
  "visible_damage",
  "damage_pair",
  "license_plate",
];

const taskDefinitions: Record<LearningTask, { label: string; model: string }> = {
  exterior_view: { label: "四角視角", model: "DINOv2 四角視角分類器" },
  vehicle_coverage: { label: "車身構圖", model: "車身構圖人工真值" },
  interior_capture: { label: "車內拍攝有效性", model: "車內拍攝有效性人工真值" },
  interior_cleanliness: { label: "車內清潔", model: "車內清潔人工真值" },
  visible_damage: { label: "單張車損特寫", model: "單張車損人工真值" },
  damage_pair: { label: "前後車損比對", model: "DINOv2 車身區塊車損比對模型" },
  license_plate: { label: "車牌辨識", model: "RapidOCR 預訓練模型人工確認紀錄" },
  capture_sharpness: { label: "車外照片清晰度", model: "手晃模糊偵測模型" },
};

const lightingLabels: Record<string, string> = {
  daytime: "日間",
  low_light: "低光",
  night: "夜間",
  overexposed: "過曝",
};

const classLabels: Record<string, string> = {
  left_front: "左前",
  right_front: "右前",
  left_rear: "左後",
  right_rear: "右後",
  acceptable: "構圖合格",
  too_far: "車輛太遠",
  too_close: "車輛太近",
  not_vehicle: "不是車輛",
  unusable: "無法使用",
  sharp: "清楚",
  valid_front: "合格前座車內照",
  valid_rear: "合格後座車內照",
  wrong_scene: "不是指定車內場景",
  blurred: "模糊／鏡頭遮擋",
  rotated: "旋轉／方向錯誤",
  unusable_other: "其他無法使用",
  clean: "乾淨",
  ordinary: "普通",
  dirty: "髒污",
  damage: "有車損",
  no_damage: "無車損",
  confirmed_new_damage: "確認新增車損",
  no_new_damage: "沒有新增車損",
  cannot_assess: "無法判斷",
  no_change: "沒有新增變化",
  review_required: "需要人工覆核",
  incompatible: "照片無法比對",
  interior_front: "前座車內",
  interior_rear: "後座車內",
};

const triggerReasonLabels: Record<string, string> = {
  retake_required: "需要重拍",
  quality_failed: "照片品質不合格",
  quality_warning: "照片品質需注意",
  interior_scene_low_confidence: "車內場景判斷信心不足",
  interior_scene_mismatch: "車內場景與指定位置不符",
  interior_rotation_veto: "車內照片方向不符",
  interior_reference_candidate: "車內拍攝參考候選",
  view_low_confidence: "拍攝視角判斷信心不足",
  view_mismatch: "拍攝視角不符",
  high_quality_reference_candidate: "高品質拍攝參考候選",
  coverage_low_confidence: "車身構圖判斷信心不足",
  coverage_pass: "車身構圖合格",
  coverage_warning: "車身構圖需注意",
  coverage_fail: "車身構圖不合格",
  coverage_pending_model: "車身構圖模型待判斷",
  confidence_below_90: "模型信心低於 90%",
  human_review_required: "需要人工覆核",
  image_unusable: "照片無法判讀",
  not_vehicle_exterior: "不是車輛外觀照片",
  single_image_damage_claim: "單張照片疑似有車損",
  pair_incompatible_retake: "取還車照片無法比對，需重拍",
  model_requested_review: "模型要求人工覆核",
  model_no_change_candidate: "模型判斷沒有新增變化",
  plate_not_detected: "未偵測到車牌",
  plate_text_unreadable: "車牌文字無法辨識",
  plate_low_confidence: "車牌辨識信心不足",
  plate_mismatch: "車牌與租單不符",
  plate_timeout: "車牌辨識超過時間上限，模型不確定",
  plate_model_uncertain: "模型不確定，請人工檢查",
  failed_after_training: "訓練後再次出現問題",
  specialist_mining_candidate: "車損比對模型挑出的可疑配對，請框選確認",
  claim_pair_candidate: "理賠車輛的取還車配對，請確認這個角度是否看得到新增車損",
  manual_upload: "後台直接上傳的樣本",
  blur_model_blocked: "手晃模糊模型要求重拍",
  blur_model_uncertain: "手晃模糊模型不確定，請確認清不清楚",
};

const trainableLabels: Record<LearningTask, string[]> = {
  exterior_view: ["left_front", "right_front", "left_rear", "right_rear"],
  vehicle_coverage: ["acceptable", "too_far", "too_close", "not_vehicle", "unusable"],
  interior_capture: ["valid_front", "valid_rear", "wrong_scene", "blurred", "rotated", "unusable_other"],
  interior_cleanliness: ["clean", "ordinary", "dirty"],
  visible_damage: ["damage", "no_damage"],
  damage_pair: ["confirmed_new_damage", "no_new_damage"],
  license_plate: [],
  capture_sharpness: ["sharp", "blurred"],
};

const regionTruthTasks = new Set<LearningTask>([
  "interior_cleanliness",
  "visible_damage",
  "damage_pair",
]);

// Plate truths only confirm OCR readings, so they are not uploaded directly.
const manualTruthOptions: ManualTruthTaskOption[] = taskOrder
  .filter((task) => trainableLabels[task].length > 0)
  .map((task) => ({
    task,
    label: taskDefinitions[task].label,
    labels: trainableLabels[task].map((value) => ({ value, text: classLabels[value] ?? value })),
    regionReview: regionTruthTasks.has(task),
  }));

function regionTruthLabel(
  sample: LearningSample,
  annotations: LearningRegionAnnotation[],
): string {
  annotations = evidenceRegions(annotations);
  if (annotations.length === 0) return "";
  if (sample.task === "interior_cleanliness") {
    return annotations.some((item) => item.kind === "issue") ? "dirty" : "clean";
  }
  if (sample.task === "visible_damage") {
    return annotations.some((item) => item.kind === "issue") ? "damage" : "no_damage";
  }
  if (sample.task === "damage_pair") {
    return annotations.some((item) => item.image_slot === "secondary" && item.kind === "issue")
      ? "confirmed_new_damage"
      : "no_new_damage";
  }
  return "";
}

function regionImageSlotsFor(sample: LearningSample): Array<"primary" | "secondary"> {
  return sample.has_secondary_image ? ["primary", "secondary"] : ["primary"];
}

type PendingReviewGroup = {
  key: string;
  rentalId: string | null;
  vehicleId: string | null;
  samples: LearningSample[];
};

function groupPendingSamples(samples: LearningSample[]): PendingReviewGroup[] {
  const groups = new Map<string, PendingReviewGroup>();
  for (const sample of samples) {
    const rentalId = sample.rental_id?.trim() || null;
    const vehicleId = sample.vehicle_id?.trim() || null;
    const key = pendingGroupKey(sample);
    let group = groups.get(key);
    if (!group) {
      group = { key, rentalId, vehicleId, samples: [] };
      groups.set(key, group);
    }
    group.samples.push(sample);
  }
  return [...groups.values()];
}

type PendingReviewItem = { key: string; samples: LearningSample[]; slot: string | null };

function groupMatchingViews(samples: LearningSample[]): PendingReviewItem[] {
  const used = new Set<string>();
  const items: PendingReviewItem[] = [];
  for (const sample of samples) {
    if (used.has(sample.id)) continue;
    used.add(sample.id);
    const otherStage = sample.inspection_stage === "pickup" ? "return" : "pickup";
    const match = sample.inspection_slot && sample.inspection_stage && !sample.has_secondary_image
      ? samples.find((candidate) => !used.has(candidate.id)
          && candidate.task === sample.task
          && candidate.inspection_slot === sample.inspection_slot
          && candidate.inspection_stage === otherStage
          && !candidate.has_secondary_image)
      : undefined;
    if (match) used.add(match.id);
    const paired = match ? [sample, match].sort((a, b) => a.inspection_stage === "pickup" ? -1 : b.inspection_stage === "pickup" ? 1 : 0) : [sample];
    items.push({ key: paired.map((entry) => entry.id).join(":"), samples: paired, slot: sample.inspection_slot });
  }
  return items;
}

const plateSlotOrder = ["left_front", "right_front", "left_rear", "right_rear"];

function plateKey(value: string | null | undefined): string {
  return (value ?? "").toUpperCase().replace(/[^A-Z0-9]/g, "");
}

// Every angle of one rental shows the same plate, so they are confirmed together.
function plateBatchSamples(samples: LearningSample[]): LearningSample[] {
  const plates = samples.filter((sample) => sample.task === "license_plate");
  if (plates.length < 2) return [];
  const slotRank = (sample: LearningSample) => {
    const rank = plateSlotOrder.indexOf(sample.inspection_slot ?? "");
    return rank < 0 ? plateSlotOrder.length : rank;
  };
  const stageRank = (sample: LearningSample) =>
    sample.inspection_stage === "pickup" ? 0 : sample.inspection_stage === "return" ? 1 : 2;
  return [...plates].sort((a, b) => slotRank(a) - slotRank(b) || stageRank(a) - stageRank(b));
}

function commonPlateReading(plates: LearningSample[], fallback: string | null): string {
  const counts = new Map<string, { text: string; count: number }>();
  for (const sample of plates) {
    const text = (sample.corrected_label ?? sample.predicted_label ?? "").trim().toUpperCase();
    if (!plateKey(text)) continue;
    const entry = counts.get(plateKey(text)) ?? { text, count: 0 };
    entry.count += 1;
    counts.set(plateKey(text), entry);
  }
  const ranked = [...counts.values()].sort((a, b) => b.count - a.count
    || Number(plateKey(b.text) === plateKey(fallback)) - Number(plateKey(a.text) === plateKey(fallback)));
  return ranked[0]?.text ?? fallback?.trim().toUpperCase() ?? "";
}

function reviewReason(sample: LearningSample): string {
  const priority = [
    "plate_timeout", "plate_mismatch", "pair_incompatible_retake",
    "model_requested_review", "retake_required", "view_mismatch",
  ];
  const reason = priority.find((value) => sample.trigger_reasons.includes(value))
    ?? sample.trigger_reasons[0];
  return reason ? triggerReasonText(reason, sample) : "請人工確認";
}

function stageLabel(sample: LearningSample): string {
  if (sample.has_secondary_image) return "取車／還車雙圖";
  if (sample.inspection_stage === "pickup") return "取車照";
  if (sample.inspection_stage === "return") return "還車照";
  return "待覆核照片";
}

function labelText(value: string | null, task?: LearningTask): string {
  if (!value) return "—";
  if (task === "license_plate") return value;
  return classLabels[value] ?? "未識別分類";
}

function triggerReasonText(value: string, sample?: LearningSample): string {
  const score = /^score_below_(\d+)$/.exec(value);
  if (score) return `照片品質分數低於 ${score[1]}`;
  if (value === "plate_timeout") {
    const latencyMs = sample?.metadata.latency_ms;
    if (typeof latencyMs === "number" && Number.isFinite(latencyMs) && latencyMs >= 1000) {
      return `辨識達 ${Math.round(latencyMs / 1000)} 秒上限，模型不確定`;
    }
  }
  return triggerReasonLabels[value] ?? "其他待覆核原因";
}

function reviewTime(value: string | null): string {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-TW", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function rentalTime(value: string): string {
  return new Intl.DateTimeFormat("zh-TW", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function automaticDatasetLabel(sample: LearningSample): string {
  const value = sample.corrected_label
    ?? (sample.task === "exterior_view"
      ? sample.expected_label ?? sample.predicted_label
      : sample.predicted_label ?? sample.expected_label);
  return value ? labelText(value, sample.task) : "目前無可用分類";
}

const vehicleTypeLabels: Record<NonNullable<LearningVehicleIdentity["vehicle_type"]>, string> = {
  passenger_car: "小客車",
  suv: "休旅車",
  van: "箱型車",
};

function savedVehicleIdentity(sample: LearningSample): string | null {
  const review = sample.metadata.human_review;
  if (!review || typeof review !== "object") return null;
  const identity = (review as Record<string, unknown>).vehicle_identity;
  if (!identity || typeof identity !== "object") return null;
  const values = identity as Record<string, unknown>;
  const parts: string[] = [];
  if (typeof values.make === "string" && values.make.trim()) {
    parts.push(`品牌：${values.make.trim()}`);
  }
  if (typeof values.model === "string" && values.model.trim()) parts.push(`車款：${values.model.trim()}`);
  if (typeof values.generation === "string" && values.generation.trim()) parts.push(`版本：${values.generation.trim()}`);
  for (const [key, title, options] of [
    ["seat_material", "座椅材質", seatMaterials], ["seat_style", "座椅款式", seatStyles], ["seat_pattern", "座椅紋路", seatPatterns],
  ] as const) {
    if (typeof values[key] === "string" && values[key]) parts.push(`${title}：${seatDescription(values[key], options)}`);
  }
  if (typeof values.vehicle_type === "string" && values.vehicle_type in vehicleTypeLabels) {
    parts.push(`車型：${vehicleTypeLabels[values.vehicle_type as keyof typeof vehicleTypeLabels]}`);
  }
  return parts.length ? parts.join(" · ") : null;
}

function savedPairAngleResult(sample: LearningSample): string | null {
  if (sample.task !== "damage_pair") return null;
  const review = sample.metadata.human_review;
  if (!review || typeof review !== "object") return null;
  const result = (review as Record<string, unknown>).pair_angle_result;
  if (result === "same_view") return "角度：同車同視角";
  if (result === "incompatible") return "角度：無法比對";
  return null;
}

type PreviousReview = {
  corrected_label: string | null;
  reviewer_notes: string;
  status: string;
  annotations: LearningRegionAnnotation[];
  review_method: string | null;
  pair_angle_result: "same_view" | "incompatible" | null;
  vehicle_identity: LearningVehicleIdentity | null;
};

/** The truth saved before a sample was reopened, so its form starts from it. */
function previousReview(sample: LearningSample): PreviousReview | null {
  const review = sample.metadata.previous_human_review;
  if (!review || typeof review !== "object") return null;
  const values = review as Record<string, unknown>;
  const identity = values.vehicle_identity;
  return {
    corrected_label: typeof values.corrected_label === "string" ? values.corrected_label : null,
    reviewer_notes: typeof values.reviewer_notes === "string" ? values.reviewer_notes : "",
    status: typeof values.status === "string" ? values.status : "approved",
    annotations: savedRegionAnnotations({ ...sample, metadata: { human_review: values } }),
    review_method: typeof values.review_method === "string" ? values.review_method : null,
    pair_angle_result: values.pair_angle_result === "same_view" || values.pair_angle_result === "incompatible"
      ? values.pair_angle_result : null,
    vehicle_identity: identity && typeof identity === "object" ? identity as LearningVehicleIdentity : null,
  };
}

const MANUAL_UPLOAD_GROUP_KEY = "manual-upload";

function pendingGroupKey(sample: LearningSample): string {
  const rentalId = sample.rental_id?.trim() || null;
  const vehicleId = sample.vehicle_id?.trim() || null;
  if (rentalId && vehicleId) return JSON.stringify([rentalId, vehicleId]);
  // Back-office uploads are not rentals, so they share one pending group.
  if (sample.trigger_reasons.includes("manual_upload")) return MANUAL_UPLOAD_GROUP_KEY;
  return `sample:${sample.id}`;
}

function pendingGroupTitle(group: PendingReviewGroup): string {
  if (group.key === MANUAL_UPLOAD_GROUP_KEY) return "後台上傳・待標註照片";
  return `${group.rentalId ?? "未提供租單編號"} · 車牌 ${group.vehicleId ?? "未提供"}`;
}

function savedRegionAnnotations(sample: LearningSample): LearningRegionAnnotation[] {
  const review = sample.metadata.human_review;
  if (!review || typeof review !== "object") return [];
  const annotations = (review as Record<string, unknown>).annotations;
  if (!Array.isArray(annotations)) return [];
  return annotations.filter((item): item is LearningRegionAnnotation => {
    if (!item || typeof item !== "object") return false;
    const box = item as Partial<LearningRegionAnnotation>;
    return (box.image_slot === "primary" || box.image_slot === "secondary")
      && (box.kind === "issue" || box.kind === "hard_negative" || box.kind === "reference")
      && typeof box.label === "string"
      && [box.x, box.y, box.width, box.height].every((value) =>
        typeof value === "number" && Number.isFinite(value))
      && box.x! >= 0 && box.y! >= 0 && box.width! > 0 && box.height! > 0
      && box.x! + box.width! <= 1.000001 && box.y! + box.height! <= 1.000001
      && (box.shape !== "polygon" || (Array.isArray(box.points) && box.points.length >= 3));
  });
}

function savedImageHighlights(
  annotations: LearningRegionAnnotation[],
  slot: "primary" | "secondary",
  task: LearningTask,
): ImageHighlight[] {
  return annotations.filter((annotation) => annotation.image_slot === slot).map((annotation) => ({
    x: annotation.x,
    y: annotation.y,
    width: annotation.width,
    height: annotation.height,
    kind: annotation.kind,
    points: annotation.shape === "polygon" ? annotation.points : undefined,
    label: `${annotation.kind === "issue"
      ? task === "interior_cleanliness" ? "髒污" : "車損"
      : annotation.kind === "reference" ? (annotation.normal_reference ? "正常外觀參考" : "部位輪廓")
      : task === "interior_cleanliness" ? "正常基準" : "無車損基準"}：${regionAnnotationLabel(annotation.label)}`,
  }));
}

function TrainingDatasetGroup({
  task,
  samples,
  busy,
  onClear,
}: {
  task: LearningTask;
  samples: LearningSample[];
  busy: string | null;
  onClear: (sample: LearningSample) => Promise<void>;
}) {
  const pager = usePhotoListPager(samples, 8, true);
  const definition = taskDefinitions[task];
  const trainedCount = samples.filter((sample) => sample.status === "trained").length;

  return <section className="training-dataset-group">
    <div className="training-dataset-heading">
      <div>
        <span className="sample-kind">{definition.label}</span>
        <strong>{definition.model}</strong>
        <small>{samples.length} 筆資料 · {trainedCount} 筆已完成訓練</small>
      </div>
    </div>
    <div className="approved-sample-grid">
      {pager.visibleItems.map((sample) => <article className="approved-sample-card" key={sample.id} title={`樣本 ${sample.id}`}>
        <div className={`approved-sample-images approved-sample-images--${sample.has_secondary_image ? "pair" : "single"}`}>
          <ZoomableImage
            src={learningImageUrl(sample.id, "primary")}
            alt={`${definition.label}主要樣本`}
            title={`${sample.vehicle_id ?? sample.id} 主要樣本`}
          />
          {sample.has_secondary_image && <ZoomableImage
            src={learningImageUrl(sample.id, "secondary")}
            alt={`${definition.label}對比樣本`}
            title={`${sample.vehicle_id ?? sample.id} 對比樣本`}
          />}
        </div>
        <div className="approved-sample-card__details">
          <span className={`dataset-status dataset-status--${sample.status}`}>
            {sample.status === "trained" ? "已完成訓練" : "等待訓練"}
          </span>
          <strong>{automaticDatasetLabel(sample)}</strong>
          {savedPairAngleResult(sample) && <small className="saved-pair-angle">{savedPairAngleResult(sample)}</small>}
          {sample.vehicle_id && <small>車輛：{sample.vehicle_id}</small>}
          {sample.score !== null && <small>品質：{sample.score} 分</small>}
          <button
            type="button"
            disabled={busy !== null}
            title={sample.status === "trained" ? "清除後不再用於下次重訓；目前模型需經下次訓練才會完全移除影響。" : undefined}
            onClick={() => void onClear(sample)}
          >{busy === `clear-${sample.id}` ? "清除中…" : "清除這筆資料"}</button>
        </div>
      </article>)}
    </div>
    <PhotoListControls
      currentPage={pager.currentPage}
      expanded={pager.expanded}
      itemLabel={`${definition.label}資料`}
      onExpandedChange={pager.setExpanded}
      onPageChange={pager.setPage}
      totalItems={samples.length}
      totalPages={pager.totalPages}
      visibleEnd={pager.visibleEnd}
      visibleStart={pager.visibleStart}
    />
  </section>;
}

export function LearningInbox({
  refreshKey = 0,
  onChanged,
}: {
  refreshKey?: number;
  onChanged?: () => Promise<void> | void;
}) {
  const [samples, setSamples] = useState<LearningSample[]>([]);
  const [approvedSamples, setApprovedSamples] = useState<LearningSample[]>([]);
  const [trainedSamples, setTrainedSamples] = useState<LearningSample[]>([]);
  const [stats, setStats] = useState<LearningStats | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showTrainingDatasets, setShowTrainingDatasets] = useState(false);
  const [showRentalDelete, setShowRentalDelete] = useState(false);
  const [selectedRentalKey, setSelectedRentalKey] = useState<string | null>(null);
  const [rentalDeleteError, setRentalDeleteError] = useState<string | null>(null);
  const [reviewLabels, setReviewLabels] = useState<Record<string, string>>({});
  const [editingPlateId, setEditingPlateId] = useState<string | null>(null);
  const [reviewerNotes, setReviewerNotes] = useState<Record<string, string>>({});
  const [regionAnnotations, setRegionAnnotations] = useState<Record<string, LearningRegionAnnotation[]>>({});
  const [regionTarget, setRegionTarget] = useState<{ sample: LearningSample; imageSlot: "primary" | "secondary" } | null>(null);
  const [vehicleIdentities, setVehicleIdentities] = useState<Record<string, LearningVehicleIdentity>>({});
  const [regionDraftActive, setRegionDraftActive] = useState(false);
  const [pairReviewMethods, setPairReviewMethods] = useState<Record<string, "region_annotation" | "manual_pair">>({});
  const [singleReviewMethods, setSingleReviewMethods] = useState<Record<string, "region_annotation" | "manual_single">>({});
  const [openPendingGroupKey, setOpenPendingGroupKey] = useState<string | null | undefined>(undefined);
  const [openPendingSampleId, setOpenPendingSampleId] = useState<string | null>(null);
  const [pairAngleResults, setPairAngleResults] = useState<Record<string, "same_view" | "incompatible">>({});
  const [pairNoDamageConfirmed, setPairNoDamageConfirmed] = useState<Record<string, boolean>>({});
  const [plateBatchLabels, setPlateBatchLabels] = useState<Record<string, string>>({});
  const [plateBatchExcluded, setPlateBatchExcluded] = useState<Record<string, boolean>>({});
  const seededSampleIds = useRef(new Set<string>());

  const refresh = useCallback(async () => {
    try {
      const [nextSamples, nextApproved, nextTrained, nextStats] = await Promise.all([
        getLearningSamples(),
        getLearningSamples("approved"),
        getLearningSamples("trained"),
        getLearningStats(),
      ]);
      setSamples(nextSamples);
      setApprovedSamples(nextApproved);
      setTrainedSamples(nextTrained);
      setStats(nextStats);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "讀取訓練資料失敗");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, 15000);
    return () => window.clearInterval(timer);
  }, [refresh, refreshKey]);

  useEffect(() => {
    if (!showTrainingDatasets) return;
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") setShowTrainingDatasets(false);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [showTrainingDatasets]);

  useEffect(() => {
    if (!showRentalDelete) return;
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && busy !== "delete-rental") setShowRentalDelete(false);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [showRentalDelete, busy]);

  useEffect(() => {
    if (!regionTarget) return;
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !regionDraftActive) setRegionTarget(null);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [regionTarget, regionDraftActive]);

  useEffect(() => {
    if (!message) return;
    const timer = window.setTimeout(() => setMessage(null), 1500);
    return () => window.clearTimeout(timer);
  }, [message]);

  // A reopened truth starts from what was saved last time instead of a blank form.
  useEffect(() => {
    for (const sample of samples) {
      const previous = previousReview(sample);
      if (!previous || seededSampleIds.current.has(sample.id)) continue;
      seededSampleIds.current.add(sample.id);
      const label = previous.corrected_label ?? "";
      const notes = previous.reviewer_notes === "人工確認訓練真值" ? "" : previous.reviewer_notes;
      if (notes) setReviewerNotes((current) => ({ ...current, [sample.id]: notes }));
      if (previous.vehicle_identity) {
        const identity = previous.vehicle_identity;
        setVehicleIdentities((current) => ({ ...current, [sample.id]: identity }));
      }
      if (previous.pair_angle_result) {
        const angle = previous.pair_angle_result;
        setPairAngleResults((current) => ({ ...current, [sample.id]: angle }));
      }
      if (previous.annotations.length) {
        setRegionAnnotations((current) => ({ ...current, [sample.id]: previous.annotations }));
      }
      if (label && previous.annotations.some((item) => item.kind === "reference")) {
        setReviewLabels((current) => ({ ...current, [sample.id]: label }));
      }
      if (sample.task === "damage_pair") {
        if (previous.review_method === "manual_pair" || !previous.annotations.length) {
          setPairReviewMethods((current) => ({ ...current, [sample.id]: "manual_pair" }));
          setPairNoDamageConfirmed((current) => ({ ...current, [sample.id]: label === "no_new_damage" }));
        }
      } else if (regionTruthTasks.has(sample.task)) {
        if (previous.review_method === "manual_single" || !previous.annotations.length) {
          setSingleReviewMethods((current) => ({ ...current, [sample.id]: "manual_single" }));
          setReviewLabels((current) => ({ ...current, [sample.id]: label }));
        }
      } else if (label) {
        setReviewLabels((current) => ({ ...current, [sample.id]: label }));
      }
    }
  }, [samples]);

  const groupedLighting = useMemo(() => {
    const pendingByLighting = samples.reduce<Record<string, number>>((counts, sample) => {
      counts[sample.lighting] = (counts[sample.lighting] ?? 0) + 1;
      return counts;
    }, {});
    return Object.entries(pendingByLighting).sort((a, b) => b[1] - a[1]);
  }, [samples]);

  const trainingSamples = useMemo(
    () => [...approvedSamples, ...trainedSamples],
    [approvedSamples, trainedSamples],
  );
  const trainableSamples = useMemo(
    () => trainingSamples.filter((sample) => sample.training_eligible),
    [trainingSamples],
  );
  const confirmedSamples = useMemo(
    () => trainingSamples
      .filter((sample) => sample.corrected_label !== null && sample.reviewed_at !== null)
      .sort((left, right) => Date.parse(right.reviewed_at!) - Date.parse(left.reviewed_at!)),
    [trainingSamples],
  );
  const datasetGroups = useMemo(
    () => taskOrder.map((task) => ({
      task,
      samples: trainableSamples.filter((sample) => sample.task === task),
    })),
    [trainableSamples],
  );
  const pendingGroups = useMemo(() => groupPendingSamples(samples), [samples]);
  const rentalDeleteOptions = useMemo(() => pendingGroups
    .filter((group) => group.rentalId && group.vehicleId)
    .map((group) => {
      const created = group.samples.map((sample) => sample.created_at)
        .sort((a, b) => Date.parse(a) - Date.parse(b));
      return {
        rental_id: group.rentalId!,
        vehicle_id: group.vehicleId!,
        sample_count: group.samples.length,
        oldest_created_at: created[0],
        newest_created_at: created[created.length - 1],
        sample_ids: group.samples.map((sample) => sample.id),
      };
    })
    .sort((a, b) => Date.parse(a.oldest_created_at) - Date.parse(b.oldest_created_at)), [pendingGroups]);
  const pendingPager = usePhotoListPager(pendingGroups, undefined, false, 10);
  const activePendingGroupKey = openPendingGroupKey === null
    ? null
    : pendingPager.visibleItems.some((group) => group.key === openPendingGroupKey)
      ? openPendingGroupKey : pendingPager.visibleItems[0]?.key;
  const confirmedTruthPager = usePhotoListPager(confirmedSamples);

  function reviewLabelFor(sample: LearningSample): string {
    if (sample.task === "damage_pair" && pairReviewMethods[sample.id] === "manual_pair") {
      if (pairAngleResults[sample.id] === "incompatible") return "cannot_assess";
      return pairAngleResults[sample.id] === "same_view" && pairNoDamageConfirmed[sample.id]
        ? "no_new_damage" : "";
    }
    if (
      (sample.task === "interior_cleanliness" || sample.task === "visible_damage")
      && singleReviewMethods[sample.id] === "manual_single"
    ) {
      return reviewLabels[sample.id] ?? "";
    }
    if (regionTruthTasks.has(sample.task)) {
      const annotations = regionAnnotations[sample.id] ?? [];
      return regionTruthLabel(sample, annotations)
        || (annotations.some((item) => item.kind === "reference") ? reviewLabels[sample.id] ?? "" : "");
    }
    if (reviewLabels[sample.id] !== undefined) return reviewLabels[sample.id];
    if (sample.corrected_label) return sample.corrected_label;
    if (sample.task === "exterior_view") {
      return sample.expected_label ?? "";
    }
    if (sample.task === "license_plate") {
      return sample.predicted_label ?? "";
    }
    return "";
  }

  async function addToTrainingSet(sample: LearningSample) {
    const correctedLabel = reviewLabelFor(sample);
    if (!correctedLabel) {
      setError(`請先為${taskDefinitions[sample.task].label}選擇人工真值。`);
      return;
    }
    if (vehicleIdentities[sample.id]?.model?.trim() && !vehicleIdentities[sample.id]?.make.trim()) {
      setError("記錄實際車款時請一併提供品牌。");
      return;
    }
    setBusy(`add-${sample.id}`);
    setMessage(null);
    setError(null);
    try {
      const added = await addLearningSampleToTrainingSet(sample.id, {
        corrected_label: correctedLabel,
        reviewer_notes: reviewerNotes[sample.id]?.trim() ?? "",
        annotations: (sample.task === "damage_pair" && pairReviewMethods[sample.id] === "manual_pair")
          || singleReviewMethods[sample.id] === "manual_single"
          ? (regionAnnotations[sample.id] ?? []).filter((item) => item.kind === "reference")
          : regionAnnotations[sample.id] ?? [],
        vehicle_identity: vehicleIdentities[sample.id] ?? null,
        review_method: sample.task === "damage_pair"
          ? pairReviewMethods[sample.id] ?? "region_annotation"
          : singleReviewMethods[sample.id] ?? (regionTruthTasks.has(sample.task) ? "region_annotation" : undefined),
        pair_angle_result: sample.task === "damage_pair"
          ? pairAngleResults[sample.id] || null : undefined,
      });
      setMessage(correctedLabel === "cannot_assess"
        ? "已記錄無法比對。"
        : sample.task === "license_plate"
          ? `已儲存車牌真值「${automaticDatasetLabel(added)}」。`
          : `已確認「${automaticDatasetLabel(added)}」並加入訓練集。`);
      setEditingPlateId(null);
      setReviewLabels((current) => {
        const next = { ...current };
        delete next[sample.id];
        return next;
      });
      setReviewerNotes((current) => {
        const next = { ...current };
        delete next[sample.id];
        return next;
      });
      setRegionAnnotations((current) => {
        const next = { ...current };
        delete next[sample.id];
        return next;
      });
      setVehicleIdentities((current) => {
        const next = { ...current };
        delete next[sample.id];
        return next;
      });
      setPairReviewMethods((current) => {
        const next = { ...current };
        delete next[sample.id];
        return next;
      });
      setSingleReviewMethods((current) => {
        const next = { ...current };
        delete next[sample.id];
        return next;
      });
      setPairAngleResults((current) => {
        const next = { ...current };
        delete next[sample.id];
        return next;
      });
      setPairNoDamageConfirmed((current) => {
        const next = { ...current };
        delete next[sample.id];
        return next;
      });
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "加入訓練集失敗");
    } finally {
      setBusy(null);
    }
  }

  function plateBatchLabelFor(group: PendingReviewGroup, plates: LearningSample[]): string {
    return plateBatchLabels[group.key] ?? commonPlateReading(plates, group.vehicleId);
  }

  async function confirmPlateBatch(group: PendingReviewGroup, plates: LearningSample[]) {
    const label = plateBatchLabelFor(group, plates).trim();
    const included = plates.filter((sample) => !plateBatchExcluded[sample.id]);
    if (!label || included.length === 0) return;
    if (included.some((sample) => plateKey(sample.predicted_label) !== plateKey(label)) && !window.confirm(
      `部分角度的模型辨識與「${label}」不同。確定以照片上的車牌為準，將這 ${included.length} 張全部確認為「${label}」嗎？`,
    )) return;
    setBusy(`plate-batch-${group.key}`);
    setMessage(null);
    setError(null);
    let saved = 0;
    try {
      for (const sample of included) {
        await addLearningSampleToTrainingSet(sample.id, {
          corrected_label: label,
          reviewer_notes: "同租單各角度車牌一鍵確認一致",
          annotations: [],
        });
        saved += 1;
      }
      setMessage(`已將 ${saved} 張車牌照片確認為「${label}」。`);
      setPlateBatchLabels((current) => {
        const next = { ...current };
        delete next[group.key];
        return next;
      });
    } catch (caught) {
      setError(`已儲存 ${saved} 張；其餘失敗：${caught instanceof Error ? caught.message : "儲存車牌真值失敗"}`);
    } finally {
      setBusy(null);
      await refresh();
    }
  }

  function renderPlateBatch(group: PendingReviewGroup, plates: LearningSample[]) {
    const label = plateBatchLabelFor(group, plates);
    const included = plates.filter((sample) => !plateBatchExcluded[sample.id]);
    const openPlate = plates.find((sample) => sample.id === openPendingSampleId);
    return <div className="learning-review-item learning-plate-batch">
      <div className="learning-review-pair-heading">
        <strong>車牌辨識 · 各角度一次確認</strong>
        <span>{plates.length} 張・取車 ↔ 還車</span>
      </div>
      <div className="learning-plate-strip">
        {plates.map((sample) => {
          const excluded = Boolean(plateBatchExcluded[sample.id]);
          const differs = plateKey(sample.predicted_label) !== plateKey(label);
          return <div
            className={`learning-plate-tile ${excluded ? "is-excluded" : ""} ${differs ? "is-different" : ""} ${openPendingSampleId === sample.id ? "is-open" : ""}`}
            key={sample.id}
          >
            <button
              type="button"
              className="learning-plate-tile-photo"
              aria-expanded={openPendingSampleId === sample.id}
              title="個別覆核這張"
              onClick={() => setOpenPendingSampleId(openPendingSampleId === sample.id ? null : sample.id)}
            >
              <img src={learningImageUrl(sample.id, "primary")} alt={`${labelText(sample.inspection_slot)} ${stageLabel(sample)}`} loading="lazy" />
            </button>
            <span className="learning-plate-tile-slot">{sample.inspection_slot ? labelText(sample.inspection_slot) : "未標角度"} · {stageLabel(sample)}</span>
            <strong>{sample.predicted_label ? labelText(sample.predicted_label, sample.task) : "未辨識"}</strong>
            <small>{sample.confidence != null && sample.confidence > 0 ? `信心 ${Math.round(sample.confidence * 100)}%` : "無信心分數"}{differs ? " · 不一致" : ""}</small>
            <label
              className="learning-plate-tile-include"
              title="勾選後，按下確認按鈕會將這張照片的車牌儲存為下方輸入值；取消勾選則保留待處理。"
            >
              <input
                type="checkbox"
                checked={!excluded}
                disabled={busy !== null}
                aria-label={`${labelText(sample.inspection_slot)} ${stageLabel(sample)}：將這張照片一起確認`}
                aria-describedby={`plate-batch-help-${group.key}`}
                onChange={(event) => setPlateBatchExcluded((current) => ({ ...current, [sample.id]: !event.target.checked }))}
              /><span>將這張一起確認</span>
            </label>
          </div>;
        })}
      </div>
      <div className="learning-plate-batch-actions">
        <label htmlFor={`plate-batch-${group.key}`}>
          <span>照片上的車牌</span>
          <input
            id={`plate-batch-${group.key}`}
            value={label}
            maxLength={10}
            placeholder="請輸入車牌"
            onChange={(event) => setPlateBatchLabels((current) => ({
              ...current, [group.key]: event.target.value.toUpperCase(),
            }))}
          />
        </label>
        <button
          className="learning-plate-batch-confirm"
          type="button"
          aria-describedby={`plate-batch-help-${group.key}`}
          aria-busy={busy === `plate-batch-${group.key}`}
          disabled={busy !== null || !label.trim() || included.length === 0}
          onClick={() => void confirmPlateBatch(group, plates)}
        >
          <span aria-hidden="true">✓</span>
          {busy === `plate-batch-${group.key}` ? "正在儲存車牌…" : `確認已勾選的 ${included.length} 張車牌`}
        </button>
        <small id={`plate-batch-help-${group.key}`}>
          已勾選 {included.length}／{plates.length} 張。按下確認後，會將勾選照片的人工車牌儲存為「{label.trim() || "下方輸入的車牌"}」。
          未勾選的照片保留待處理；車牌不同時請取消勾選，點照片個別覆核。
        </small>
      </div>
      {openPlate && <div className="learning-review-item-grid">{renderPendingCard(openPlate)}</div>}
    </div>;
  }

  async function clearSample(sample: LearningSample) {
    const trainedWarning = sample.status === "trained"
      ? "\n\n這筆資料已用於目前模型；清除後會停止用於下次重訓，但目前模型須在下次訓練後才會完全移除影響。"
      : "";
    if (!window.confirm(
      `確定清除這筆${taskDefinitions[sample.task].label}資料嗎？\n\n將永久刪除資料、標註及主動學習資料夾中的來源照片；其他資料集與租還車私有原圖不受影響。此操作無法復原。${trainedWarning}`,
    )) return;
    setBusy(`clear-${sample.id}`);
    setMessage(null);
    setError(null);
    try {
      await deleteLearningSample(sample.id);
      setMessage("已刪除資料與來源照片。");
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "清除資料失敗");
    } finally {
      setBusy(null);
    }
  }

  function openRentalDelete() {
    setShowRentalDelete(true);
    setRentalDeleteError(null);
    setSelectedRentalKey(null);
  }

  async function clearRental() {
    const selected = rentalDeleteOptions.find((rental) =>
      JSON.stringify([rental.rental_id, rental.vehicle_id]) === selectedRentalKey);
    if (!selected) return;
    const description = `確定將租單 ${selected.rental_id}、車牌 ${selected.vehicle_id} 的 ${selected.sample_count} 筆待處理照片從此清單移除嗎？已標註人工真值、訓練資料、所有照片與租還車紀錄都會保留。`;
    if (!window.confirm(description)) return;
    setBusy("delete-rental");
    setRentalDeleteError(null);
    try {
      const deleted = await deleteLearningRental(selected, selected.sample_ids);
      setShowRentalDelete(false);
      setSelectedRentalKey(null);
      setMessage(`已從待處理清單移除 ${deleted.deleted_samples} 筆照片，人工真值與其他資料保留。`);
      await refresh();
      await onChanged?.();
    } catch (caught) {
      setRentalDeleteError(caught instanceof Error ? caught.message : "移除待處理租單失敗");
      setSelectedRentalKey(null);
      await refresh();
    } finally {
      setBusy(null);
    }
  }

  async function cancelApproval(sample: LearningSample) {
    const plate = sample.task === "license_plate";
    const trainedNote = sample.status === "trained"
      ? "\n\n這筆已用於訓練；修改後會重新計入待訓練，下一次訓練才會套用新的真值。"
      : "";
    if (!window.confirm(plate
      ? `確定刪除車牌真值「${labelText(sample.corrected_label, sample.task)}」嗎？資料會退回待處理；來源照片仍保留。`
      : `重新標註「${labelText(sample.corrected_label, sample.task)}」這筆人工真值？\n\n資料會退回待處理並自動載入上次的真值、框選與備註，修改後再儲存即可；照片與上次的標註都會保留。${trainedNote}`,
    )) return;
    setBusy(`cancel-${sample.id}`);
    setMessage(null);
    setError(null);
    try {
      const reopened = await cancelLearningSampleApproval(sample.id);
      if (reopened.status !== "pending_review" || reopened.corrected_label !== null) {
        throw new Error("取消回應未將樣本退回待處理，請重新整理確認狀態");
      }
      setApprovedSamples((current) => current.filter((item) => item.id !== sample.id));
      setSamples((current) => [reopened, ...current.filter((item) => item.id !== sample.id)]
        .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at)));
      setOpenPendingGroupKey(pendingGroupKey(reopened));
      setMessage(plate ? "已刪除車牌真值，樣本退回待處理。" : "已退回待處理，並載入上次的真值供編輯。");
      await refresh();
      await onChanged?.();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "取消人工真值失敗");
    } finally {
      setBusy(null);
    }
  }

  const trainableDatasets = (stats?.datasets ?? []).filter((dataset) => dataset.runner !== "none") ?? [];

  function renderPendingCard(sample: LearningSample) {
    return <article
              className={`learning-card learning-row-card ${openPendingSampleId === sample.id ? "is-open" : ""}`}
              key={sample.id}
            >
          <button
            className="learning-row"
            type="button"
            aria-expanded={openPendingSampleId === sample.id}
            onClick={() => setOpenPendingSampleId(openPendingSampleId === sample.id ? null : sample.id)}
          >
            <span className="learning-row-thumbs">
              {regionImageSlotsFor(sample).map((imageSlot) => <img
                key={imageSlot}
                src={learningImageUrl(sample.id, imageSlot)}
                alt=""
                loading="lazy"
              />)}
            </span>
            <span className="learning-row-main">
              <strong>{taskDefinitions[sample.task].label}{sample.inspection_slot ? ` · ${labelText(sample.inspection_slot)}` : ""}</strong>
              <small>{stageLabel(sample)} · {reviewReason(sample)}</small>
            </span>
            <span className="learning-row-facts">
              {sample.predicted_label && <span>模型 <strong>{labelText(sample.predicted_label, sample.task)}</strong></span>}
              {sample.confidence != null && sample.confidence > 0 && <span>信心 <strong>{Math.round(sample.confidence * 100)}%</strong></span>}
            </span>
            <span className="learning-row-toggle">{openPendingSampleId === sample.id ? "收合 ▾" : "覆核 ▸"}</span>
          </button>
          {openPendingSampleId === sample.id && <>
          {!regionTruthTasks.has(sample.task) && <div className={`learning-images learning-images--${sample.has_secondary_image ? "pair" : "single"}`}>
            <ZoomableImage
              src={learningImageUrl(sample.id, "primary")}
              alt={`${sample.id} 主要樣本`}
              title={`${sample.id} 主要樣本`}
            />
            {sample.has_secondary_image && <ZoomableImage
              src={learningImageUrl(sample.id, "secondary")}
              alt={`${sample.id} 對比樣本`}
              title={`${sample.id} 對比樣本`}
            />}
            <strong className="learning-photo-stage">{stageLabel(sample)}</strong>
            {sample.lighting !== "daytime" && <span>{lightingLabels[sample.lighting] ?? sample.lighting}</span>}
          </div>}
          <div className="learning-body">
            {previousReview(sample) && <p className="learning-previous-truth">
              重新標註中：已載入上次的真值「{labelText(previousReview(sample)!.corrected_label, sample.task)}」
              {previousReview(sample)!.status === "trained" ? "（曾用於訓練）" : ""}，修改後按儲存即可。
            </p>}
            {((sample.expected_label && sample.task !== "license_plate" && sample.expected_label !== sample.inspection_slot) || sample.score != null) && <div className="learning-review-facts">
              {sample.expected_label && sample.task !== "license_plate" && sample.expected_label !== sample.inspection_slot && <span>預期 <strong>{labelText(sample.expected_label, sample.task)}</strong></span>}
              {sample.score != null && <span>品質 <strong>{sample.score} 分</strong></span>}
            </div>}
            {!regionTruthTasks.has(sample.task) && <div className={`learning-eligibility ${sample.task === "license_plate" ? "learning-eligibility--plate" : ""}`}>
              <label className="learning-label" htmlFor={`truth-${sample.id}`}>
                <span>{sample.task === "license_plate" ? "確認車牌" : "人工真值"}</span>
                {sample.task === "license_plate"
                  ? <input
                    id={`truth-${sample.id}`}
                    value={reviewLabelFor(sample)}
                    maxLength={10}
                    placeholder="新增車牌編號，請輸入車牌"
                    onChange={(event) => setReviewLabels((current) => ({
                      ...current, [sample.id]: event.target.value.toUpperCase(),
                    }))}
                    aria-describedby={`truth-help-${sample.id}`}
                  />
                  : <select
                    id={`truth-${sample.id}`}
                    value={reviewLabelFor(sample)}
                    onChange={(event) => setReviewLabels((current) => ({
                      ...current,
                      [sample.id]: event.target.value,
                    }))}
                    aria-describedby={`truth-help-${sample.id}`}
                  >
                    <option value="">請人工選擇</option>
                    {trainableLabels[sample.task].map((label) => <option value={label} key={label}>
                      {labelText(label, sample.task)}
                    </option>)}
                  </select>}
              </label>
              <small id={`truth-help-${sample.id}`}>{sample.task === "license_plate"
                ? "請檢查模型是否辨識正確；以照片上的車牌為準。"
                : "模型建議不會自動成為真值；按下加入即代表登入中的營運人員完成確認。"}</small>
              {sample.task === "license_plate" && <div className="plate-truth-actions">
                {sample.predicted_label && <button type="button" disabled={busy !== null} onClick={() => setReviewLabels((current) => ({ ...current, [sample.id]: sample.predicted_label ?? "" }))}>採用模型辨識</button>}
                <button type="button" disabled={busy !== null} onClick={() => setReviewLabels((current) => ({ ...current, [sample.id]: "" }))}>清除輸入</button>
              </div>}
              <label className="learning-label" htmlFor={`review-notes-${sample.id}`}>
                <span>覆核備註</span>
                <input
                  id={`review-notes-${sample.id}`}
                  type="text"
                  maxLength={500}
                  placeholder="判斷依據（選填）"
                  value={reviewerNotes[sample.id] ?? ""}
                  onChange={(event) => setReviewerNotes((current) => ({
                    ...current,
                    [sample.id]: event.target.value,
                  }))}
                />
              </label>
            </div>}
            {(sample.task === "interior_cleanliness" || sample.task === "visible_damage" || sample.task === "damage_pair") && <div className="localized-learning-review">
              {(sample.task === "interior_cleanliness" || sample.task === "visible_damage") && !sample.has_secondary_image && <fieldset className="pair-truth-method">
                <legend>人工真值處理方式</legend>
                <label><input
                  type="radio"
                  name={`single-truth-method-${sample.id}`}
                  checked={singleReviewMethods[sample.id] !== "manual_single"}
                  onChange={() => setSingleReviewMethods((current) => ({ ...current, [sample.id]: "region_annotation" }))}
                />框選問題或正常區域</label>
                <label><input
                  type="radio"
                  name={`single-truth-method-${sample.id}`}
                  checked={singleReviewMethods[sample.id] === "manual_single"}
                  onChange={() => setSingleReviewMethods((current) => ({ ...current, [sample.id]: "manual_single" }))}
                />人工選擇真值</label>
                {singleReviewMethods[sample.id] === "manual_single" && <label className="single-manual-truth">
                  <span>人工確認結果</span>
                  <select
                    value={reviewLabels[sample.id] ?? ""}
                    onChange={(event) => setReviewLabels((current) => ({ ...current, [sample.id]: event.target.value }))}
                  >
                    <option value="">請選擇</option>
                    {trainableLabels[sample.task].map((label) => <option value={label} key={label}>
                      {labelText(label, sample.task)}
                    </option>)}
                  </select>
                </label>}
              </fieldset>}
              {sample.task === "damage_pair" && <fieldset className="pair-truth-method">
                <legend>人工真值處理方式</legend>
                <label><input
                  type="radio"
                  name={`pair-truth-method-${sample.id}`}
                  checked={pairReviewMethods[sample.id] !== "manual_pair"}
                  onChange={() => {
                    setPairReviewMethods((current) => ({ ...current, [sample.id]: "region_annotation" }));
                    if (pairAngleResults[sample.id] === "incompatible") {
                      setPairAngleResults((current) => {
                        const next = { ...current };
                        delete next[sample.id];
                        return next;
                      });
                    }
                  }}
                />框選區域建立車損真值</label>
                <label><input
                  type="radio"
                  name={`pair-truth-method-${sample.id}`}
                  checked={pairReviewMethods[sample.id] === "manual_pair"}
                  onChange={() => setPairReviewMethods((current) => ({ ...current, [sample.id]: "manual_pair" }))}
                />不框選，人工確認無新增車損或無法比對</label>
                <div className="pair-manual-truth">
                  <label htmlFor={`pair-angle-${sample.id}`}>① 核對取車與還車照片角度{pairReviewMethods[sample.id] !== "manual_pair" && "（選填）"}</label>
                  <select
                    id={`pair-angle-${sample.id}`}
                    value={pairAngleResults[sample.id] ?? ""}
                    onChange={(event) => {
                      const value = event.target.value;
                      setPairAngleResults((current) => {
                        const next = { ...current };
                        if (value === "same_view" || value === "incompatible") {
                          next[sample.id] = value;
                        } else {
                          delete next[sample.id];
                        }
                        return next;
                      });
                      setPairNoDamageConfirmed((current) => ({ ...current, [sample.id]: false }));
                    }}
                  >
                    <option value="">{pairReviewMethods[sample.id] === "manual_pair" ? "請選擇角度核對結果" : "尚未記錄角度"}</option>
                    <option value="same_view">同車同視角，符合指定角度</option>
                    {pairReviewMethods[sample.id] === "manual_pair" && <option value="incompatible">角度不符或無法比對</option>}
                  </select>
                  <small>流程指定角度：{labelText(sample.expected_label, sample.task)}。選擇角度後，按下方確認按鈕才會存入系統。框選模式可同時保存角度與區域。</small>
                  {pairReviewMethods[sample.id] === "manual_pair" && pairAngleResults[sample.id] === "same_view" && <label className="pair-no-damage-confirmation">
                    <input
                      type="checkbox"
                      checked={pairNoDamageConfirmed[sample.id] ?? false}
                      onChange={(event) => setPairNoDamageConfirmed((current) => ({
                        ...current,
                        [sample.id]: event.target.checked,
                      }))}
                    />② 我已核對兩張照片，確認沒有新增車損
                  </label>}
                  {pairReviewMethods[sample.id] === "manual_pair" && pairAngleResults[sample.id] === "incompatible" && <small>
                    確認後會記錄為「無法判斷」，不會加入車損訓練集。
                  </small>}
                </div>
              </fieldset>}
              <div className="region-truth-previews">
                {regionImageSlotsFor(sample).map((imageSlot) => {
                  const modelHighlights = learningModelHighlights(sample, imageSlot);
                  const imageTitle = imageSlot === "primary"
                    ? sample.has_secondary_image ? "取車／基準照" : "待覆核照片"
                    : "還車／對比照";
                  if ((sample.task === "damage_pair" && pairReviewMethods[sample.id] === "manual_pair")
                    || singleReviewMethods[sample.id] === "manual_single") {
                    return <ZoomableImage
                      key={imageSlot}
                      className="region-truth-preview region-truth-preview--manual"
                      imageClassName="region-truth-preview-manual-image"
                      src={learningImageUrl(sample.id, imageSlot)}
                      highlights={[...modelHighlights, ...savedImageHighlights(
                        (regionAnnotations[sample.id] ?? []).filter((item) => item.kind === "reference"), imageSlot, sample.task)]}
                      alt={`${sample.id} ${imageTitle}`}
                      title={`${sample.vehicle_id ?? sample.id} · ${imageTitle}`}
                      overlay={<span>⌕ 點擊檢視 · {imageTitle}</span>}
                    />;
                  }
                  const slotAnnotations = (regionAnnotations[sample.id] ?? []).filter((item) => item.image_slot === imageSlot);
                  return <button
                    className="region-truth-preview"
                    type="button"
                    key={imageSlot}
                    onClick={() => setRegionTarget({ sample, imageSlot })}
                    aria-label={`放大並框選${imageTitle}`}
                  >
                    <span className="region-truth-preview-image">
                      <span className="region-truth-preview-frame">
                        <img src={learningImageUrl(sample.id, imageSlot)} alt="" />
                        {modelHighlights.map((highlight, index) => <i
                          key={`model-${index}`}
                          className="region-truth-preview-box region-truth-preview-box--model"
                          title={highlight.label}
                          style={{ left: `${highlight.x * 100}%`, top: `${highlight.y * 100}%`,
                            width: `${highlight.width * 100}%`, height: `${highlight.height * 100}%` }}
                        />)}
                        <RegionAnnotationOverlay annotations={slotAnnotations} />
                      </span>
                    </span>
                    <span className="region-truth-preview-label">{imageTitle}{modelHighlights.length > 0 ? " · 黃色虛線為模型疑似區域" : " · 無模型框選，顯示原圖"}</span>
                  </button>;
                })}
              </div>
              {(regionAnnotations[sample.id] ?? []).some((item) => item.kind === "reference")
                && evidenceRegions(regionAnnotations[sample.id] ?? []).length === 0
                && pairReviewMethods[sample.id] !== "manual_pair" && singleReviewMethods[sample.id] !== "manual_single"
                && <label className="region-outline-truth">
                  <span>部位輪廓不代表正常；請另外確認整張照片的真值</span>
                  <select value={reviewLabels[sample.id] ?? ""}
                    onChange={(event) => setReviewLabels((current) => ({ ...current, [sample.id]: event.target.value }))}>
                    <option value="">請人工選擇</option>
                    {trainableLabels[sample.task].map((label) => <option key={label} value={label}>{labelText(label, sample.task)}</option>)}
                  </select>
                </label>}
              {regionTruthTasks.has(sample.task) && <p className="region-truth-derived-label">
                確認結果：{labelText(reviewLabelFor(sample), sample.task)}
                {!reviewLabelFor(sample) && (sample.task === "damage_pair" && pairReviewMethods[sample.id] === "manual_pair"
                  ? "（請核對角度；同角度時確認沒有新增車損）"
                  : singleReviewMethods[sample.id] === "manual_single"
                    ? "（請選擇人工真值）"
                  : "（請標記問題／正常細節；只有部位輪廓時請另外選擇真值）")}
              </p>}
              <label className="region-truth-notes" htmlFor={`review-notes-${sample.id}`}>
                <span>覆核備註</span>
                <input
                  id={`review-notes-${sample.id}`}
                  type="text"
                  maxLength={500}
                  placeholder="判斷依據（選填）"
                  value={reviewerNotes[sample.id] ?? ""}
                  onChange={(event) => setReviewerNotes((current) => ({
                    ...current,
                    [sample.id]: event.target.value,
                  }))}
                />
              </label>
              <VehicleIdentityEditor key={sample.id} value={vehicleIdentities[sample.id]}
                interior={sample.task === "interior_cleanliness"}
                onChange={(identity) => setVehicleIdentities((current) => ({ ...current, [sample.id]: identity }))} />
            </div>}
            <div className="card-actions">
              <button className="danger-outline" disabled={busy !== null} onClick={() => void clearSample(sample)}>
                {busy === `clear-${sample.id}` ? "清除中…" : "清除"}
              </button>
              <button
                className="primary"
                disabled={busy !== null || !reviewLabelFor(sample)}
                onClick={() => void addToTrainingSet(sample)}
              >
                {busy === `add-${sample.id}` ? "儲存中…" : sample.task === "license_plate"
                  ? "確認或修改後儲存" : sample.task === "damage_pair"
                  ? reviewLabelFor(sample) === "cannot_assess" ? "儲存角度：無法比對"
                    : pairAngleResults[sample.id] ? "儲存角度與車損真值" : "儲存車損真值"
                  : "確認真值並加入"}
              </button>
            </div>
          </div>
          </>}
        </article>;
  }

  return <section className="learning-inbox" id="learning-inbox">
    <div className="learning-heading">
      <div>
        <p className="eyebrow">MODEL TRAINING</p>
        <h2>訓練模型專區</h2>
        <p>模型分類只供參考。操作員必須逐筆確認人工真值，後端才會允許加入對應的獨立訓練集。</p>
      </div>
    </div>

    {stats && <div className="learning-stats">
      <div><span>待處理資料</span><strong>{stats.pending_review}</strong></div>
      <button
        className="learning-stat-action"
        type="button"
        onClick={() => setShowTrainingDatasets(true)}
        aria-haspopup="dialog"
      >
        <span>已加入訓練集</span>
        <strong>{trainableSamples.length}</strong>
        <small>依七種資料集查看照片、分類、狀態與清除</small>
      </button>
      <div>
        <span>等待訓練</span>
        <strong>{stats.approved_untrained ?? 0}</strong>
        <small>{trainableDatasets.filter((dataset) => dataset.approved_untrained > 0).length} 種資料集有新的人工真值</small>
      </div>
      <div><span>已完成訓練</span><strong>{stats.trained}</strong></div>
    </div>}

    {groupedLighting.length > 0 && <div className="learning-filters">
      {groupedLighting.map(([lighting, count]) => <span key={lighting}>
        {lightingLabels[lighting] ?? lighting} {count}
      </span>)}
    </div>}
    {message && <div className="learning-success-toast" role="status" aria-live="polite">
      <span aria-hidden="true">✓</span>
      <div><strong>操作完成</strong><p>{message}</p></div>
      <button type="button" aria-label="關閉成功通知" onClick={() => setMessage(null)}>×</button>
    </div>}
    {error && <div className="error-box"><strong>訓練資料操作失敗</strong><span>{error}</span></div>}

    <ManualTruthUpload
      options={manualTruthOptions}
      disabled={busy !== null}
      onUploaded={async (notice) => {
        setMessage(notice);
        await refresh();
        await onChanged?.();
      }}
    />

    <section className="recent-truths" aria-labelledby="recent-truths-title">
      <div className="recent-truths-heading">
        <div>
          <p className="eyebrow">HUMAN TRUTH AUDIT</p>
          <h3 id="recent-truths-title">已確認的人工真值</h3>
          <p>依確認時間排列；尚未訓練者可退回待處理，且不會刪除照片。</p>
        </div>
        <strong>{confirmedSamples.length} 筆</strong>
      </div>
      {confirmedSamples.length === 0
        ? <div className="recent-truths-empty">目前沒有已確認的人工真值</div>
        : <>
        <div className="recent-truths-grid">{confirmedTruthPager.visibleItems.map((sample) => {
          const annotations = savedRegionAnnotations(sample);
          return <article key={sample.id}>
          <div className="recent-truths-photos">
            {regionImageSlotsFor(sample).map((slot) => {
              const highlights = savedImageHighlights(annotations, slot, sample.task);
              const highlightLabels = [...new Set(highlights.map((highlight) => highlight.label.trim()).filter(Boolean))];
              const photoSummary = [
                ...(sample.has_secondary_image ? [slot === "primary" ? "取車／基準照" : "還車／對比照"] : []),
                ...highlightLabels,
              ].join(" · ");
              const photoLabel = sample.has_secondary_image
                ? slot === "primary" ? "取車／基準照" : "還車／對比照"
                : "已確認照片";
              return <div className="recent-truths-photo" key={slot}>
                <ZoomableImage
                  src={learningImageUrl(sample.id, slot)}
                  alt={`${photoLabel}，${taskDefinitions[sample.task].label}已確認樣本`}
                  title={`${sample.vehicle_id ?? sample.id} · ${photoLabel}`}
                  highlights={highlights}
                />
                {photoSummary && <span className="recent-truths-photo-label">{photoSummary}</span>}
              </div>;
            })}
          </div>
          <div>
            <span>{taskDefinitions[sample.task].label}{sample.inspection_slot ? ` · ${labelText(sample.inspection_slot)}` : ""}{sample.has_secondary_image ? " · 雙圖" : ""}</span>
            <strong>{labelText(sample.corrected_label, sample.task)}</strong>
            {sample.task === "license_plate" && <>
              {sample.predicted_label?.trim() && <small>模型辨識：{sample.predicted_label}</small>}
              {editingPlateId === sample.id && <>
                <label htmlFor={`edit-plate-${sample.id}`}>修改車牌編號</label>
                <input
                  id={`edit-plate-${sample.id}`}
                  value={reviewLabelFor(sample)}
                  maxLength={10}
                  onChange={(event) => setReviewLabels((current) => ({
                    ...current, [sample.id]: event.target.value.toUpperCase(),
                  }))}
                />
                <small>請檢查模型是否辨識正確，並核對原始照片。</small>
                <button type="button" disabled={busy !== null || !reviewLabelFor(sample)} onClick={() => void addToTrainingSet(sample)}>儲存修改</button>
              </>}
              {editingPlateId !== sample.id && sample.status !== "trained" && <button
                type="button" disabled={busy !== null}
                onClick={() => { setReviewLabels((current) => ({ ...current, [sample.id]: sample.corrected_label ?? "" })); setEditingPlateId(sample.id); }}
              >修改車牌編號</button>}
            </>}
            {savedPairAngleResult(sample) && <small className="saved-pair-angle">{savedPairAngleResult(sample)}</small>}
            {sample.task !== "license_plate" && sample.predicted_label?.trim() && <small>模型建議：{labelText(sample.predicted_label, sample.task)}</small>}
            {savedVehicleIdentity(sample) && <small>{savedVehicleIdentity(sample)}</small>}
            <time dateTime={sample.reviewed_at ?? undefined}>{reviewTime(sample.reviewed_at)}</time>
            <button
              type="button"
              disabled={busy !== null}
              onClick={() => void cancelApproval(sample)}
            >{busy === `cancel-${sample.id}` ? "處理中…" : sample.task === "license_plate" ? "刪除車牌真值" : "重新標註"}</button>
          </div>
        </article>})}</div>
        <PhotoListControls
          currentPage={confirmedTruthPager.currentPage}
          expanded={confirmedTruthPager.expanded}
          itemLabel="已確認真值"
          onExpandedChange={confirmedTruthPager.setExpanded}
          onPageChange={confirmedTruthPager.setPage}
          totalItems={confirmedSamples.length}
          totalPages={confirmedTruthPager.totalPages}
          visibleEnd={confirmedTruthPager.visibleEnd}
          visibleStart={confirmedTruthPager.visibleStart}
        />
      </>}
    </section>

    {samples.length === 0
      ? <section className="learning-case-list" aria-labelledby="pending-rentals-title">
          <div className="learning-case-list-heading">
            <h3 id="pending-rentals-title">待處理照片</h3>
            <button type="button" disabled={busy !== null} onClick={() => void openRentalDelete()}>
              移除待處理租單
            </button>
            <strong>0 筆</strong>
          </div>
          <div className="empty-state">目前沒有待處理資料<br /><small>新出現的夜間、低光、低信心與重拍照片會自動分類後出現在這裡</small></div>
        </section>
      : <>
        {pendingPager.visibleItems.some((group) => group.samples.some((sample) => regionTruthTasks.has(sample.task))) && <div className="region-truth-intro">
          <p className="eyebrow">HUMAN TRUTH AUDIT</p>
          <h3>照片人工真值</h3>
          <span>單圖可框選區域或直接選擇真值；取還車雙圖可人工核對角度。紅框代表問題，藍框代表正常。</span>
        </div>}
        <section className="learning-case-list" aria-labelledby="pending-rentals-title">
          <div className="learning-case-list-heading">
            <h3 id="pending-rentals-title">待處理照片</h3>
            <button type="button" disabled={busy !== null} onClick={() => void openRentalDelete()}>
              移除待處理租單
            </button>
            <strong>{samples.length} 筆</strong>
          </div>
          {pendingPager.visibleItems.map((group) => <section className="learning-case-group" key={group.key}>
          <button
            className="learning-case-heading"
            type="button"
            aria-expanded={group.key === activePendingGroupKey}
            onClick={() => setOpenPendingGroupKey(group.key === activePendingGroupKey ? null : group.key)}
          >
            <strong>{pendingGroupTitle(group)}</strong>
            <span>共 {group.samples.length} 筆 {group.key === activePendingGroupKey ? "▾" : "▸"}</span>
          </button>
          {group.key === activePendingGroupKey && <div className="learning-grid">{plateBatchSamples(group.samples).length > 0
            && renderPlateBatch(group, plateBatchSamples(group.samples))}{groupMatchingViews(plateBatchSamples(group.samples).length > 0
            ? group.samples.filter((sample) => sample.task !== "license_plate") : group.samples).map((item) => <div
            className={`learning-review-item ${item.samples.length === 2 ? "learning-review-item--paired" : item.samples[0].has_secondary_image ? "learning-review-item--double" : ""}`}
            key={item.key}
          >
            {item.samples.length === 2 && <div className="learning-review-pair-heading">
              <strong>{labelText(item.slot)} · {taskDefinitions[item.samples[0].task].label}</strong>
              <span>取車 ↔ 還車</span>
            </div>}
            <div className="learning-review-item-grid">{item.samples.map(renderPendingCard)}</div>
        </div>)}</div>}
        </section>)}
        </section>
        <PhotoListControls
          collapsedItemCount={pendingPager.collapsedItemCount}
          currentPage={pendingPager.currentPage}
          expanded={pendingPager.expanded}
          itemLabel="待處理租單"
          itemUnit="組"
          onExpandedChange={pendingPager.setExpanded}
          onPageChange={pendingPager.setPage}
          totalItems={pendingGroups.length}
          totalPages={pendingPager.totalPages}
          visibleEnd={pendingPager.visibleEnd}
          visibleStart={pendingPager.visibleStart}
        />
      </>}


    {showRentalDelete && <div
      className="learning-rental-delete-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && busy !== "delete-rental") setShowRentalDelete(false);
      }}
    >
      <section className="learning-rental-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="learning-rental-delete-title">
        <div className="learning-rental-delete-heading">
          <div>
            <h3 id="learning-rental-delete-title">移除待處理租單</h3>
            <p>依建立時間排列：舊 → 新。請選擇要從待處理清單移除的租單與車牌。</p>
          </div>
          <button type="button" aria-label="關閉移除租單清單" disabled={busy === "delete-rental"} onClick={() => setShowRentalDelete(false)}>×</button>
        </div>
        <p className="learning-rental-delete-scope">僅將所選租單的待處理照片從此清單移除；已標註人工真值、訓練資料、所有照片與租還車紀錄都會保留。</p>
        {rentalDeleteOptions.length === 0 && !rentalDeleteError
            ? <p className="learning-rental-delete-empty">目前沒有可移除的待處理租單</p>
            : rentalDeleteOptions.length > 0 && <div className="learning-rental-delete-list" role="radiogroup" aria-label="選擇要移除的待處理租單">
              {rentalDeleteOptions.map((rental, index) => {
                const key = JSON.stringify([rental.rental_id, rental.vehicle_id]);
                return <label className="learning-rental-delete-option" key={key}>
                  <input
                    type="radio"
                    name="learning-rental-delete-target"
                    value={key}
                    checked={selectedRentalKey === key}
                    onChange={() => setSelectedRentalKey(key)}
                    disabled={busy === "delete-rental"}
                  />
                  <span className="learning-rental-delete-option-copy">
                    <strong>{rental.rental_id} · 車牌 {rental.vehicle_id}</strong>
                    <small>最早 {rentalTime(rental.oldest_created_at)} · 最近 {rentalTime(rental.newest_created_at)} · 共 {rental.sample_count} 筆待處理照片</small>
                  </span>
                  {index === 0 && <em>最舊</em>}
                  {index === rentalDeleteOptions.length - 1 && <em>最新</em>}
                </label>;
              })}
            </div>}
        {rentalDeleteError && <div className="learning-rental-delete-error" role="alert">
          <span>{rentalDeleteError}</span>
          <button type="button" disabled={busy !== null} onClick={() => void refresh()}>重新讀取</button>
        </div>}
        <div className="learning-rental-delete-actions">
          <button type="button" disabled={busy === "delete-rental"} onClick={() => setShowRentalDelete(false)}>取消</button>
          <button type="button" className="danger" disabled={busy !== null || !selectedRentalKey} onClick={() => void clearRental()}>
            {busy === "delete-rental" ? "移除中…" : "移除所選待處理租單"}
          </button>
        </div>
      </section>
    </div>}

    {regionTarget && <div
      className="region-truth-modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !regionDraftActive) setRegionTarget(null);
      }}
    >
      <section className="region-truth-modal" role="dialog" aria-modal="true" aria-labelledby="region-truth-modal-title">
        <div className="region-truth-modal-heading">
          <div>
            <h3 id="region-truth-modal-title">放大照片並標記輪廓／問題區域</h3>
          </div>
          <button type="button" disabled={regionDraftActive} onClick={() => setRegionTarget(null)} aria-label="關閉區域標註">×</button>
        </div>
        <RegionAnnotationEditor
          key={`${regionTarget.sample.id}-${regionTarget.imageSlot}`}
          onDraftChange={setRegionDraftActive}
          modelHighlights={learningModelHighlights(regionTarget.sample, regionTarget.imageSlot)}
          src={learningImageUrl(regionTarget.sample.id, regionTarget.imageSlot)}
          task={regionTarget.sample.task}
          inspectionSlot={regionTarget.sample.inspection_slot}
          imageSlot={regionTarget.imageSlot}
          title={regionTarget.imageSlot === "primary"
            ? regionTarget.sample.has_secondary_image ? "取車／基準照" : "待覆核照片"
            : "還車／對比照"}
          annotations={regionAnnotations[regionTarget.sample.id] ?? []}
          onChange={(annotations) => setRegionAnnotations((current) => ({
            ...current,
            [regionTarget.sample.id]: annotations,
          }))}
        />
      </section>
    </div>}

    {showTrainingDatasets && <div
      className="learning-modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) setShowTrainingDatasets(false);
      }}
    >
      <section className="learning-modal" role="dialog" aria-modal="true" aria-labelledby="training-datasets-title">
        <div className="learning-modal-heading">
          <div>
            <p className="eyebrow">TRAINING DATASETS</p>
            <h3 id="training-datasets-title">已加入的訓練資料</h3>
            <p>每種任務都有獨立資料集。照片可放大核對；每一筆都能清除，不會混入其他模型。</p>
          </div>
          <button type="button" onClick={() => setShowTrainingDatasets(false)} aria-label="關閉訓練資料清單">×</button>
        </div>
        <div className="dataset-overview">
          {datasetGroups.map(({ task, samples: groupedSamples }) => <div key={task}>
            <span>{taskDefinitions[task].label}</span>
            <strong>{groupedSamples.length}</strong>
          </div>)}
        </div>
        {trainingSamples.length === 0
          ? <div className="empty-state">目前沒有已加入的訓練資料</div>
          : datasetGroups.map(({ task, samples: groupedSamples }) => groupedSamples.length > 0 && <TrainingDatasetGroup
            key={task}
            task={task}
            samples={groupedSamples}
            busy={busy}
            onClear={clearSample}
          />)}
      </section>
    </div>}
  </section>;
}
