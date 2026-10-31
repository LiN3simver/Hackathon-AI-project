import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  activateDamagePairModel,
  deleteManagedModel,
  getManagedReferenceGuides,
  getModelManagement,
  getPlateModelInfo,
  getReadiness,
  getRecognitionSettings,
  managedReferenceGuideImageUrl,
  updateManagedReferenceGuide,
  updateModelManagement,
  updateRecognitionSettings,
  type ExpectedView,
  type ManagedReferenceGuide,
  type ManagedModelKind,
  type ManagedModelMetric,
  type ManagedViewModel,
  type ModelManagementResponse,
  type PlateModelInfo,
  type RecognitionSettings,
  type ReadinessReport,
  type AuthPrincipal,
} from "../lib/api";
import { AccountManager } from "./AccountManager";
import { BlacklistManager } from "./BlacklistManager";
import { RagSeedManager } from "./RagSeedManager";
import { VehicleHistoryManager } from "./VehicleHistoryManager";
import { LearningInbox } from "./LearningInbox";
import { OperationsActivityHistory } from "./OperationsActivityHistory";
import { RecognitionProcessPage } from "./RecognitionProcessPage";
import { useRecognitionUnread } from "../lib/useRecognitionUnread";

type NumericKey = { [K in keyof RecognitionSettings]: RecognitionSettings[K] extends number ? K : never }[keyof RecognitionSettings];
type BooleanKey = { [K in keyof RecognitionSettings]: RecognitionSettings[K] extends boolean ? K : never }[keyof RecognitionSettings];
type NumberField = {
  key: NumericKey;
  label: string;
  help: string;
  // Plain-language advice on which way to turn the value.
  adjust?: string;
  min: number;
  max: number;
  step: number;
  percent?: boolean;
  unit?: string;
};

const PERCENT_TIP = "數值越高代表 AI 要越有把握才算數。";

const fieldGroups: { title: string; description: string; fields: NumberField[] }[] = [
  { title: "辨識時間上限", description: "每個 AI 最多等幾秒。車牌只在拍照時辨識，四角照片讀不到完整車牌即阻擋重拍；後台不再辨識車牌。整潔與車損比對在後台排隊處理。", fields: [
    { key: "timeout_view_seconds", label: "四角拍攝角度", help: "判斷照片是左前、右前、左後或右後", adjust: "常看到「逾時」就調高；希望拍照回饋更快就調低。", min: 1, max: 600, step: 1, unit: "秒" },
    { key: "timeout_vehicle_detection_seconds", label: "車身位置偵測", help: "找出車子在畫面中的位置與大小", adjust: "常看到「逾時」就調高；希望拍照回饋更快就調低。", min: 1, max: 600, step: 1, unit: "秒" },
    { key: "timeout_interior_capture_seconds", label: "車內前後座檢查", help: "確認車內照片拍的是前座或後座", adjust: "常看到「逾時」就調高；希望拍照回饋更快就調低。", min: 1, max: 600, step: 1, unit: "秒" },
    { key: "timeout_plate_seconds", label: "車牌辨識（前台）", help: "預設 5 秒，含首次載入與裁切重試；逾時或讀不到完整車牌，該張四角照片會被阻擋並要求重拍。", adjust: "可設定 1～600 秒；常因逾時被要求重拍就調高，調低可縮短前台等待。", min: 1, max: 600, step: 1, unit: "秒" },
    { key: "timeout_front_cleanliness_seconds", label: "前座髒污辨識", help: "前座髒污專用模型的等待上限；正式還車與測試辨識共用此設定", adjust: "超過上限會標示逾時並送人工覆核，首次載入模型可能較久。", min: 1, max: 600, step: 5, unit: "秒" },
    { key: "timeout_rear_cleanliness_seconds", label: "後座髒污辨識", help: "後座髒污專用模型的等待上限；正式還車與測試辨識共用此設定", adjust: "超過上限會標示逾時並送人工覆核，首次載入模型可能較久。", min: 1, max: 600, step: 5, unit: "秒" },
    { key: "timeout_damage_pair_seconds", label: "取還車車損比對", help: "比對取車與還車同角度照片是否有新車損", adjust: "逾時的組別會直接送人工比對，不會被當成沒車損。", min: 1, max: 600, step: 5, unit: "秒" },
  ]},
  { title: "照片品質檢查", description: "拍照當下立即檢查照片夠不夠清楚。「要求重拍」會擋住上傳；「提醒」只提示、不阻擋。", fields: [
    { key: "quality_min_short_edge", label: "照片最小尺寸", help: "照片較短那一邊至少要有幾個像素", adjust: "調高：太小的照片要重拍；調低：舊手機也能通過。", min: 240, max: 4096, step: 10, unit: "px" },
    { key: "quality_blur_fail", label: "太模糊：要求重拍", help: "清晰度分數低於此值就要重拍", adjust: "調高會更嚴格，輕微晃動也要重拍。不可高於「有點模糊：提醒」。", min: 0, max: 1000, step: 1, unit: "分" },
    { key: "quality_blur_warning", label: "有點模糊：提醒", help: "清晰度分數低於此值就提醒拿穩手機", adjust: "調高：更常提醒；調低：較少提醒。", min: 0, max: 2000, step: 1, unit: "分" },
    { key: "blur_model_fail_probability", label: "手晃模糊：要求重拍", help: "車外四角照片的手晃模糊機率達到此值就要重拍；專門抓夜拍時被路燈、雜訊撐高分數的模糊照", adjust: "調低：更嚴格，輕微手晃也要重拍；調高：較少誤擋。不可低於「手晃模糊：提醒」。", min: 0, max: 100, step: 1, percent: true },
    { key: "blur_model_warning_probability", label: "手晃模糊：提醒", help: "手晃模糊機率達到此值就提醒拿穩手機，並收進「訓練模型專區」給人確認清不清楚", adjust: "調低：更常提醒、收集更多照片；調高：較少提醒。", min: 0, max: 100, step: 1, percent: true },
    { key: "quality_brightness_low_fail", label: "太暗：要求重拍", help: "畫面平均亮度（0 最暗、255 最亮）低於此值", adjust: "調高：地下室等偏暗環境更容易被擋。", min: 0, max: 255, step: 1 },
    { key: "quality_brightness_low_warning", label: "偏暗：提醒", help: "平均亮度低於此值就提醒開燈或換位置", adjust: "需介於「太暗」與「偏亮」之間。", min: 0, max: 255, step: 1 },
    { key: "quality_brightness_high_warning", label: "偏亮：提醒", help: "平均亮度高於此值就提醒避開強光", adjust: "調低：大太陽下更常提醒。", min: 0, max: 255, step: 1 },
    { key: "quality_brightness_high_fail", label: "太亮／過曝：要求重拍", help: "平均亮度高於此值就要重拍", adjust: "不可低於「偏亮：提醒」。", min: 0, max: 255, step: 1 },
    { key: "quality_min_dynamic_range", label: "明暗層次最少要有多少", help: "畫面最亮和最暗的差距；太小代表霧霧的、看不清細節", adjust: "調高：灰濛濛的照片更容易被要求重拍。", min: 0, max: 255, step: 1 },
    { key: "quality_max_clipped_ratio", label: "全白或全黑區域上限", help: "過亮或過暗到看不見細節的區域最多佔畫面幾成", adjust: "調低：反光、逆光的照片更容易被要求重拍。", min: 0, max: 100, step: 1, percent: true },
  ]},
  { title: "車外四角與車身位置", description: "判斷拍攝角度對不對、車子在畫面中大小與位置是否合適。百分比代表 AI 的把握程度或車身佔畫面的比例。", fields: [
    { key: "view_confidence_threshold", label: "角度判斷要多有把握", help: "AI 判斷拍攝角度的把握低於此值，就提醒重拍", adjust: `${PERCENT_TIP}調高：角度不夠標準就提醒；調低：較少提醒，但可能放過拍錯角度的照片。`, min: 0, max: 100, step: 1, percent: true },
    { key: "vehicle_detection_confidence_threshold", label: "認出車子要多有把握", help: "AI 認出畫面中有車的把握低於此值，就當作沒拍到車", adjust: "調高：車子被遮住一部分時較容易判成沒拍到；調低：較寬鬆。", min: 0, max: 100, step: 1, percent: true },
    { key: "vehicle_coverage_low_fail", label: "車子太小：要求重拍", help: "車身只佔畫面這麼少就要重拍", adjust: "調高：要站得更近才拍得過。", min: 0, max: 100, step: 1, percent: true },
    { key: "vehicle_coverage_low_warning", label: "車子偏小：提醒", help: "車身佔畫面低於此值就提醒靠近一點", adjust: "需介於「太小」與「偏大」之間。", min: 0, max: 100, step: 1, percent: true },
    { key: "vehicle_coverage_high_warning", label: "車子偏大：提醒", help: "車身佔畫面高於此值就提醒後退一點", adjust: "調低：更常提醒後退。", min: 0, max: 100, step: 1, percent: true },
    { key: "vehicle_coverage_high_fail", label: "車子太大：要求重拍", help: "車身佔畫面超過此值，代表太近、容易被裁切", adjust: "不可低於「偏大：提醒」。", min: 0, max: 100, step: 1, percent: true },
    { key: "vehicle_edge_margin_warning", label: "車身太貼邊：提醒", help: "車身離畫面邊緣小於此值視為貼邊；同時貼到兩側以上邊界才要求重拍", adjust: "調高：車身要離邊緣更遠才不算貼邊。", min: 0, max: 25, step: .1, percent: true },
    { key: "vehicle_edge_margin_fail", label: "車身貼邊：要求重拍", help: "車身離畫面邊緣小於此值視為被切到；同時兩側以上被切到才判失敗", adjust: "不可高於「太貼邊：提醒」。", min: 0, max: 25, step: .1, percent: true },
  ]},
  { title: "車內照片", description: "前座與後座各自檢查拍攝位置，並由專用模型篩選髒污；疑似髒污送人工覆核。", fields: [
    { key: "interior_scene_veto_confidence", label: "拍錯位置：多有把握才要求重拍", help: "AI 認為這張不是指定的前座／後座，把握要達到此值才強制重拍", adjust: "調低：更容易要求重拍，但可能誤擋正確照片；調高：只擋明顯拍錯的。", min: 50, max: 100, step: 1, percent: true },
    { key: "cleanliness_fast_lane_color_threshold", label: "快速通道：色差多大算異物", help: "物品顏色與車內主色系（黑、灰、米色內裝）的色差超過此值，才視為可疑異物", adjust: "調低：顏色稍有不同就框選，誤報變多；調高：只抓紅、藍、綠等鮮豔突兀的物品。", min: 5, max: 60, step: 1 },
    { key: "cleanliness_fast_lane_min_area_ratio", label: "快速通道：異物最小面積", help: "色差異常的區塊至少要佔畫面這麼多，才直接判為疑似垃圾", adjust: "調低：較小的物品也會被框，但安全帶紅色按鈕、貼紙等也可能被誤框；調高：只抓較大的物品。", min: .1, max: 10, step: .1, percent: true },
  ]},
  { title: "車牌", description: "拍照時讀取車牌文字並保存結果，後台沿用同一次 OCR 結果供人工確認。", fields: [
    { key: "plate_recognition_confidence_threshold", label: "車牌讀取多有把握才算讀對", help: "低於此值會標示「模型不確定」，請人工核對", adjust: "調高：更多車牌給人核對；調低：較少打擾，但讀錯可能沒發現。", min: 0, max: 100, step: 1, percent: true },
    { key: "plate_hint_min_width_px", label: "拍照提示：車牌至少多寬", help: "照片中的車牌小於此寬度，就提醒靠近一點", adjust: "調高：要拍得更近才不提醒。只是提醒，不會擋上傳。", min: 10, max: 1000, step: 5, unit: "px" },
    { key: "plate_hint_min_height_px", label: "拍照提示：車牌至少多高", help: "照片中的車牌小於此高度，就提醒靠近一點", adjust: "調高：要拍得更近才不提醒。", min: 5, max: 400, step: 1, unit: "px" },
    { key: "plate_hint_min_sharpness", label: "拍照提示：車牌清晰度", help: "車牌區域清晰度低於此值，就提醒拿穩手機", adjust: "調高：更常提醒；調低：較少提醒。", min: 0, max: 1000, step: 1, unit: "分" },
  ]},
  { title: "自動分流", description: "決定哪些結果可以不經人工、直接結案或立案。目前為觀察模式時，只記錄建議、不會真的自動處理。", fields: [
    { key: "automation_auto_clear_confidence", label: "多有把握「沒問題」才自動結案", help: "AI 認為沒有異常的把握達到此值，建議自動結案", adjust: "調高：更謹慎，更多案件給人看；調低：更省人力。", min: 0, max: 100, step: 1, percent: true },
    { key: "automation_auto_case_confidence", label: "多有把握「有問題」才自動立案", help: "AI 認為有異常的把握達到此值，建議自動建立案件", adjust: "調高：較少誤立案；調低：較快立案，但誤判變多。", min: 0, max: 100, step: 1, percent: true },
  ]},
  { title: "收集學習資料", description: "決定哪些照片要收進「訓練模型專區」給人標註。", fields: [
    { key: "active_learning_score_threshold", label: "品質分數低於多少就收集", help: "照片品質分數低於此值，就收進待處理給人標註", adjust: "調高：收更多照片；調低：收得少。", min: 0, max: 100, step: 1, unit: "分" },
    { key: "active_learning_confidence_threshold", label: "AI 把握低於多少就收集", help: "任一 AI 的把握低於此值，就收進待處理", adjust: "調高：收更多照片；調低：收得少。", min: 0, max: 100, step: 1, percent: true },
    { key: "active_learning_min_training_score", label: "品質至少幾分才可用來訓練", help: "人工標註後，照片品質仍需達到此分數才會拿去訓練", adjust: "調高：只用清楚的照片訓練；調低：資料較多，但可能混入模糊照片。", min: 0, max: 100, step: 1, unit: "分" },
  ]},
];

// A pending group has no checkpoint yet; it only shows that the model still needs training.
const modelGroups: { kind: ManagedModelKind | "interior_rear_cleanliness"; eyebrow: string; title: string; description: string; pending?: boolean }[] = [
  { kind: "exterior_view", eyebrow: "EXTERIOR VIEW", title: "四角車外視角模型", description: "保留既有最多三個候選順位與失敗回退，不受其他專屬模型影響。" },
  { kind: "interior_front", eyebrow: "INTERIOR FRONT", title: "前座車內模型", description: "只判斷前座車內照片與非前座場景，與後座模型分開訓練。" },
  { kind: "interior_rear", eyebrow: "INTERIOR REAR", title: "後座車內模型", description: "只判斷後座車內照片與非後座場景，與前座模型分開訓練。" },
  { kind: "interior_front_cleanliness", eyebrow: "FRONT CLEANLINESS", title: "前座髒汙模型", description: "還車上傳前座照片後自動執行；疑似髒汙送人工覆核，分數作為篩選參考。" },
  { kind: "interior_rear_cleanliness", eyebrow: "REAR CLEANLINESS", title: "後座髒汙模型", description: "還車上傳後座照片後自動執行；疑似髒污或垃圾送人工覆核。" },
  { kind: "damage_pair", eyebrow: "DAMAGE PAIR", title: "前後車損比對模型", description: "以人工框選的取還車車損訓練；偵測到疑似新增車損時只送交後台覆核，不會自動判定車損。" },
];

const latencyPartLabels: Record<string, string> = {
  vehicle_detection: "車輛偵測",
  dinov2_vits14: "ViT-S",
  dinov2_vitb14: "ViT-B",
  dinov2_vitl14: "ViT-L",
  "timm:vit_base_patch16_siglip_512.v2_webli": "SigLIP2",
  "timm:convnext_base.fb_in22k": "ConvNeXt",
  "hf-vit:beingamit99/car_damage_detection": "CarDD ViT",
};

function formatMetricValue(metric: ManagedModelMetric, value: number): string {
  if (metric.unit === "percent") return `${(value * 100).toFixed(1)}%`;
  if (metric.unit === "seconds") return `${value.toFixed(1)} 秒`;
  return value.toFixed(3);
}

function metricTarget(metric: ManagedModelMetric): string | null {
  if (metric.target == null) return null;
  return `${metric.higher_is_better ? "目標 ≥" : "上限 ≤"} ${formatMetricValue(metric, metric.target)}`;
}

function formatSeconds(milliseconds: number): string {
  return (milliseconds / 1000).toFixed(milliseconds < 10000 ? 2 : 1);
}

function latencyBreakdown(parts: Record<string, number>): string {
  return Object.entries(parts)
    .map(([name, value]) => `${latencyPartLabels[name] ?? name} ${formatSeconds(value)} 秒`)
    .join("・");
}

function modelStageLabel(model: ManagedViewModel): string {
  if (model.stage === "trial") return "第一版試行";
  if (model.stage === "first_version") return "第一版";
  if (model.stage === "deployed") return "正式部署";
  return "歷史版本";
}

const toggles: { key: BooleanKey; label: string; help: string }[] = [
  { key: "inspection_plate_check_enabled", label: "核對車牌是否與租單相同", help: "租車、還車各讀一次車牌；讀不出來只會提醒，不會擋住流程" },
  { key: "active_learning_enabled", label: "自動收集難判斷的照片", help: "開啟後，AI 沒把握或需要重拍的照片會收進「訓練模型專區」給人標註；關閉就不再收集" },
  { key: "active_learning_collect_good_samples", label: "也收集拍得很好的照片", help: "當作正確範例保存；一樣要人工確認後才會拿去訓練" },
  { key: "blur_model_enabled", label: "手晃模糊偵測", help: "車外四角照片在車身範圍內檢查手晃、拖影與疊影；關閉後只用整張清晰度分數判斷" },
  { key: "cleanliness_fast_lane_enabled", label: "車內異物快速通道", help: "先用色差在 0.1 秒內找出與車內主色差異大、且夠大的物品並直接框選；找到就不再等待慢速 AI 定位。灰塵、腳印等同色髒污仍交給模型判斷" },
  { key: "cleanliness_vlm_only_when_suspected", label: "只在髒污模型懷疑時才做 AI 精細定位", help: "開啟後，髒污模型判斷乾淨的照片直接完成，不再呼叫視覺語言模型（CPU 約 2 分鐘）；關閉則每張車內照片都會精細定位" },
];
const settingLabels = new Map<keyof RecognitionSettings, string>([
  ...toggles.map((item) => [item.key, item.label] as const),
  ...fieldGroups.flatMap((group) => group.fields.map((field) => [field.key, field.label] as const)),
]);

function displayNumber(field: NumberField, value: number): number {
  return field.percent ? Math.round(value * 1000) / 10 : value;
}

function sameSettingValue(left: unknown, right: unknown): boolean {
  return typeof left === "number" && typeof right === "number"
    ? Math.abs(left - right) < 1e-9
    : left === right;
}

const guideViews: { view: ExpectedView; label: string }[] = [
  { view: "left_front", label: "左前方" }, { view: "right_front", label: "右前方" },
  { view: "left_rear", label: "左後方" }, { view: "right_rear", label: "右後方" },
  { view: "interior_front", label: "前座車內" }, { view: "interior_rear", label: "後座車內" },
  { view: "damage_detail", label: "車損特寫" },
];

export function AdminSettings({ currentPrincipal, sidebarCollapsed, onCloseSidebar, onEnterSidebar, onLeaveSidebar }: { currentPrincipal: AuthPrincipal; sidebarCollapsed: boolean; onCloseSidebar: () => void; onEnterSidebar: () => void; onLeaveSidebar: () => void }) {
  const { count: recognitionUnread, markViewed: markRecognitionViewed } = useRecognitionUnread(currentPrincipal.subject);
  const [settings, setSettings] = useState<RecognitionSettings | null>(null);
  const [modelManagement, setModelManagement] = useState<ModelManagementResponse | null>(null);
  const [readiness, setReadiness] = useState<ReadinessReport | null>(null);
  const [plateModel, setPlateModel] = useState<PlateModelInfo | null>(null);
  const [plateModelError, setPlateModelError] = useState<string | null>(null);
  const [learningRefreshKey, setLearningRefreshKey] = useState(0);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  // Damage-pair model ticked in the list; null = the one in use. Saving opens the password confirmation.
  const [damagePairChoice, setDamagePairChoice] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ManagedViewModel | null>(null);
  const [deleteStep, setDeleteStep] = useState<1 | 2>(1);
  const [deleteConfirmed, setDeleteConfirmed] = useState(false);
  const [adminPassword, setAdminPassword] = useState("");
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [activateTarget, setActivateTarget] = useState<ManagedViewModel | null>(null);
  const [activateConfirmed, setActivateConfirmed] = useState(false);
  const [activatePassword, setActivatePassword] = useState("");
  const [activateError, setActivateError] = useState<string | null>(null);
  const [guides, setGuides] = useState<ManagedReferenceGuide[]>([]);
  const [selectedView, setSelectedView] = useState<ExpectedView>("left_front");
  const [title, setTitle] = useState("左前方正確拍攝範本");
  const [description, setDescription] = useState("完整拍入車頭與左側車身，避免裁切。");
  const [guideFile, setGuideFile] = useState<File | null>(null);
  const [guidePreviewUrl, setGuidePreviewUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [numberDrafts, setNumberDrafts] = useState<Partial<Record<NumericKey, string>>>({});
  const [recognitionDraft, setRecognitionDraft] = useState<Partial<RecognitionSettings>>({});
  // Last settings the server confirmed; every save builds on it so unrelated drafts never leak in.
  const savedSettingsRef = useRef<RecognitionSettings | null>(null);
  const settingsQueueRef = useRef<Promise<void>>(Promise.resolve());
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState(() => {
    const validSections = ["admin-accounts", "rag-seeds", "vehicle-history", "blacklist", "model-management", "learning-inbox", "recognition-progress", "recognition-settings", "reference-guides", "operations-activity"];
    const section = window.location.hash.slice(1);
    if (section && validSections.includes(section)) return section;
    const savedSection = window.localStorage.getItem("guardian-admin-section");
    return savedSection && validSections.includes(savedSection) ? savedSection : "model-management";
  });
  const initialLoadStarted = useRef(false);
  const recognitionDirtyCount = Object.keys(recognitionDraft).length;
  const recognitionDirty = recognitionDirtyCount > 0;
  const modelQueueDirty = Boolean(modelManagement)
    && selectedModelIds.join("|") !== modelManagement?.queue.model_ids.join("|");

  // Unsaved recognition edits would silently vanish on reload, so ask the browser to warn first.
  useEffect(() => {
    if (!recognitionDirty) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [recognitionDirty]);

  const selectedGuide = useMemo(
    () => guides.find((guide) => guide.view === selectedView),
    [guides, selectedView],
  );

  useEffect(() => {
    if (initialLoadStarted.current) return;
    initialLoadStarted.current = true;
    void (async () => {
      let lastError: unknown = null;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          const [response, nextGuides, models, nextReadiness] = await Promise.all([
            getRecognitionSettings(),
            getManagedReferenceGuides(),
            getModelManagement(),
            getReadiness(),
          ]);
          savedSettingsRef.current = response.settings;
          setSettings(response.settings);
          setGuides(nextGuides);
          setModelManagement(models);
          setSelectedModelIds(models.queue.model_ids);
          setReadiness(nextReadiness);
          setError(null);
          return;
        } catch (caught) {
          lastError = caught;
          if (attempt < 2) {
            await new Promise((resolve) => window.setTimeout(resolve, 400 * (attempt + 1)));
          }
        }
      }
      setError(`管理設定載入失敗：${
        lastError instanceof Error ? lastError.message : "未知錯誤"
      }`);
    })();
  }, []);

  // Loaded on its own so a missing OCR environment never blocks the settings page.
  useEffect(() => {
    getPlateModelInfo()
      .then((info) => { setPlateModel(info); setPlateModelError(null); })
      .catch((caught) => setPlateModelError(caught instanceof Error ? caught.message : "車牌模型資訊載入失敗"));
  }, []);

  const refreshLearning = useCallback(() => {
    setLearningRefreshKey((current) => current + 1);
  }, []);

  useEffect(() => {
    const guide = guides.find((item) => item.view === selectedView);
    const label = guideViews.find((item) => item.view === selectedView)?.label ?? selectedView;
    setTitle(guide?.title ?? `${label}正確拍攝範本`);
    setDescription(guide?.description ?? "請完整拍入指定位置，保持影像清晰並避免嚴重裁切。");
    setGuideFile(null);
  }, [guides, selectedView]);

  useEffect(() => {
    if (!guideFile) {
      setGuidePreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(guideFile);
    setGuidePreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [guideFile]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 1500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  function settingValue<K extends keyof RecognitionSettings>(key: K): RecognitionSettings[K] | undefined {
    return key in recognitionDraft ? recognitionDraft[key] as RecognitionSettings[K] : settings?.[key];
  }

  function draftRecognitionSetting<K extends keyof RecognitionSettings>(key: K, value: RecognitionSettings[K]) {
    setRecognitionDraft((current) => {
      const next = { ...current };
      if (settings && sameSettingValue(settings[key], value)) delete next[key];
      else next[key] = value;
      return next;
    });
  }

  function discardRecognitionDraft() {
    setRecognitionDraft({});
    setNumberDrafts({});
  }

  function saveRecognitionSettings() {
    const draft = recognitionDraft;
    const keys = Object.keys(draft) as (keyof RecognitionSettings)[];
    if (!keys.length) return;
    setBusy("recognition");
    setError(null);
    // Saves run one at a time so two quick saves cannot overwrite each other.
    settingsQueueRef.current = settingsQueueRef.current.then(async () => {
      const saved = savedSettingsRef.current;
      try {
        if (!saved) throw new Error("設定尚未載入");
        const response = await updateRecognitionSettings({ ...saved, ...draft });
        const mismatched = keys.filter((key) => !sameSettingValue(response.settings[key], draft[key]));
        savedSettingsRef.current = response.settings;
        setSettings((current) => current ? { ...current, ...Object.fromEntries(keys.map((key) => [key, response.settings[key]])) } : current);
        // Keep anything edited while the request was in flight.
        setRecognitionDraft((current) => {
          const next = { ...current };
          keys.forEach((key) => { if (sameSettingValue(current[key], draft[key])) delete next[key]; });
          return next;
        });
        if (mismatched.length) {
          throw new Error(`伺服器回傳的值與送出的不同：${mismatched.map((key) => settingLabels.get(key) ?? key).join("、")}`);
        }
        setNotice(`已儲存 ${keys.length} 項辨識設定，立即套用到下一筆辨識。`);
      } catch (caught) {
        setError(`辨識設定未儲存：${caught instanceof Error ? caught.message : "設定儲存失敗"}；變更仍保留在頁面上，可再按一次儲存。`);
      } finally {
        setBusy(null);
      }
    });
  }

  function changeNumber(field: NumberField, event: ChangeEvent<HTMLInputElement>) {
    setNumberDrafts((current) => ({ ...current, [field.key]: event.target.value }));
  }

  function commitNumber(field: NumberField) {
    const draft = numberDrafts[field.key];
    if (draft === undefined || !settings) return;
    setNumberDrafts((current) => {
      const next = { ...current };
      delete next[field.key];
      return next;
    });
    const raw = Number(draft);
    if (draft.trim() === "" || !Number.isFinite(raw) || raw < field.min || raw > field.max) {
      setError(`「${field.label}」必須介於 ${field.min} 到 ${field.max}${field.percent ? "%" : ""}，已還原原本的值。`);
      return;
    }
    draftRecognitionSetting(field.key, field.percent ? raw / 100 : raw);
  }

  function changeBoolean(key: BooleanKey, checked: boolean) {
    if (!settings) return;
    draftRecognitionSetting(key, checked);
  }

  async function saveModelQueue() {
    const modelIds = selectedModelIds;
    if (!modelIds.length) {
      setError("第一順位必須選擇一個模型，至少要保留一個勾選。");
      return;
    }
    if (new Set(modelIds).size !== modelIds.length) {
      setError("同一個模型不可重複排入候選隊列。");
      return;
    }
    setBusy("models"); setError(null); setNotice(null);
    try {
      const response = await updateModelManagement(modelIds);
      if (response.queue.model_ids.join("|") !== modelIds.join("|")) {
        throw new Error("伺服器回傳的順位與送出的不同，變更未生效");
      }
      setModelManagement(response);
      setSelectedModelIds(response.queue.model_ids);
      setNotice(`四角模型順位已儲存（共 ${response.queue.model_ids.length} 個），四角推論立即改用新順位。`);
    } catch (caught) {
      setError(`四角模型順位未儲存：${caught instanceof Error ? caught.message : "模型隊列儲存失敗"}`);
    } finally { setBusy(null); }
  }

  function toggleModel(modelId: string, checked: boolean) {
    setError(null);
    if (!checked) {
      setSelectedModelIds((current) => current.filter((id) => id !== modelId));
      return;
    }
    if (selectedModelIds.includes(modelId)) return;
    if (selectedModelIds.length >= 3) {
      setError("候選隊列最多只能勾選三個模型。");
      return;
    }
    setSelectedModelIds([...selectedModelIds, modelId]);
  }

  function moveModel(modelId: string, direction: -1 | 1) {
    setSelectedModelIds((current) => {
      const index = current.indexOf(modelId);
      const nextIndex = index + direction;
      if (index < 0 || nextIndex < 0 || nextIndex >= current.length) return current;
      const next = [...current];
      [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
      return next;
    });
  }

  function openDeleteConfirmation(model: ManagedViewModel) {
    setDeleteTarget(model);
    setDeleteStep(1);
    setDeleteConfirmed(false);
    setAdminPassword("");
    setDeleteError(null);
  }

  function closeDeleteConfirmation() {
    if (busy === "delete") return;
    setDeleteTarget(null);
    setDeleteStep(1);
    setDeleteConfirmed(false);
    setAdminPassword("");
    setDeleteError(null);
  }

  async function confirmDeleteModel() {
    if (!deleteTarget || !deleteConfirmed || !adminPassword) return;
    setBusy("delete"); setDeleteError(null); setError(null); setNotice(null);
    try {
      const response = await deleteManagedModel(deleteTarget.id, adminPassword);
      setModelManagement(response.management);
      setSelectedModelIds(response.management.queue.model_ids);
      setNotice(`${deleteTarget.version} 已永久刪除。`);
      setDeleteTarget(null);
      setDeleteStep(1);
      setDeleteConfirmed(false);
      setAdminPassword("");
    } catch (caught) { setDeleteError(caught instanceof Error ? caught.message : "模型刪除失敗"); }
    finally { setBusy(null); }
  }

  function openActivateConfirmation(model: ManagedViewModel) {
    setActivateTarget(model);
    setActivateConfirmed(false);
    setActivatePassword("");
    setActivateError(null);
  }

  function closeActivateConfirmation() {
    if (busy === "activate") return;
    setActivateTarget(null);
    setActivateConfirmed(false);
    setActivatePassword("");
    setActivateError(null);
  }

  async function confirmActivateModel() {
    if (!activateTarget || !activateConfirmed || !activatePassword) return;
    setBusy("activate"); setActivateError(null); setError(null); setNotice(null);
    try {
      const response = await activateDamagePairModel(activateTarget.id, activatePassword);
      setModelManagement(response.management);
      setNotice(`前後車損比對已改用 ${activateTarget.version}，下一組比對立即生效${response.backup_model_id ? `；原本的模型已備份為 ${response.backup_model_id}` : ""}。`);
      setActivateTarget(null);
      setActivateConfirmed(false);
      setActivatePassword("");
      setDamagePairChoice(null);
    } catch (caught) { setActivateError(caught instanceof Error ? caught.message : "模型切換失敗"); }
    finally { setBusy(null); }
  }

  async function saveGuide() {
    setBusy("guide"); setError(null); setNotice(null);
    try {
      const saved = await updateManagedReferenceGuide(selectedView, { title, description, image: guideFile });
      setGuides((current) => [...current.filter((item) => item.view !== saved.view), saved]);
      setGuideFile(null); setNotice("示範照片與說明已更新。");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "示範資料儲存失敗"); }
    finally { setBusy(null); }
  }

  const navigationItems = [
    { id: "model-management", label: "模型管理" },
    { id: "admin-accounts", label: "帳號管理" },
    { id: "learning-inbox", label: "訓練模型專區" },
    { id: "recognition-progress", label: "辨識過程與紀錄" },
    { id: "rag-seeds", label: "模擬資料區" },
    { id: "vehicle-history", label: "同車牌租借歷程" },
    { id: "blacklist", label: "黑名單管理" },
    { id: "recognition-settings", label: "辨識與學習門檻" },
    { id: "reference-guides", label: "正確樣本示範" },
    { id: "operations-activity", label: "操作時間軸" },
  ];

  function openSection(section: string) {
    setActiveSection(section);
    window.localStorage.setItem("guardian-admin-section", section);
    window.history.replaceState(null, "", `#${section}`);
    window.scrollTo(0, 0);
  }

  function renderNumberField(field: NumberField) {
    if (!settings) return null;
    const changed = field.key in recognitionDraft;
    const unit = field.percent ? "%" : field.unit;
    return <label className={`setting-field${changed ? " settings-item--changed" : ""}`} key={field.key}><span><strong>{field.label}{changed && <em className="settings-changed-tag">未儲存</em>}</strong><small>{field.help}</small>{field.adjust && <small className="setting-field-adjust">怎麼調：{field.adjust}</small>}</span><div><input type="number" min={field.min} max={field.max} step={field.step} value={numberDrafts[field.key] ?? displayNumber(field, Number(settingValue(field.key)))} onChange={(event) => changeNumber(field, event)} onBlur={() => commitNumber(field)} onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }} />{unit && <b>{unit}</b>}</div></label>;
  }

  return <>
    <aside id="ops-sidebar-navigation" className={`ops-sidebar ${sidebarCollapsed ? "ops-sidebar--collapsed" : ""}`} hidden={sidebarCollapsed} onMouseEnter={onEnterSidebar} onMouseLeave={onLeaveSidebar}>
      <>
        <div className="ops-sidebar-heading"><span aria-hidden="true">G</span><strong>快速導覽</strong></div>
        <nav aria-label="管理員設定快速導覽">
          {navigationItems.map((item) => <button type="button" key={item.id}
            className={activeSection === item.id ? "active" : ""}
            aria-current={activeSection === item.id ? "page" : undefined}
            onClick={() => { openSection(item.id); onCloseSidebar(); }}
          ><b>{item.label}</b>{item.id === "recognition-progress" && recognitionUnread > 0 &&
            <small aria-label={`${recognitionUnread} 筆未查看辨識`} aria-live="polite">{recognitionUnread}</small>}
            {item.id === "recognition-settings" && recognitionDirty && <small className="sidebar-unsaved-dot">未儲存</small>}</button>)}
        </nav>
      </>
    </aside>
    <main className={`admin-settings-page ops-page ${activeSection === "learning-inbox" ? "admin-settings-page--learning-inbox" : ""} ${sidebarCollapsed
      ? "ops-page--sidebar-collapsed"
      : "ops-page--sidebar-open"}`}>
    {(notice || error) && <div className="admin-toast-stack">
      {notice && <div className="admin-save-toast" role="status" aria-live="polite">
        <span aria-hidden="true">✓</span>
        <div><strong>操作完成</strong><p>{notice}</p></div>
        <button type="button" aria-label="關閉儲存成功通知" onClick={() => setNotice(null)}>×</button>
      </div>}
      {error && <div className="admin-save-toast admin-save-toast--error" role="alert">
        <span aria-hidden="true">!</span>
        <div><strong>操作未完成</strong><p>{error}</p></div>
        <button type="button" aria-label="關閉錯誤通知" onClick={() => setError(null)}>×</button>
      </div>}
    </div>}
    <div className="ops-section" id="admin-accounts" hidden={activeSection !== "admin-accounts"}><AccountManager currentPrincipal={currentPrincipal} onNotice={setNotice} onError={setError} /></div>
    {activeSection === "rag-seeds" && <div className="ops-section" id="rag-seeds"><RagSeedManager /></div>}
    {activeSection === "vehicle-history" && <div className="ops-section" id="vehicle-history"><VehicleHistoryManager /></div>}
    {activeSection === "blacklist" && <div className="ops-section" id="blacklist"><BlacklistManager /></div>}
    {activeSection === "operations-activity" && <div className="ops-section"><OperationsActivityHistory scope="all" onNotice={setNotice} /></div>}
    {activeSection === "recognition-progress" && <RecognitionProcessPage onRecordsViewed={markRecognitionViewed} />}
    <div className="ops-section" hidden={activeSection !== "learning-inbox"}>
      <LearningInbox refreshKey={learningRefreshKey} onChanged={refreshLearning} />
    </div>
    {readiness && <section className={`readiness-panel readiness-panel--${readiness.overall} ops-section`} hidden={activeSection !== "model-management"}>
      <div className="readiness-heading">
        <div><p className="eyebrow">系統與模型狀態</p><h2>{readiness.overall === "ready" ? "現場系統已就緒" : readiness.overall === "degraded" ? "可降級展示" : "展示前需要處理"}</h2></div>
        <strong>{readiness.ready_count} / {readiness.total_count}</strong>
      </div>
      <div className="readiness-checks">{readiness.checks.map((check) => <div className={`readiness-check readiness-check--${check.status}`} key={check.key} title={check.detail ?? ""}>
        <i /><span><b>{check.label}</b><small>{check.message}</small></span>
      </div>)}</div>
    </section>}
    {modelManagement && <section className="model-manager ops-section" id="model-management" hidden={activeSection !== "model-management"}>
      <div className="model-manager-heading">
        <div><p className="eyebrow">MODEL REGISTRY</p><h2>模型管理</h2><p>四角視角、前座、後座、車內整潔、前後座髒汙、車損比對與車牌辨識各自顯示，不再把不同任務混在同一份模型清單。</p></div>
        <span>{modelManagement.models.length} 個模型</span>
      </div>
      <div className="model-family-list">
        {modelGroups.map((group) => {
          const isExterior = group.kind === "exterior_view";
          const isDamagePairGroup = group.kind === "damage_pair";
          const groupModels = modelManagement.models.filter((model) => model.kind === group.kind);
          // The model in use is listed first, like the first slot of the four-corner queue.
          if (isDamagePairGroup) groupModels.sort((a, b) => Number(Boolean(b.in_use)) - Number(Boolean(a.in_use)) || Number(b.is_current) - Number(a.is_current));
          const damagePairInUse = isDamagePairGroup ? groupModels.find((model) => model.in_use)?.id ?? null : null;
          const damagePairSelected = damagePairChoice ?? damagePairInUse;
          const damagePairDirty = isDamagePairGroup && damagePairChoice != null && damagePairChoice !== damagePairInUse;
          const damagePairTarget = damagePairDirty ? groupModels.find((model) => model.id === damagePairChoice) ?? null : null;
          return <section className={`model-family model-family--${group.kind}`} key={group.kind}>
            <div className="model-family-heading">
              <div><p className="eyebrow">{group.eyebrow}</p><h3>{group.title}</h3><p>{group.description}</p></div>
              <span>{group.pending ? "待訓練" : isExterior ? `${selectedModelIds.length} / 3 已勾選${modelQueueDirty ? "・尚未儲存" : ""}` : isDamagePairGroup ? `1 / ${groupModels.length} 已勾選${damagePairDirty ? "・尚未儲存" : ""}` : `${groupModels.length} 個版本`}</span>
            </div>
            {groupModels.length === 0
              ? <div className="model-family-empty">{group.pending ? "待訓練：尚未有可用的後座髒汙模型" : "尚未登錄此類模型"}</div>
              : <div className="model-check-list" aria-label={`${group.title}清單`}>
                {groupModels.map((model) => {
                  const isDamagePair = isDamagePairGroup;
                  const rank = isDamagePair ? (model.id === damagePairSelected ? 0 : -1) : selectedModelIds.indexOf(model.id);
                  const checked = (isExterior || isDamagePair) && rank >= 0;
                  const selectable = isExterior || isDamagePair;
                  const persisted = isExterior && modelManagement.queue.model_ids.includes(model.id);
                  const deleteBlocked = model.is_current || Boolean(model.in_use) || checked || persisted;
                  const quality = model.quality_score == null ? null : Math.round(model.quality_score * 1000) / 10;
                  const minimum = model.minimum_quality_score == null ? null : Math.round(model.minimum_quality_score * 100);
                  const evaluation = model.evaluation ?? [];
                  return <article className={`model-check-card${checked ? " model-check-card--selected" : ""}${selectable ? "" : " model-check-card--specialist"}`} key={model.id}>
                    {isExterior
                      ? <input type="checkbox" aria-label={`選擇 ${model.version}`} checked={checked} disabled={busy === "models" || (!checked && selectedModelIds.length >= 3)} onChange={(event) => toggleModel(model.id, event.target.checked)} />
                      : isDamagePair
                        ? <input type="checkbox" aria-label={`改用 ${model.version}`} checked={checked} disabled={busy === "activate" || (!model.in_use && !model.can_activate)} onChange={(event) => { if (event.target.checked) setDamagePairChoice(model.id); }} />
                        : <span className="model-kind-mark" aria-hidden="true">AI</span>}
                    <span className="model-rank">{selectable ? (checked ? rank + 1 : "—") : model.is_current ? "✓" : "—"}</span>
                    <div className="model-check-info">
                      <strong>{model.version}{isDamagePair || group.kind === "interior_front_cleanliness" || group.kind === "interior_rear_cleanliness" ? (model.in_use ? <em className="model-in-use">使用中</em> : null) : model.is_current ? <em>目前版本</em> : null}<em className={`model-stage model-stage--${model.stage}`}>{modelStageLabel(model)}</em></strong>
                      <small>{isDamagePair && model.is_current ? "線上檔案 damage_pair_specialist.pt · " : isDamagePair ? `${model.id} · ` : ""}{model.architecture.replace("dinov2_vits14_", "DINOv2 ")} · SHA {model.sha256.slice(0, 10)}</small>
                      {model.backbone_summary && <small className="model-backbones">{model.backbone_summary}</small>}
                      {isDamagePair && evaluation.length > 0
                        ? <details className="model-metrics-details"><summary>查看評分</summary><dl className="model-metrics" aria-label={`${model.version} 評分`}>{evaluation.map((metric) => <div key={metric.key} className={metric.passed == null ? undefined : metric.passed ? "model-metric--pass" : "model-metric--fail"}>
                          <dt>{metric.label}</dt>
                          <dd>{formatMetricValue(metric, metric.value)}{metric.passed != null && <span aria-label={metric.passed ? "達標" : "未達標"}>{metric.passed ? " ✓" : " ✗"}</span>}</dd>
                          {metricTarget(metric) && <small>{metricTarget(metric)}</small>}
                        </div>)}</dl>{model.average_latency_ms != null && <small className="model-latency">平均辨識時間 {formatSeconds(model.average_latency_ms)} 秒／組{model.latency_breakdown_ms ? `（${latencyBreakdown(model.latency_breakdown_ms)}）` : ""}</small>}</details>
                        : evaluation.length > 0
                        ? <dl className="model-metrics" aria-label={`${model.version} 評分`}>{evaluation.map((metric) => <div key={metric.key} className={metric.passed == null ? undefined : metric.passed ? "model-metric--pass" : "model-metric--fail"}>
                          <dt>{metric.label}</dt>
                          <dd>{formatMetricValue(metric, metric.value)}{metric.passed != null && <span aria-label={metric.passed ? "達標" : "未達標"}>{metric.passed ? " ✓" : " ✗"}</span>}</dd>
                          {metricTarget(metric) && <small>{metricTarget(metric)}</small>}
                        </div>)}</dl>
                        : !isExterior && quality != null && <small className="model-quality">評估 {quality} 分{minimum != null ? ` · 第一版門檻 ${minimum} 分` : ""}{model.quality_metric ? ` · ${model.quality_metric}` : ""}</small>}
                      {!isDamagePair && model.average_latency_ms != null && <small className="model-latency">平均辨識時間 {formatSeconds(model.average_latency_ms)} 秒／組{model.latency_breakdown_ms ? `（${latencyBreakdown(model.latency_breakdown_ms)}）` : ""}</small>}
                      {model.label_source && !isDamagePair && <small>{model.label_source}</small>}
                      {!model.production_approved && !selectable && <small className="model-review-note">待人工真值擴充；目前保留為可迭代的第一版候選</small>}
                    </div>
                    <div className="model-card-actions">
                      {isExterior && checked && <div className="model-order-actions"><button type="button" aria-label={`將 ${model.version} 提前`} disabled={busy === "models" || rank === 0} onClick={() => moveModel(model.id, -1)}>↑</button><button type="button" aria-label={`將 ${model.version} 往後`} disabled={busy === "models" || rank === selectedModelIds.length - 1} onClick={() => moveModel(model.id, 1)}>↓</button></div>}
                      <button type="button" className="danger-outline" disabled={deleteBlocked} title={model.is_current || model.in_use ? "使用中的模型不可刪除" : checked || persisted ? "請先取消勾選並儲存順位" : "刪除此模型"} onClick={() => openDeleteConfirmation(model)}>刪除</button>
                    </div>
                  </article>;
                })}
              </div>}
            {isDamagePairGroup && groupModels.length > 0 && <div className="model-manager-footer"><p>前後車損比對一次只用一個模型。勾選另一個版本後按儲存，輸入管理員密碼即替換；原本的模型會先備份，隨時可以切回。</p><span className={`model-queue-status${damagePairDirty ? " model-queue-status--dirty" : ""}`} role="status">{damagePairDirty ? "勾選已變更，尚未儲存；按下儲存後才會生效" : "目前使用中的模型已生效"}</span><div className="model-queue-actions">{damagePairDirty && <button type="button" disabled={busy === "activate"} onClick={() => setDamagePairChoice(null)}>還原</button>}<button className="primary" disabled={busy === "activate" || !damagePairTarget} onClick={() => damagePairTarget && openActivateConfirmation(damagePairTarget)}>儲存車損模型勾選</button></div></div>}
            {isExterior && <div className="model-manager-footer"><p>只有四角車外模型使用候選順位。其他專屬模型各自獨立，不會被排進四角推論隊列。</p><span className={`model-queue-status${modelQueueDirty ? " model-queue-status--dirty" : ""}`} role="status">{modelQueueDirty ? "勾選或順位已變更，尚未儲存；按下儲存後才會生效" : "目前順位已儲存並生效"}</span><div className="model-queue-actions">{modelQueueDirty && <button type="button" disabled={busy === "models"} onClick={() => setSelectedModelIds(modelManagement?.queue.model_ids ?? [])}>還原</button>}<button className="primary" disabled={busy === "models" || !modelQueueDirty || !selectedModelIds.length} onClick={() => void saveModelQueue()}>{busy === "models" ? "儲存中…" : "儲存四角模型勾選與順位"}</button></div></div>}
          </section>;
        })}
        <section className="model-family model-family--license_plate">
          <div className="model-family-heading">
            <div><p className="eyebrow">LICENSE PLATE OCR</p><h3>車牌辨識模型</h3><p>使用公開預訓練模型，不在本系統訓練。辨識結果只作為人工確認或修改的預填值；租單車牌不參與辨識。</p></div>
            <span>{plateModel ? (plateModel.available ? "使用中" : "尚未安裝") : plateModelError ? "無法讀取" : "讀取中"}</span>
          </div>
          {!plateModel
            ? <div className="model-family-empty">{plateModelError ?? "正在讀取車牌模型資訊…"}</div>
            : <div className="model-check-list" aria-label="車牌辨識模型清單">
              <article className="model-check-card model-check-card--specialist">
                <span className="model-kind-mark" aria-hidden="true">OCR</span>
                <span className="model-rank">{plateModel.available ? "✓" : "—"}</span>
                <div className="model-check-info">
                  <strong>{plateModel.name}{plateModel.available ? <em>目前版本</em> : null}<em className="model-stage">預訓練・免訓練</em></strong>
                  <small>{plateModel.version} · {plateModel.engine} · {plateModel.license}</small>
                  <small className={plateModel.available ? undefined : "model-review-note"}>{plateModel.message}</small>
                  <small>{plateModel.source}</small>
                </div>
              </article>
            </div>}
        </section>
      </div>
    </section>}
    {deleteTarget && <div className="model-delete-backdrop"><section className="model-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="model-delete-title">
      {deleteStep === 1 ? <>
        <p className="eyebrow">DELETE MODEL · 1 / 2</p><h2 id="model-delete-title">第一次確認：永久刪除模型？</h2>
        <p>即將刪除 <strong>{deleteTarget.version}</strong>。模型檔案刪除後無法從後台復原；若未另外備份，就必須重新訓練或重新匯入。</p>
        <dl><div><dt>模型 ID</dt><dd>{deleteTarget.id}</dd></div><div><dt>SHA-256</dt><dd>{deleteTarget.sha256}</dd></div></dl>
        <div className="model-delete-actions"><button type="button" onClick={closeDeleteConfirmation}>取消</button><button type="button" className="danger" onClick={() => setDeleteStep(2)}>我了解，繼續確認</button></div>
      </> : <>
        <p className="eyebrow">DELETE MODEL · 2 / 2</p><h2 id="model-delete-title">第二次確認：驗證管理員</h2>
        <p>請再次確認刪除目標，並輸入目前登入帳號的管理員密碼。密碼只用於這次重新驗證。</p>
        <label className="delete-confirm-check"><input type="checkbox" checked={deleteConfirmed} onChange={(event) => setDeleteConfirmed(event.target.checked)} /><span>我確認要永久刪除「{deleteTarget.version}」</span></label>
        <label className="delete-password"><span>管理員密碼</span><input type="password" autoComplete="current-password" value={adminPassword} onChange={(event) => setAdminPassword(event.target.value)} /></label>
        {deleteError && <div className="model-delete-error" role="alert">{deleteError}</div>}
        <div className="model-delete-actions"><button type="button" disabled={busy === "delete"} onClick={() => { setDeleteStep(1); setDeleteConfirmed(false); setAdminPassword(""); setDeleteError(null); }}>返回上一步</button><button type="button" className="danger" disabled={busy === "delete" || !deleteConfirmed || !adminPassword} onClick={() => void confirmDeleteModel()}>{busy === "delete" ? "驗證並刪除中…" : "驗證密碼並永久刪除"}</button></div>
      </>}
    </section></div>}
    {activateTarget && <div className="model-delete-backdrop"><section className="model-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="model-activate-title">
      <p className="eyebrow">SWITCH DAMAGE-PAIR MODEL</p><h2 id="model-activate-title">改用這個前後車損比對模型？</h2>
      <p>下一組取還車比對就會改用 <strong>{activateTarget.version}</strong>。目前使用中的模型會先備份到 history，之後可以隨時在這裡切回去。</p>
      {activateTarget.backbone_summary && <p>{activateTarget.backbone_summary}</p>}
      {(activateTarget.evaluation ?? []).some((metric) => metric.passed === false) && <div className="model-delete-error" role="note">此模型有未達標的評分：{(activateTarget.evaluation ?? []).filter((metric) => metric.passed === false).map((metric) => `${metric.label} ${formatMetricValue(metric, metric.value)}`).join("、")}。偵測到疑似車損仍只會送人工覆核。</div>}
      <label className="delete-confirm-check"><input type="checkbox" checked={activateConfirmed} onChange={(event) => setActivateConfirmed(event.target.checked)} /><span>我確認要讓線上比對改用「{activateTarget.version}」</span></label>
      <label className="delete-password"><span>管理員密碼</span><input type="password" autoComplete="current-password" value={activatePassword} onChange={(event) => setActivatePassword(event.target.value)} /></label>
      {activateError && <div className="model-delete-error" role="alert">{activateError}</div>}
      <div className="model-delete-actions"><button type="button" disabled={busy === "activate"} onClick={closeActivateConfirmation}>取消</button><button type="button" className="primary" disabled={busy === "activate" || !activateConfirmed || !activatePassword} onClick={() => void confirmActivateModel()}>{busy === "activate" ? "驗證並切換中…" : "驗證密碼並切換"}</button></div>
    </section></div>}
    {settings && <div className="ops-section" hidden={activeSection !== "recognition-settings"}>
      <header className="recognition-settings-heading recognition-settings-heading--actions" id="recognition-settings">
        <div>
          <p className="eyebrow">RECOGNITION SETTINGS</p>
          <h2>辨識與學習門檻</h2>
        </div>
        <div className="recognition-save-actions">
          <span className={`model-queue-status${recognitionDirty ? " model-queue-status--dirty" : ""}`} role="status">{recognitionDirty ? `有 ${recognitionDirtyCount} 項變更尚未儲存，記得按「儲存設定」` : "目前設定已儲存並生效"}</span>
          {recognitionDirty && <button type="button" disabled={busy === "recognition"} onClick={discardRecognitionDraft}>放棄變更</button>}
          <button type="button" className="primary" disabled={busy === "recognition" || !recognitionDirty} onClick={saveRecognitionSettings}>{busy === "recognition" ? "儲存中…" : "儲存設定"}</button>
        </div>
      </header>
      <p className="settings-autosave-note">修改後不會馬上生效：請按右上角「儲存設定」，所有變更會一起套用到下一筆辨識。改過但還沒儲存的欄位會標示「未儲存」，離開或重新整理頁面前也會提醒。</p>
      <section className="settings-toggles">{toggles.map((item) => {
        const changed = item.key in recognitionDraft;
        return <label key={item.key} className={changed ? "settings-item--changed" : undefined}><span><strong>{item.label}{changed && <em className="settings-changed-tag">未儲存</em>}</strong><small>{item.help}</small></span><input type="checkbox" checked={Boolean(settingValue(item.key))} disabled={busy === "recognition"} onChange={(event) => changeBoolean(item.key, event.target.checked)} /></label>;
      })}</section>
      <aside className="threshold-guide">
        <div><strong>「把握」是什麼？</strong><p>AI 對自己答案有多確定，0–100%。門檻調高代表 AI 要更確定才算數，會有更多照片送人工看；調低可以省人力，但出錯的機會變大。</p></div>
        <div><strong>「要求重拍」和「提醒」差在哪？</strong><p>「要求重拍」會擋住上傳，使用者必須重拍；「提醒」只顯示建議，照片仍可上傳。成對的兩個門檻，「要求重拍」一定要比「提醒」更嚴。</p></div>
        <div><strong>不確定怎麼調？</strong><p>一次只改一個數值，觀察幾筆實際結果再決定。改錯了也沒關係，每個欄位都能隨時改回；每欄下方的「怎麼調」會說明往哪個方向調會有什麼效果。</p></div>
      </aside>
      <nav className="settings-group-nav" aria-label="辨識設定分類">
        {fieldGroups.map((group, index) => <button type="button" key={group.title} onClick={() => document.getElementById(`settings-group-${index}`)?.scrollIntoView({ behavior: "smooth", block: "start" })}>{group.title}</button>)}
        {recognitionDirty && <span className="settings-unsaved-reminder" role="status">{recognitionDirtyCount} 項尚未儲存<button type="button" className="primary" disabled={busy === "recognition"} onClick={saveRecognitionSettings}>{busy === "recognition" ? "儲存中…" : "儲存設定"}</button></span>}
      </nav>
      {fieldGroups.map((group, index) => <section className="settings-group" id={`settings-group-${index}`} key={group.title}><div className="settings-group-heading"><h2>{group.title}</h2><p>{group.description}</p></div><div className="settings-grid">{group.fields.map(renderNumberField)}</div></section>)}
    </div>}
    <section className="guide-manager ops-section" id="reference-guides" hidden={activeSection !== "reference-guides"}><div className="settings-group-heading"><p className="eyebrow">CAPTURE EXAMPLES</p><h2>正確樣本示範</h2><p>每個拍攝位置可上傳或替換一張主要示範圖，並直接修改使用者看到的標題與說明。</p></div><div className="guide-editor">
      <div className="guide-preview">{guidePreviewUrl ? <img src={guidePreviewUrl} alt="新示範預覽" /> : selectedGuide?.image_available ? <img src={`${managedReferenceGuideImageUrl(selectedView)}?v=${encodeURIComponent(selectedGuide.updated_at ?? "1")}`} alt={`${title}預覽`} /> : <div>尚未上傳示範照片</div>}</div>
      <div className="guide-form"><label><span>拍攝位置</span><select value={selectedView} onChange={(event) => setSelectedView(event.target.value as ExpectedView)}>{guideViews.map((item) => <option value={item.view} key={item.view}>{item.label}</option>)}</select></label><label><span>顯示標題</span><input maxLength={120} value={title} onChange={(event) => setTitle(event.target.value)} /></label><label><span>下方文字說明</span><textarea maxLength={500} rows={4} value={description} onChange={(event) => setDescription(event.target.value)} /></label><label className="file-picker"><span>{selectedGuide?.image_available ? "替換示範照片（可不選）" : "上傳示範照片"}</span><input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => setGuideFile(event.target.files?.[0] ?? null)} /><small>{guideFile?.name ?? selectedGuide?.filename ?? "JPG、PNG、WebP，最大 15MB"}</small></label><button className="primary" disabled={busy === "guide" || !title.trim() || !description.trim() || (!selectedGuide?.image_available && !guideFile)} onClick={() => void saveGuide()}>{busy === "guide" ? "更新中…" : "儲存示範照片與文字"}</button></div>
    </div></section>
  </main></>;
}
