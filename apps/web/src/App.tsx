import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";

import { CameraCapture } from "./components/CameraCapture";
import { RecognitionTimer, type RecognitionTiming } from "./components/RecognitionTimer";
import { CleanlinessFeedback, CleanlinessPhoto, Verdict } from "./components/CleanlinessPhoto";
import { DamageComparisonPanel } from "./components/DamageComparisonPanel";
import { ZoomableImage } from "./components/ImageViewer";
import { InteriorCleanlinessPanel } from "./components/InteriorCleanlinessPanel";
import { InspectionProgress, InspectionProgressSummary } from "./components/InspectionProgress";
import { QualityResult } from "./components/QualityResult";
import { UserLogin } from "./components/UserLogin";
import { MobileAccess } from "./components/MobileAccess";
import { confirmRemoval } from "./lib/confirmRemoval";
import {
  checkImageQuality,
  completeInspectionSession,
  createInspectionSession,
  getInspectionSession,
  getInspectionRentalContext,
  getManagedReferenceGuides,
  getReferenceSamples,
  getStoredUserSession,
  learningImageUrl,
  logoutUser,
  managedReferenceGuideImageUrl,
  saveInspectionPhoto,
  testCaptureQuality,
  type AuthSession,
  type ActiveRentalContext,
  type ExpectedView,
  type InspectionSession,
  type InspectionStage,
  type ManagedReferenceGuide,
  type QualityResponse,
  type ReferenceSample,
} from "./lib/api";

type ViewType = "左前方" | "右前方" | "左後方" | "右後方" | "前座車內" | "後座車內" | "車損特寫";
type Theme = "light" | "dark";
// damageView: the exterior angle a damage close-up replaces.
type PreparedPhoto = { file: File; quality: QualityResponse; id: string; damageView?: ExpectedView };

const views: ViewType[] = ["左前方", "右前方", "左後方", "右後方", "前座車內", "後座車內", "車損特寫"];
const expectedViews: Record<ViewType, ExpectedView> = {
  左前方: "left_front",
  右前方: "right_front",
  左後方: "left_rear",
  右後方: "right_rear",
  前座車內: "interior_front",
  後座車內: "interior_rear",
  車損特寫: "damage_detail",
};
const viewTypesByExpected: Record<ExpectedView, ViewType> = {
  left_front: "左前方",
  right_front: "右前方",
  left_rear: "左後方",
  right_rear: "右後方",
  interior_front: "前座車內",
  interior_rear: "後座車內",
  damage_detail: "車損特寫",
};
const pickupViews: ViewType[] = ["左前方", "右前方", "左後方", "右後方", "車損特寫"];
const damageAngleViews: ViewType[] = ["左前方", "右前方", "左後方", "右後方"];
const SESSION_STORAGE_KEY = "guardian-active-inspection";
const MAX_DAMAGE_DETAIL_PHOTOS = 4;

const exteriorReferenceGuides: Partial<Record<ViewType, {
  src: string;
  sourceLabel: string;
  detail: string;
}>> = {
  左前方: {
    src: "/reference-views/left-front.jfif",
    sourceLabel: "拍攝參考範本 · 左前方",
    detail: "地下停車場完整車頭與左側車身",
  },
  右前方: {
    src: "/reference-views/right-front.jfif",
    sourceLabel: "拍攝參考範本 · 右前方",
    detail: "雨天完整車頭與右側車身",
  },
  左後方: {
    src: "/reference-views/left-rear.png",
    sourceLabel: "拍攝參考範本 · 左後方",
    detail: "左後方拍攝參考圖",
  },
  右後方: {
    src: "/reference-views/right-rear.png",
    sourceLabel: "拍攝參考範本 · 右後方",
    detail: "右後方拍攝參考圖",
  },
};

function initialTheme(): Theme {
  if (typeof window === "undefined") return "light";
  const saved = window.localStorage.getItem("guardian-theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function inspectionSlotState(
  session: InspectionSession | null,
  slot: ExpectedView,
  preparedSlots: Set<ExpectedView>,
  damageCoveredSlots: Set<ExpectedView>,
) {
  if (preparedSlots.has(slot)) return "prepared";
  if (!session) return damageCoveredSlots.has(slot) ? "covered" : "waiting";
  if (session.photos.some(
    (photo) => photo.slot === slot && photo.capture_status === "accepted",
  )) return "accepted";
  if (damageCoveredSlots.has(slot)) return "covered";
  const latest = [...session.photos].reverse().find((photo) => photo.slot === slot);
  return latest?.capture_status === "retry_required" ? "retry" : "waiting";
}

function PreparedDamagePhoto({
  prepared,
  index,
  disabled,
  onRemove,
}: {
  prepared: PreparedPhoto;
  index: number;
  disabled: boolean;
  onRemove: () => void;
}) {
  const preview = useMemo(() => URL.createObjectURL(prepared.file), [prepared.file]);

  useEffect(() => () => URL.revokeObjectURL(preview), [preview]);

  return <article className="damage-detail-preview">
    <ZoomableImage
      src={preview}
      alt={`車損特寫第 ${index + 1} 張預覽`}
      title={`車損特寫 ${index + 1} · ${prepared.file.name}`}
      overlay={<QualityResult result={prepared.quality} />}
    />
    <div>
      <span><strong>第 {index + 1} 張{prepared.damageView && ` · ${viewTypesByExpected[prepared.damageView]}`}</strong><small>{prepared.file.name}</small>
        <RecognitionTimer durationMs={prepared.quality.recognition_ms} /></span>
      <button type="button" disabled={disabled} onClick={onRemove}>移除</button>
    </div>
  </article>;
}

/** 通過 (null) only when the comparison found no new damage. */
function damageReason(status: string | undefined, pending?: boolean): string | null {
  if (pending) return "車損比對尚未啟用。";
  if (status === "no_change") return null;
  if (status === "review_required") return "發現新增車損，已送人工覆核。";
  return "照片無法比對，請依同角度重拍。";
}

function CaptureAnalysisResults({ result, label }: { result: QualityResponse; label?: string }) {
  return <>{result.analyses?.map((analysis) => {
    const assessment = analysis.result.assessment;
    if (analysis.task === "interior_cleanliness") return <div className={`cleanliness-result cleanliness-result--${assessment?.level ?? "cannot_assess"}`} role="status" key={analysis.task}>
      <strong>{label ?? "車內整潔辨識"}</strong>
      {analysis.error_message ? <Verdict reason={result.recognition_abandoned ? analysis.error_message : "整潔辨識失敗，請重拍。"} />
        : <CleanlinessFeedback assessment={assessment} pending={analysis.result.model_pending} />}
    </div>;
    return <div className="cleanliness-result" role="status" key={analysis.task}>
      <strong>{label ?? (analysis.task === "interior_cleanliness" ? "車內整潔辨識" : "車損辨識")}</strong>
      {analysis.error_message ? <Verdict reason={analysis.error_message} /> : <>
        <Verdict reason={damageReason(analysis.result.comparison_status, analysis.result.model_pending)} />
        {analysis.result.matched_pickup_detail != null && <small>比對{result.pickup_reference_labels?.[analysis.result.matched_pickup_detail]
          ?? `借車車損特寫第 ${analysis.result.matched_pickup_detail + 1} 張`}</small>}
      </>}
    </div>;
  })}</>;
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [selectedView, setSelectedView] = useState<ViewType>("左前方");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<QualityResponse | null>(null);
  const [pendingSlots, setPendingSlots] = useState<Set<ExpectedView>>(new Set());
  const [captureTimings, setCaptureTimings] = useState<Partial<Record<ExpectedView, RecognitionTiming>>>({});
  const autoSubmitKey = useRef<string | null>(null);
  const pendingSlotsRef = useRef(new Set<ExpectedView>());
  const captureGeneration = useRef(0);
  const selectedViewRef = useRef(selectedView);
  selectedViewRef.current = selectedView;
  const captureResults = useRef<Partial<Record<ExpectedView, {
    file: File | null; quality: QualityResponse | null; error: string | null;
  }>>>({});
  const loading = pendingSlots.has(expectedViews[selectedView]);
  const anyLoading = pendingSlots.size > 0;
  const [error, setError] = useState<string | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [referenceSamples, setReferenceSamples] = useState<ReferenceSample[]>([]);
  const [managedGuides, setManagedGuides] = useState<ManagedReferenceGuide[]>([]);
  const bootstrapRef = useRef<{ auth: AuthSession; promise: Promise<{ context: ActiveRentalContext; session: InspectionSession }> } | null>(null);
  const [authSession, setAuthSession] = useState<AuthSession | null>(getStoredUserSession);
  const [loginNotice, setLoginNotice] = useState<string | null>(null);
  const [rentalContext, setRentalContext] = useState<ActiveRentalContext | null>(null);
  const [contextLoading, setContextLoading] = useState(true);
  const [inspectionStage, setInspectionStage] = useState<InspectionStage>("pickup");
  const [liveInspectionSession, setInspectionSession] = useState<InspectionSession | null>(null);
  const [testMode, setTestMode] = useState(false);
  const captureFlowId = useRef(window.crypto.randomUUID());
  const testPickupPhotos = useRef<Partial<Record<ExpectedView, PreparedPhoto>>>({});
  const testPickupDetails = useRef<PreparedPhoto[]>([]);
  const [testCompleted, setTestCompleted] = useState(false);
  const realCaptureRef = useRef<{
    stage: InspectionStage;
    photos: Partial<Record<ExpectedView, PreparedPhoto>>;
    details: PreparedPhoto[];
    view: ViewType;
    file: File | null;
    result: QualityResponse | null;
  } | null>(null);
  const inspectionSession: InspectionSession | null = testMode ? {
    id: "local-capture-test", rental_id: "辨識測試", vehicle_id: "辨識測試",
    stage: inspectionStage, status: testCompleted ? "capture_completed" : "in_progress",
    required_slots: (inspectionStage === "pickup" ? pickupViews : views)
      .filter((view) => view !== "車損特寫").map((view) => expectedViews[view]),
    completed_slots: [], missing_slots: [], photos: [],
    created_at: "", updated_at: "", capture_completed_at: null,
  } : liveInspectionSession;
  const [inspectionLoading, setInspectionLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [pickupSending, setPickupSending] = useState(false);
  const [pickupSendError, setPickupSendError] = useState<string | null>(null);
  const pickupJobRef = useRef<{
    session: InspectionSession;
    photos: Partial<Record<ExpectedView, PreparedPhoto>>;
    details: PreparedPhoto[];
    promise: Promise<InspectionSession> | null;
    returnSession: InspectionSession | null;
  } | null>(null);
  const [preparedPhotos, setPreparedPhotos] = useState<Partial<Record<ExpectedView, PreparedPhoto>>>({});
  const [damageDetailPhotos, setDamageDetailPhotos] = useState<PreparedPhoto[]>([]);
  // Angle the next damage close-up replaces; required before uploading one.
  const [damageView, setDamageView] = useState<ExpectedView | null>(null);
  const returnDraftRef = useRef<{
    photos: Partial<Record<ExpectedView, PreparedPhoto>>; details: PreparedPhoto[];
  } | null>(null);
  const [saveNotice, setSaveNotice] = useState<string | null>(null);
  const [inspectionError, setInspectionError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("guardian-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!loginNotice) return;
    const timer = window.setTimeout(() => setLoginNotice(null), 1500);
    return () => window.clearTimeout(timer);
  }, [loginNotice]);

  useEffect(() => {
    if (!saveNotice) return;
    const timer = window.setTimeout(() => setSaveNotice(null), 1500);
    return () => window.clearTimeout(timer);
  }, [saveNotice]);

  useEffect(() => {
    if (!authSession) {
      bootstrapRef.current = null;
      return;
    }
    let cancelled = false;
    setContextLoading(true);
    setInspectionLoading(true);
    setInspectionError(null);
    // Reuse the request during StrictMode effect replay to avoid creating two sessions.
    if (bootstrapRef.current?.auth !== authSession) {
      bootstrapRef.current = { auth: authSession, promise: (async () => {
        const context = await getInspectionRentalContext();
        if (bootstrapRef.current?.auth === authSession) setRentalContext(context);
        let session: InspectionSession | null = null;
        const currentSessionId = context.completed_return_session_id ?? context.active_session_id;
        if (currentSessionId) {
          session = await getInspectionSession(currentSessionId);
        }
        if (!session) {
          window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
          session = await createInspectionSession(context.return_allowed ? "return" : "pickup");
        }
        return { context, session };
      })() };
    }
    void bootstrapRef.current.promise.then(({ context, session }) => {
      if (cancelled) return;
      setRentalContext(context);
      setInspectionSession(session);
      setInspectionStage(session.stage);
      const nextSlot = session.missing_slots[0] ?? session.required_slots[0];
      if (nextSlot) setSelectedView(viewTypesByExpected[nextSlot]);
      window.sessionStorage.setItem(SESSION_STORAGE_KEY, session.id);
      setCameraOpen(false);
    }).catch((caught) => {
      if (!cancelled) setInspectionError(caught instanceof Error ? caught.message : "無法開始拍攝，請重試。");
    }).finally(() => {
      if (!cancelled) {
        setContextLoading(false);
        setInspectionLoading(false);
      }
    });
    return () => { cancelled = true; };
  }, [authSession]);

  useEffect(() => {
    if (!authSession || testMode || !liveInspectionSession || anyLoading || saving
        || pickupSending || inspectionLoading || contextLoading) return;
    let stopped = false;
    let polling = false;
    const sessionId = liveInspectionSession.id;
    const timer = window.setInterval(() => {
      if (polling) return;
      polling = true;
      void (async () => {
        try {
          const context = await getInspectionRentalContext();
          const currentId = context.completed_return_session_id ?? context.active_session_id;
          if (!currentId) return;
          const current = await getInspectionSession(currentId);
          if (stopped) return;
          setRentalContext(context);
          if (current.id !== sessionId) {
            clearAllPreparedPhotos();
            setInspectionStage(current.stage);
            const nextSlot = current.missing_slots[0] ?? current.required_slots[0];
            if (nextSlot) setSelectedView(viewTypesByExpected[nextSlot]);
            setCameraOpen(false);
            setInspectionError(null);
            setSaveNotice("租單流程已更新，已切換至目前拍攝進度");
          }
          setInspectionSession(current);
          window.sessionStorage.setItem(SESSION_STORAGE_KEY, current.id);
        } catch { /* Keep the current page during a temporary connection failure. */ }
        finally { polling = false; }
      })();
    }, 2500);
    return () => { stopped = true; window.clearInterval(timer); };
  }, [authSession, testMode, liveInspectionSession?.id, anyLoading, saving,
      pickupSending, inspectionLoading, contextLoading]);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!authSession) return;
    void getManagedReferenceGuides()
      .then(setManagedGuides)
      .catch(() => setManagedGuides([]));
  }, [authSession]);

  useEffect(() => {
    const expectedView = expectedViews[selectedView];
    if (expectedView.startsWith("interior") || expectedView === "damage_detail") {
      setReferenceSamples([]);
      return;
    }
    let disposed = false;
    setReferenceSamples([]);
    void getReferenceSamples(expectedView)
      .then((samples) => { if (!disposed) setReferenceSamples(samples); })
      .catch(() => { if (!disposed) setReferenceSamples([]); });
    return () => { disposed = true; };
  }, [selectedView]);

  const analysisStatusKey = inspectionSession?.photos
    .map((photo) => `${photo.id}:${photo.analysis_status}`)
    .join("|");

  useEffect(() => {
    if (testMode || !inspectionSession) return;
    const hasActiveAnalysis = inspectionSession.photos.some(
      (photo) => photo.analysis_status === "queued" || photo.analysis_status === "processing",
    );
    if (!hasActiveAnalysis) return;
    const sessionId = inspectionSession.id;
    const interval = window.setInterval(() => {
      void getInspectionSession(sessionId)
        .then((session) => setInspectionSession((current) => (
          current?.id === session.id ? session : current
        )))
        .catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(interval);
  }, [inspectionSession?.id, analysisStatusKey, testMode]);

  function toggleTestMode() {
    if (anyLoading || saving || pickupJobRef.current || inspectionLoading || contextLoading) return;
    captureFlowId.current = window.crypto.randomUUID();
    setCameraOpen(false);
    setInspectionError(null);
    setTestCompleted(false);
    testPickupPhotos.current = {};
    testPickupDetails.current = [];
    if (!testMode) {
      realCaptureRef.current = {
        stage: inspectionStage, photos: preparedPhotos, details: damageDetailPhotos,
        view: selectedView, file, result,
      };
      clearAllPreparedPhotos();
      setSelectedView("左前方");
      setTestMode(true);
    } else {
      clearAllPreparedPhotos();
      const saved = realCaptureRef.current;
      if (saved) {
        setInspectionStage(saved.stage);
        setPreparedPhotos(saved.photos);
        setDamageDetailPhotos(saved.details);
        setSelectedView(saved.view);
        setFile(saved.file);
        setResult(saved.result);
      }
      realCaptureRef.current = null;
      setTestMode(false);
    }
  }

  function selectTestStage(stage: InspectionStage) {
    captureFlowId.current = window.crypto.randomUUID();
    if (inspectionStage === "pickup" && stage === "return") {
      testPickupPhotos.current = { ...preparedPhotos };
      testPickupDetails.current = [...damageDetailPhotos];
    } else if (stage === "pickup") {
      testPickupPhotos.current = {};
      testPickupDetails.current = [];
    }
    setInspectionStage(stage);
    setTestCompleted(false);
    setSelectedView("左前方");
    setCameraOpen(false);
    clearAllPreparedPhotos();
  }

  async function startInspection(restartPickup = false) {
    if (inspectionLoading) return;
    setInspectionLoading(true);
    setInspectionError(null);
    try {
      const context = await getInspectionRentalContext();
      setRentalContext(context);
      const currentSessionId = context.completed_return_session_id ?? context.active_session_id;
      const session = !restartPickup && currentSessionId
        ? await getInspectionSession(currentSessionId)
        : await createInspectionSession(restartPickup ? "pickup" : context.return_allowed ? "return" : "pickup", restartPickup);
      setInspectionSession(session);
      setInspectionStage(session.stage);
      setSelectedView(viewTypesByExpected[session.required_slots[0]]);
      window.sessionStorage.setItem(SESSION_STORAGE_KEY, session.id);
      clearAllPreparedPhotos();
      setCameraOpen(false);
    } catch (caught) {
      setInspectionError(caught instanceof Error ? caught.message : "無法開始拍攝，請重試。");
    } finally {
      setInspectionLoading(false);
    }
  }

  function startNewInspection() {
    if (testMode) {
      selectTestStage(inspectionStage);
      return;
    }
    void startInspection(inspectionSession?.stage === "return" && inspectionSession.status === "capture_completed");
  }

  // Test-mode pickup photos of one angle: its standard photo, then the
  // close-ups that replaced or supplemented it.
  function testPickupReferences(view: ExpectedView) {
    const standard = testPickupPhotos.current[view];
    const closeups = testPickupDetails.current.filter((photo) => photo.damageView === view);
    const label = viewTypesByExpected[view];
    return {
      standard: standard?.file,
      closeups: closeups.map((photo) => photo.file),
      labels: [
        ...(standard ? [`借車${label}一般照`] : []),
        ...closeups.map((_, index) => `借車${label}車損特寫第 ${index + 1} 張`),
      ],
    };
  }

  async function analyze(nextFile: File) {
    if (!inspectionSession) {
      setInspectionError("請先建立本次借車或還車拍攝流程。");
      return;
    }
    if (inspectionSession.status === "capture_completed") {
      setInspectionError("此拍攝流程已完成；請建立新的流程後再上傳。");
      return;
    }
    const selectedSlot = expectedViews[selectedView];
    if (pendingSlotsRef.current.has(selectedSlot) || saving) return;
    const generation = captureGeneration.current;
    const startedAt = performance.now();
    setCaptureTimings((current) => ({
      ...current, [selectedSlot]: { start: startedAt, filename: nextFile.name },
    }));
    captureResults.current[selectedSlot] = { file: nextFile, quality: null, error: null };
    setFile(nextFile);
    setResult(null);
    setError(null);
    setInspectionError(null);
    setPreparedPhotos((current) => {
      const next = { ...current };
      delete next[selectedSlot];
      return next;
    });
    pendingSlotsRef.current.add(selectedSlot);
    setPendingSlots(new Set(pendingSlotsRef.current));
    try {
      const id = window.crypto.randomUUID();
      let quality: QualityResponse;
      if (testMode) {
        // Without a standard pickup photo, compare with the pickup close-ups of this angle.
        const references = testPickupReferences(selectedSlot);
        const closeupsOnly = inspectionStage === "return" && !references.standard;
        const response = await testCaptureQuality(
          nextFile, selectedSlot, inspectionStage, references.standard,
          closeupsOnly ? references.closeups : [],
          undefined, captureFlowId.current,
        );
        quality = {
          ...response, recognition_ms: response.recognition_ms ?? performance.now() - startedAt,
          pickup_reference_labels: closeupsOnly ? references.labels : undefined,
        };
      } else {
        // Check only; photos are saved when the user submits the whole set.
        quality = await checkImageQuality(
          inspectionSession.id === "pending-return" ? null : inspectionSession.id,
          nextFile,
          selectedSlot,
          captureFlowId.current,
        );
        quality.recognition_ms ??= performance.now() - startedAt;
      }
      if (generation !== captureGeneration.current) return;
      setCaptureTimings((current) => ({
        ...current, [selectedSlot]: { start: startedAt, durationMs: quality.recognition_ms ?? performance.now() - startedAt, filename: nextFile.name },
      }));
      captureResults.current[selectedSlot] = { file: nextFile, quality, error: null };
      if (expectedViews[selectedViewRef.current] === selectedSlot) setResult(quality);
      setSaveNotice(null);
      if (quality.status !== "fail") {
        setPreparedPhotos((current) => ({
          ...current,
          [selectedSlot]: { file: nextFile, quality, id },
        }));
      }
      return quality;
    } catch (caught) {
      if (generation !== captureGeneration.current) return;
      const message = caught instanceof Error ? caught.message : "無法分析圖片，請稍後再試。";
      setCaptureTimings((current) => ({
        ...current, [selectedSlot]: { start: startedAt, durationMs: performance.now() - startedAt, filename: nextFile.name },
      }));
      captureResults.current[selectedSlot] = { file: nextFile, quality: null, error: message };
      if (expectedViews[selectedViewRef.current] === selectedSlot) setError(message);
    } finally {
      if (generation === captureGeneration.current) {
        pendingSlotsRef.current.delete(selectedSlot);
        setPendingSlots(new Set(pendingSlotsRef.current));
      }
    }
  }

  async function analyzeDamageDetails(selectedFiles: File[]) {
    if (pendingSlotsRef.current.has("damage_detail") || saving) return;
    const generation = captureGeneration.current;
    if (!inspectionSession) {
      setInspectionError("請先建立本次借車或還車拍攝流程。");
      return;
    }
    if (inspectionSession.status === "capture_completed") {
      setInspectionError("此拍攝流程已完成；請建立新的流程後再上傳。");
      return;
    }
    const angle = damageView;
    if (!angle) {
      setError("請先選擇車損角度，再上傳車損特寫。");
      return;
    }
    const remaining = MAX_DAMAGE_DETAIL_PHOTOS - damageDetailPhotos.length;
    if (remaining <= 0) {
      setError(`車損特寫一次最多 ${MAX_DAMAGE_DETAIL_PHOTOS} 張；請先移除照片再新增。`);
      return;
    }

    const candidates = selectedFiles.slice(0, remaining);
    const messages = selectedFiles.length > remaining
      ? [`已達上限，只加入前 ${remaining} 張；車損特寫一次最多 ${MAX_DAMAGE_DETAIL_PHOTOS} 張。`]
      : [];
    const accepted = [...damageDetailPhotos];
    let latestQuality: QualityResponse | null = null;

    pendingSlotsRef.current.add("damage_detail");
    setPendingSlots(new Set(pendingSlotsRef.current));
    setResult(null);
    setError(null);
    setInspectionError(null);
    setSaveNotice(null);
    try {
      for (const candidate of candidates) {
        const startedAt = performance.now();
        setCaptureTimings((current) => ({
          ...current, damage_detail: { start: startedAt, filename: candidate.name },
        }));
        const id = window.crypto.randomUUID();
        // Close-ups skip the four-corner and plate checks. Test mode compares a
        // return close-up right away with the pickup photos of the same angle.
        const references = testPickupReferences(angle);
        const quality = testMode ? {
          ...await testCaptureQuality(
            candidate, "damage_detail", inspectionStage, undefined,
            inspectionStage === "return" ? [...(references.standard ? [references.standard] : []), ...references.closeups] : [],
            angle,
          ),
          pickup_reference_labels: references.labels,
        } : await checkImageQuality(
          inspectionSession.id === "pending-return" ? null : inspectionSession.id,
          candidate,
          "damage_detail",
        );
        if (generation !== captureGeneration.current) return;
        quality.recognition_ms ??= performance.now() - startedAt;
        setCaptureTimings((current) => ({
          ...current, damage_detail: { start: startedAt, durationMs: quality.recognition_ms!, filename: candidate.name },
        }));
        latestQuality = quality;
        if (quality.status === "fail") {
          messages.push(`${candidate.name} 未通過品質檢查，請重拍。`);
          continue;
        }
        const prepared = { file: candidate, quality, id, damageView: angle };
        accepted.push(prepared);
        setDamageDetailPhotos([...accepted]);
      }
      const message = messages.length ? messages.join(" ") : null;
      captureResults.current.damage_detail = { file: null, quality: latestQuality, error: message };
      if (selectedViewRef.current === "車損特寫") {
        setFile(null);
        setResult(latestQuality);
        setError(message);
      }
      return latestQuality;
    } catch (caught) {
      if (generation !== captureGeneration.current) return;
      setCaptureTimings((current) => {
        const timing = current.damage_detail;
        return timing ? { ...current, damage_detail: { ...timing, durationMs: timing.durationMs ?? performance.now() - timing.start } } : current;
      });
      const message = caught instanceof Error ? caught.message : "無法分析車損特寫，請稍後再試。";
      captureResults.current.damage_detail = { file: null, quality: latestQuality, error: message };
      if (selectedViewRef.current === "車損特寫") setError(message);
    } finally {
      if (generation === captureGeneration.current) {
        pendingSlotsRef.current.delete("damage_detail");
        setPendingSlots(new Set(pendingSlotsRef.current));
      }
    }
  }

  function sendPickupInBackground(): Promise<InspectionSession> {
    const job = pickupJobRef.current!;
    if (job.promise) return job.promise;
    setPickupSending(true);
    setPickupSendError(null);
    const promise = (async () => {
      // Keep the pickup snapshot separate so responses cannot overwrite return photos.
      for (const slot of job.session.required_slots) {
        if (job.session.completed_slots.includes(slot)) continue;
        const prepared = job.photos[slot];
        if (!prepared) continue;
        const capture = await saveInspectionPhoto(job.session.id, prepared.file, slot);
        job.session = capture.session;
        if (capture.quality.retake_required) {
          job.photos[slot] = { ...prepared, quality: capture.quality };
          throw new Error(`${viewTypesByExpected[slot]}需要重拍，請返回借車修正。`);
        }
      }
      for (const prepared of [...job.details]) {
        const capture = await saveInspectionPhoto(job.session.id, prepared.file, "damage_detail", prepared.damageView);
        job.session = capture.session;
        if (capture.quality.retake_required) {
          job.details = job.details.map((item) => item.id === prepared.id ? { ...item, quality: capture.quality } : item);
          throw new Error("借車車損特寫需要重拍，請返回借車修正。");
        }
        job.details = job.details.filter((item) => item.id !== prepared.id);
      }
      if (job.session.status !== "capture_completed") {
        job.session = await completeInspectionSession(job.session.id);
      }
      const returnSession = job.returnSession ?? await createInspectionSession("return");
      job.returnSession = returnSession;
      if (pickupJobRef.current === job) {
        setInspectionSession(returnSession);
        window.sessionStorage.setItem(SESSION_STORAGE_KEY, returnSession.id);
        setRentalContext((context) => context ? { ...context, pickup_completed: true, return_allowed: true } : context);
        setSaveNotice("借車照片已保存，完成還車後才會一併送入後台；可繼續還車拍攝");
        pickupJobRef.current = null;
      }
      return returnSession;
    })();
    job.promise = promise;
    void promise.catch((caught) => {
      if (pickupJobRef.current === job) {
        setPickupSendError(caught instanceof Error ? caught.message : "借車照片送出失敗，請重試。");
      }
    }).finally(() => {
      job.promise = null;
      if (pickupJobRef.current === job || pickupJobRef.current === null) setPickupSending(false);
    });
    return promise;
  }

  function restorePickupForCorrection() {
    const job = pickupJobRef.current;
    if (!job || job.promise || anyLoading || saving) return;
    returnDraftRef.current = { photos: preparedPhotos, details: damageDetailPhotos };
    clearAllPreparedPhotos();
    setInspectionSession(job.session);
    setInspectionStage("pickup");
    setPreparedPhotos(Object.fromEntries(Object.entries(job.photos).filter(([, photo]) => !photo.quality.retake_required)));
    setDamageDetailPhotos(job.details.filter((photo) => !photo.quality.retake_required));
    setSelectedView("左前方");
    setPickupSendError(null);
    pickupJobRef.current = null;
  }

  async function saveAllPhotos() {
    if (!inspectionSession || pendingSlotsRef.current.size || saving) return;
    const alreadySaved = new Set(
      inspectionSession.photos
        .filter((photo) => photo.capture_status === "accepted")
        .map((photo) => photo.slot),
    );
    const missingPrepared = inspectionSession.required_slots.filter(
      (slot) => !alreadySaved.has(slot) && !preparedPhotos[slot] && !damageCoveredSlots.has(slot),
    );
    if (missingPrepared.length) {
      setInspectionError("所有必拍角度完成後，才可以一鍵儲存。");
      return;
    }
    if (testMode) {
      if (inspectionStage === "pickup") {
        selectTestStage("return");
        setSaveNotice("借車辨識測試完成，已切換還車測試；未送入後台");
      } else {
        setTestCompleted(true);
        setCameraOpen(false);
        setSaveNotice("借還車辨識測試完成；未送入後台");
      }
      return;
    }
    if (inspectionSession.stage === "pickup") {
      pickupJobRef.current = {
        session: inspectionSession, photos: { ...preparedPhotos }, details: [...damageDetailPhotos],
        promise: null, returnSession: null,
      };
      const slots = views.filter((view) => view !== "車損特寫").map((view) => expectedViews[view]);
      clearAllPreparedPhotos();
      if (returnDraftRef.current) {
        setPreparedPhotos(returnDraftRef.current.photos);
        setDamageDetailPhotos(returnDraftRef.current.details);
        returnDraftRef.current = null;
      }
      setInspectionStage("return");
      setInspectionSession({
        ...inspectionSession, id: "pending-return", stage: "return", status: "in_progress",
        required_slots: slots, completed_slots: [], missing_slots: slots, photos: [], capture_completed_at: null,
      });
      setSelectedView("左前方");
      setCameraOpen(false);
      setInspectionError(null);
      setSaveNotice("已切換還車，借車照片正在背景送出");
      void sendPickupInBackground();
      return;
    }
    setSaving(true);
    setError(null);
    setInspectionError(null);
    let activeSlot: ExpectedView | null = null;
    try {
      const uploadSession = inspectionSession.id === "pending-return"
        ? await sendPickupInBackground() : inspectionSession;
      const uploadSlots = [...uploadSession.required_slots];
      let latestSession = uploadSession;
      for (const slot of uploadSlots) {
        if (alreadySaved.has(slot)) continue;
        const prepared = preparedPhotos[slot];
        if (!prepared) continue;
        activeSlot = slot;
        const capture = await saveInspectionPhoto(
          uploadSession.id,
          prepared.file,
          slot,
        );
        latestSession = capture.session;
        setInspectionSession(capture.session);
        if (capture.quality.status === "fail") {
          setSelectedView(viewTypesByExpected[slot]);
          setFile(prepared.file);
          setResult(capture.quality);
          setPreparedPhotos((current) => {
            const next = { ...current };
            delete next[slot];
            return next;
          });
          setError("伺服器最終檢查要求重拍；本次尚未全部送出。");
          return;
        }
        setPreparedPhotos((current) => {
          const next = { ...current };
          delete next[slot];
          return next;
        });
      }
      for (const prepared of damageDetailPhotos) {
        activeSlot = "damage_detail";
        const capture = await saveInspectionPhoto(
          uploadSession.id,
          prepared.file,
          "damage_detail",
          prepared.damageView,
        );
        latestSession = capture.session;
        setInspectionSession(capture.session);
        if (capture.quality.status === "fail") {
          setSelectedView("車損特寫");
          setResult(capture.quality);
          setDamageDetailPhotos((current) => current.filter(
            (item) => item.id !== prepared.id,
          ));
          setError("伺服器最終檢查要求重拍這張車損特寫；其他照片仍保留。");
          return;
        }
        setDamageDetailPhotos((current) => current.filter(
          (item) => item.id !== prepared.id,
        ));
      }
      setInspectionSession(latestSession);
      const completed = await completeInspectionSession(uploadSession.id);
      clearCurrentPhoto();
      setPreparedPhotos({});
      setDamageDetailPhotos([]);
      setInspectionSession(completed);
      window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
      setSaveNotice("儲存成功，已送入後台");
    } catch (caught) {
      if (activeSlot && activeSlot !== "damage_detail") {
        const failedSlot = activeSlot;
        const prepared = preparedPhotos[failedSlot];
        setSelectedView(viewTypesByExpected[failedSlot]);
        setFile(prepared?.file ?? null);
        setResult(prepared?.quality ?? null);
        setPreparedPhotos((current) => {
          const next = { ...current };
          delete next[failedSlot];
          return next;
        });
      }
      setError(caught instanceof Error ? caught.message : "照片儲存失敗，請稍後再試。");
    } finally {
      setSaving(false);
    }
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files ?? []);
    if (isDamageDetail && selectedFiles.length) {
      void analyzeDamageDetails(selectedFiles);
    } else if (selectedFiles[0]) {
      void analyze(selectedFiles[0]);
    }
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (!inspectionSession || inspectionSession.status === "capture_completed" || loading || saving) return;
    const selectedFiles = Array.from(event.dataTransfer.files ?? []);
    if (isDamageDetail && selectedFiles.length) {
      void analyzeDamageDetails(selectedFiles);
    } else if (selectedFiles[0]) {
      void analyze(selectedFiles[0]);
    }
  }

  function clearCurrentPhoto() {
    setFile(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  function clearAllPreparedPhotos() {
    captureGeneration.current += 1;
    captureResults.current = {};
    pendingSlotsRef.current.clear();
    setPendingSlots(new Set());
    setCaptureTimings({});
    autoSubmitKey.current = null;
    setPreparedPhotos({});
    setDamageDetailPhotos([]);
    setDamageView(null);
    setSaveNotice(null);
    clearCurrentPhoto();
  }

  function removeSelectedPhoto() {
    if (loading || saving) return;
    if (!confirmRemoval("只會移除目前選取的照片與本頁品質檢查結果；已送出的租還車照片紀錄和電腦原檔不受影響。")) return;
    const slot = expectedViews[selectedView];
    delete captureResults.current[slot];
    setPreparedPhotos((current) => {
      const next = { ...current };
      delete next[slot];
      return next;
    });
    clearCurrentPhoto();
  }

  function removeDamageDetailPhoto(id: string) {
    if (loading || saving) return;
    if (!confirmRemoval("只會移除這張尚未送出的車損特寫照片；已保存的照片和電腦原檔不受影響。")) return;
    const next = damageDetailPhotos.filter((item) => item.id !== id);
    setDamageDetailPhotos(next);
    captureResults.current.damage_detail = { file: null, quality: next.at(-1)?.quality ?? null, error: null };
    setResult(next.at(-1)?.quality ?? null);
    setError(null);
  }

  function selectView(view: ViewType) {
    if (view === selectedView) return;
    selectedViewRef.current = view;
    setSelectedView(view);
    if (view === "車損特寫") {
      setFile(null);
      setResult(captureResults.current.damage_detail?.quality ?? damageDetailPhotos.at(-1)?.quality ?? null);
      setError(captureResults.current.damage_detail?.error ?? null);
      if (inputRef.current) inputRef.current.value = "";
      return;
    }
    const prepared = preparedPhotos[expectedViews[view]];
    const capture = captureResults.current[expectedViews[view]];
    setFile(capture?.file ?? prepared?.file ?? null);
    setResult(capture?.quality ?? prepared?.quality ?? null);
    setError(capture?.error ?? null);
    if (inputRef.current) inputRef.current.value = "";
  }

  const isInterior = selectedView === "前座車內" || selectedView === "後座車內";
  const isDamageDetail = selectedView === "車損特寫";
  const damageUploadBlocked = isDamageDetail
    && (!damageView || damageDetailPhotos.length >= MAX_DAMAGE_DETAIL_PHOTOS);
  const selectedExteriorGuide = exteriorReferenceGuides[selectedView];
  const selectedManagedGuide = managedGuides.find(
    (guide) => guide.view === expectedViews[selectedView] && guide.image_available,
  );
  const selectedGuide = selectedManagedGuide ? {
    src: managedReferenceGuideImageUrl(selectedManagedGuide.view),
    sourceLabel: selectedManagedGuide.title,
    detail: selectedManagedGuide.description,
  } : selectedExteriorGuide;
  const availableViews = inspectionStage === "pickup" ? pickupViews : views;
  const inspectionClosed = inspectionSession?.status === "capture_completed";
  const preparedSlots = new Set(Object.keys(preparedPhotos) as ExpectedView[]);
  if (damageDetailPhotos.length) preparedSlots.add("damage_detail");
  // An angle with a damage close-up no longer needs its standard photo.
  const damageCoveredSlots = new Set<ExpectedView>([
    ...damageDetailPhotos.flatMap((photo) => photo.damageView ? [photo.damageView] : []),
    ...(inspectionSession?.photos ?? []).flatMap((photo) => (
      photo.damage_view && photo.capture_status === "accepted" ? [photo.damage_view] : []
    )),
  ]);
  const readySlots = new Set([...preparedSlots, ...damageCoveredSlots]);

  useEffect(() => {
    if (testMode || !inspectionSession || inspectionClosed || anyLoading || saving || inspectionLoading) return;
    const ready = new Set([...inspectionSession.completed_slots, ...readySlots]);
    if (!inspectionSession.required_slots.every((slot) => ready.has(slot))) return;
    const key = `${inspectionSession.id}:${inspectionSession.photos.map((photo) => photo.id).join(",")}:${Object.values(preparedPhotos).map((photo) => photo?.id).join(",")}:${damageDetailPhotos.map((photo) => photo.id).join(",")}`;
    if (autoSubmitKey.current === key) return;
    autoSubmitKey.current = key;
    void saveAllPhotos();
  }, [testMode, inspectionSession, inspectionClosed, anyLoading, saving, inspectionLoading, preparedPhotos, damageDetailPhotos]);

  if (!authSession) {
    return <UserLogin onLoggedIn={(session) => {
      setAuthSession(session);
      setLoginNotice("已登入。");
    }} />;
  }

  return (
    <div className="app-shell">
      {loginNotice && <div className="login-success-toast" role="status" aria-live="polite">
        <span aria-hidden="true">✓</span>
        <div><strong>登入成功</strong><p>{loginNotice}</p></div>
        <button type="button" aria-label="關閉登入成功通知" onClick={() => setLoginNotice(null)}>×</button>
      </div>}
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Guardian 首頁">
          <span className="brand-mark">G</span>
          <span><strong>Guardian</strong><small>智能車況管家</small></span>
        </a>
        <div className="header-actions">
          <MobileAccess />
          <button
            type="button"
            className="theme-toggle"
            aria-label={`切換至${theme === "dark" ? "淺色" : "深色"}主題`}
            title={`切換至${theme === "dark" ? "淺色" : "深色"}主題`}
            aria-pressed={theme === "dark"}
            onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}
          >
            <span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span>
          </button>
          <span className="prototype-badge"><i /> Local AI</span>
          <strong className="user-account-name">{authSession.principal.display_name}</strong>
          <button type="button" className="logout-button" disabled={pickupSending || saving} onClick={() => {
            void logoutUser();
            window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
            setInspectionSession(null);
            setRentalContext(null);
            clearAllPreparedPhotos();
            setTestMode(false);
            setTestCompleted(false);
            realCaptureRef.current = null;
            pickupJobRef.current = null;
            returnDraftRef.current = null;
            setAuthSession(null);
          }}>登出</button>
        </div>
      </header>

      <main id="top">
        {!testMode && inspectionClosed && inspectionStage === "return" ? <section className="inspection-completion" role="status">
          <p className="eyebrow">還車完成</p>
          <h1>借還車照片已送出</h1>
          <p>租單 {inspectionSession!.rental_id} · 車輛 {inspectionSession!.vehicle_id}</p>
          <p>辨識會在背景繼續，現在可以離開此頁。</p>
          <InspectionProgress session={inspectionSession!} preparedSlots={readySlots}
            submitting={inspectionLoading} onSubmitAll={() => void saveAllPhotos()} onStartNew={startNewInspection} />
          {inspectionError && <div className="error-box">{inspectionError}</div>}
        </section> : <>
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">CAPTURE COACH · 即時防呆</p>
            <h1>拍對一次，<br /><em>安心{inspectionStage === "pickup" ? "借車" : "還車"}。</em></h1>
            <p className="lead">上傳前先攔截模糊、過暗、角度與構圖問題，讓每張照片都成為可信的車況證據。</p>
          </div>
        </section>

        <section className="workspace">
          <div className="capture-panel">
            <p className="eyebrow" role="status">{inspectionStage === "pickup" ? "借車拍攝" : "還車拍攝"}</p>
            {(pickupSending || pickupSendError) && <div className="inspection-status" role="status">
              {pickupSending && <p>借車照片正在背景送出，可繼續拍攝還車照片。</p>}
              {pickupSendError && <>
                <div className="error-box"><strong>借車照片尚未送出完成</strong><span>{pickupSendError}</span></div>
                <button type="button" disabled={pickupSending || saving} onClick={() => void sendPickupInBackground()}>重試借車送出</button>
                <button type="button" disabled={pickupSending || anyLoading || saving} onClick={restorePickupForCorrection}>返回借車修正</button>
              </>}
            </div>}
            {(inspectionLoading || inspectionError) && <div className="inspection-status">
              {inspectionLoading && <p role="status">正在開始拍攝…</p>}
              {!inspectionSession && inspectionError && !inspectionLoading && (
                <button type="button" className="inspection-start" onClick={() => void startInspection()}>重試</button>
              )}
              {inspectionError && (
                <div className="error-box"><strong>流程尚未更新</strong><span>{inspectionError}</span></div>
              )}
            </div>}

            <div className="section-heading capture-position-heading">
              <div><p className="eyebrow">STEP 1</p><h2>選擇拍攝位置</h2></div>
              <div className="capture-position-actions">
              {!testMode && inspectionStage === "return" && <button type="button" className="capture-test-toggle"
                disabled={anyLoading || saving || Boolean(pickupJobRef.current) || inspectionLoading || contextLoading}
                onClick={() => void startInspection(true)}>重新開始借車</button>}
              <button type="button" className="capture-test-toggle" aria-pressed={testMode}
                disabled={anyLoading || saving || Boolean(pickupJobRef.current) || inspectionLoading || contextLoading}
                onClick={toggleTestMode}>{testMode ? "結束測試" : "測試辨識"}</button>
              {inspectionSession && <InspectionProgressSummary session={inspectionSession} preparedSlots={readySlots} />}
              </div>
            </div>
            {testMode && <div className="capture-test-banner" role="status">
              <strong>辨識測試模式</strong>
              <p>使用正式流程的角度、構圖與車牌檢查；{inspectionStage === "return" && "還車照片會接續前座與後座髒污專用模型、同角度借還車損比較；"}車損特寫需選擇角度，該角度不跑四角與車牌辨識，還車時立即與借車同角度照片比較。請先完成借車測試以提供比較基準。照片僅暫存在此頁，不保存、不加入訓練、不送入後台紀錄。</p>
              <div className="capture-test-stages">
                <button type="button" aria-pressed={inspectionStage === "pickup"}
                  disabled={anyLoading || saving} onClick={() => selectTestStage("pickup")}>測試借車</button>
                <button type="button" aria-pressed={inspectionStage === "return"}
                  disabled={anyLoading || saving} onClick={() => selectTestStage("return")}>測試還車</button>
              </div>
            </div>}
            <p className="selection-hint">點選要拍攝的角度，紅色卡片代表目前選擇。</p>
            <div className="view-tabs" role="list" aria-label="拍攝位置">
              {availableViews.map((view) => {
                const slotState = inspectionSlotState(
                  inspectionSession,
                  expectedViews[view],
                  preparedSlots,
                  damageCoveredSlots,
                );
                const slot = expectedViews[view];
                const capture = captureResults.current[slot];
                const borderState = pendingSlots.has(slot) ? null
                  : capture?.error || capture?.quality?.status === "fail" ? "failed"
                  : capture?.quality || ["prepared", "accepted", "covered"].includes(slotState) ? "passed"
                  : slotState === "retry" ? "failed" : null;
                return (
                  <button
                    className={`view-tab ${borderState ? `view-tab--${borderState}` : ""} ${
                      selectedView === view ? "view-tab--active" : ""
                    }`}
                    key={view}
                    onClick={() => selectView(view)}
                    type="button"
                    disabled={saving || inspectionClosed}
                    aria-pressed={selectedView === view}
                    aria-label={`${view}${borderState === "passed" ? "，通過" : borderState === "failed" ? "，不通過" : ""}`}
                  >
                    {view}
                  </button>
                );
              })}
            </div>

            {inspectionSession && (
              <InspectionProgress
                session={inspectionSession}
                preparedSlots={readySlots}
                submitting={saving || anyLoading}
                onSubmitAll={() => void saveAllPhotos()}
                onStartNew={startNewInspection}
                testMode={testMode}
              />
            )}

            <div className="section-heading upload-heading">
              <div><p className="eyebrow">STEP 2</p><h2>拍攝或上傳照片</h2></div>
              <span>{selectedView}</span>
            </div>

            {isDamageDetail && <div className="damage-angle-picker" role="group" aria-label="車損角度">
              <p><strong>車損角度</strong><span>選擇損傷所在角度；該角度不需再拍一般照片，也不進行四角與車牌辨識。{testMode
                ? inspectionStage === "return" ? "上傳後立即與借車同角度照片比對。" : "還車測試時會用來比對。"
                : "還車完成後才會一併送後台做同角度車損比對。"}</span></p>
              <div>
                {damageAngleViews.map((view) => <button type="button" key={view}
                  aria-pressed={damageView === expectedViews[view]}
                  disabled={loading || saving || inspectionClosed}
                  onClick={() => { setDamageView(expectedViews[view]); setError(null); }}>{view}</button>)}
              </div>
            </div>}

            {cameraOpen && inspectionSession && !inspectionClosed && <CameraCapture
              key={selectedView}
              expectedView={expectedViews[selectedView]}
              captureDisabled={loading || saving || damageUploadBlocked}
              onCapture={(photo) => isDamageDetail ? analyzeDamageDetails([photo]) : analyze(photo)}
              onClose={() => setCameraOpen(false)}
              onChooseFile={() => { setCameraOpen(false); inputRef.current?.click(); }}
            />}
            <div
              hidden={cameraOpen}
              className={`dropzone ${isDamageDetail ? "dropzone--damage-multiple" : ""} ${(isDamageDetail ? damageDetailPhotos.length > 0 : preview) ? "dropzone--preview" : ""} ${
                !inspectionSession || inspectionClosed ? "dropzone--disabled" : ""
              }`}
              onClick={(event) => event.target !== inputRef.current && !cameraOpen && !loading && !saving && inspectionSession && !inspectionClosed
                && !damageUploadBlocked
                && inputRef.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onDrop}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.target === event.currentTarget && event.key === "Enter" && inspectionSession && !inspectionClosed
                  && !damageUploadBlocked) {
                  if (!loading && !saving) inputRef.current?.click();
                }
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                multiple={isDamageDetail}
                onChange={onFileChange}
                disabled={!inspectionSession || inspectionClosed || loading || saving}
                hidden
              />
              {isDamageDetail && damageDetailPhotos.length > 0 ? (
                <div className="damage-detail-preview-grid" onClick={(event) => event.stopPropagation()}>
                  {damageDetailPhotos.map((prepared, index) => <PreparedDamagePhoto
                    key={prepared.id}
                    prepared={prepared}
                    index={index}
                    disabled={loading || saving}
                    onRemove={() => removeDamageDetailPhoto(prepared.id)}
                  />)}
                  {damageDetailPhotos.length < MAX_DAMAGE_DETAIL_PHOTOS && <button
                    type="button"
                    className="damage-detail-add"
                    disabled={loading || saving || !damageView}
                    onClick={() => inputRef.current?.click()}
                  >
                    <strong>＋ 繼續新增{damageView && ` · ${viewTypesByExpected[damageView]}`}</strong>
                    <span>{damageDetailPhotos.length} / {MAX_DAMAGE_DETAIL_PHOTOS} 張</span>
                  </button>}
                </div>
              ) : preview ? (
                <>
                  <CleanlinessPhoto src={preview} alt="待檢查的車況照片"
                    evidence={result?.analyses?.find((analysis) => analysis.task === "interior_cleanliness")?.result.assessment?.evidence}
                    plate={result?.plate_prediction} />
                  {result && <QualityResult result={result} />}
                  {loading && <div className="camera-capture__notice" role="status">{isInterior ? "正在辨識髒汙與雜物…" : "正在評分…"}</div>}
                </>
              ) : (
                <div className="dropzone-copy">
                  <div className="camera-icon"><span /></div>
                  <h3>
                    {!inspectionSession
                      ? "正在準備拍攝流程"
                      : inspectionClosed
                        ? "本次拍攝流程已完成"
                        : isDamageDetail
                          ? damageView ? `上傳${viewTypesByExpected[damageView]}車損特寫，最多 4 張` : "請先選擇車損角度"
                          : "對齊車身與車牌，輕鬆完成拍攝"}
                  </h3>
                  <p>
                    {!inspectionSession
                      ? "確認租單與車輛後，系統才會保存每個拍攝位置的進度"
                      : inspectionClosed
                        ? testMode ? "可重新測試或回到正式流程" : "完整 AI 可在背景繼續，不需要留在此頁等待"
                        : isDamageDetail
                          ? "該角度不需再拍一般照片；可一次選取多張，包含全景、特寫與斜角補拍"
                          : "開啟相機後依半透明框對齊，系統會自動尋找車身與車牌"}
                  </p>
                  <div className="capture-actions">
                    <button type="button" disabled={!inspectionSession || inspectionClosed || loading || saving || damageUploadBlocked} onClick={(event) => { event.stopPropagation(); setCameraOpen(true); }}>開啟相機</button>
                    <button type="button" disabled={!inspectionSession || inspectionClosed || loading || saving || damageUploadBlocked} onClick={(event) => { event.stopPropagation(); inputRef.current?.click(); }}>{isDamageDetail ? "選擇最多 4 張" : "選擇照片"}</button>
                  </div>
                  <small>支援 JPG、PNG、WebP · 每張最大 15MB{isDamageDetail ? " · 最多 4 張" : ""}</small>
                </div>
              )}
              <span className="corner corner--tl" /><span className="corner corner--tr" />
              <span className="corner corner--bl" /><span className="corner corner--br" />
            </div>

            {!isDamageDetail && result && <CaptureAnalysisResults result={result} />}
            <div className="capture-recognition-time" aria-label="本張照片辨識時間">
              {isDamageDetail && captureTimings.damage_detail && <small>{captureTimings.damage_detail.filename}</small>}
              <RecognitionTimer timing={captureTimings[expectedViews[selectedView]]} durationMs={result?.recognition_ms} />
            </div>
            {isDamageDetail && damageDetailPhotos.map((photo, index) => <CaptureAnalysisResults key={photo.id} result={photo.quality} label={`車損特寫 ${index + 1}`} />)}

            {!isDamageDetail && file && <div className="file-row">
              <span>{file.name}</span>
              <div>
                <button type="button" disabled={loading || saving} onClick={() => setCameraOpen(true)}>重新拍攝</button>
                <button type="button" disabled={loading || saving} onClick={() => inputRef.current?.click()}>重新選擇照片</button>
                <button type="button" disabled={loading || saving} onClick={removeSelectedPhoto}>移除</button>
              </div>
            </div>}
            {loading && !cameraOpen && !preview && <div className="analyzing" role="status"><span /><p><strong>正在評分…</strong></p></div>}
            {((testMode && !isDamageDetail && file && result && result.status !== "fail")
              || (isDamageDetail && damageDetailPhotos.length > 0)) && (
              <div className="save-action">
                <div><strong>{isDamageDetail ? `已準備 ${damageDetailPhotos.length} 張車損特寫` : "此角度已準備完成"}</strong><small>{testMode ? "此照片僅供辨識測試；完成測試不會儲存或送入後台。" : isDamageDetail ? "已選角度不需再拍一般照片；完成其餘必拍位置後會自動送出，還車完成才送後台比對。" : "完成必拍位置後，特寫會自動送出。"}</small></div>
                <span className="prepared-badge">{isDamageDetail ? `${damageDetailPhotos.length} / ${MAX_DAMAGE_DETAIL_PHOTOS} 張` : "測試完成"}</span>
              </div>
            )}
            {saveNotice && <div className="save-notice" role="status" aria-live="polite"><strong>{saveNotice}</strong></div>}
            {error && <div className="error-box"><strong>分析失敗</strong><span>{error}</span></div>}
          </div>

          <aside className="guidance-panel">
            <p className="eyebrow">拍攝小教練</p>
            <h2>{selectedView}怎麼拍？</h2>
            {!isDamageDetail && (selectedGuide ? <>
              <div className="fixed-reference-guide">
                <div className="fixed-reference-guide__heading">
                  <strong>{selectedView}固定正確範本</strong>
                  <span>跟著範本構圖即可</span>
                </div>
                <figure className="fixed-reference-card fixed-reference-card--single">
                  <ZoomableImage
                    className="fixed-reference-zoom"
                    src={selectedGuide.src}
                    alt={`${selectedView}正確拍攝範本`}
                    title={`${selectedView}固定正確範本`}
                    overlay={<span>⌕ 點擊放大範本</span>}
                  />
                  <figcaption><strong>{selectedView}</strong><span>{selectedGuide.sourceLabel} · {selectedGuide.detail}</span></figcaption>
                </figure>
              </div>
              {referenceSamples[0] && <div className="car-guide car-guide--reference">
                <ZoomableImage
                  src={learningImageUrl(referenceSamples[0].id)}
                  alt={`${selectedView}後台核准拍攝範例`}
                  title={`${selectedView}後台核准拍攝範例`}
                  overlay={<span>⌕ 點擊放大範例</span>}
                />
                <span>後台人工核准範例 · 品質 {referenceSamples[0].score ?? "—"} 分</span>
              </div>}
            </> : <div className="car-guide" aria-hidden="true">
              <div className="car-shape"><span className="windshield" /><span className="plate" /></div>
              <span className="guide-ray guide-ray--one" /><span className="guide-ray guide-ray--two" />
            </div>)}
            <ol className="tips">
              {isDamageDetail ? <>
                <li><span>01</span><p><strong>損傷佔畫面 20–60%</strong>同時保留周邊板件作為位置依據</p></li>
                <li><span>02</span><p><strong>斜角補拍</strong>凹陷請稍微側拍，讓變形與反光線可見</p></li>
                <li><span>03</span><p><strong>避免閃光直射</strong>反光可能遮住細刮痕或被誤認為刮痕</p></li>
                <li><span>04</span><p><strong>先拍全景再拍特寫</strong>便於後台確認位置與嚴重度</p></li>
              </> : isInterior ? <>
                <li><span>01</span><p><strong>座椅與腳踏區</strong>同時拍入椅面、椅背與地板</p></li>
                <li><span>02</span><p><strong>移開私人物品</strong>讓垃圾、污漬與碎屑清楚可見</p></li>
                <li><span>03</span><p><strong>打開車內燈</strong>避免陰影遮住座椅或地毯</p></li>
                <li><span>04</span><p><strong>廣角穩定拍攝</strong>勿貼近單一污點或遮住鏡頭</p></li>
              </> : <>
                <li><span>01</span><p><strong>主要車身清楚</strong>允許少量裁切，不必為了塞滿引導框反覆重拍</p></li>
                <li><span>02</span><p><strong>完整車牌清楚入鏡</strong>開啟相機即自動辨識，請讓車牌對齊黃色框並保持清楚</p></li>
                <li><span>03</span><p><strong>穩定拍攝</strong>停穩手機一秒再按快門</p></li>
                <li><span>04</span><p><strong>避開逆光</strong>夜間請開啟環境照明</p></li>
              </>}
            </ol>
            <div className="privacy-note"><span>◎</span><p><strong>隱私保護</strong>照片與模型分析在本機處理。</p></div>
          </aside>
        </section>

        <DamageComparisonPanel />
        {inspectionStage === "return" && <InteriorCleanlinessPanel />}
        </>}
      </main>

      <footer><span>Guardian · Hackathon Project</span><span>Human-in-the-loop · 不自動判責</span></footer>
    </div>
  );
}
