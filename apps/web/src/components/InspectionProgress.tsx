import type { ExpectedView, InspectionSession } from "../lib/api";

interface InspectionProgressProps {
  session: InspectionSession;
  preparedSlots: Set<ExpectedView>;
  submitting: boolean;
  onSubmitAll: () => void;
  onStartNew: () => void;
  testMode?: boolean;
}

export function InspectionProgressSummary({ session, preparedSlots }: {
  session: InspectionSession;
  preparedSlots: Set<ExpectedView>;
}) {
  const readySlots = new Set([...session.completed_slots, ...preparedSlots]);
  const completed = session.required_slots.filter((slot) => readySlots.has(slot)).length;
  const required = session.required_slots.length;
  const progress = required ? Math.round((completed / required) * 100) : 0;
  return <div className="capture-progress-summary">
    <span aria-live="polite">{completed} / {required} 完成</span>
    <div className="capture-progress-summary__bar" role="progressbar"
      aria-label="必要拍攝位置完成進度" aria-valuemin={0} aria-valuemax={required}
      aria-valuenow={completed}>
      <i style={{ width: `${progress}%` }} />
    </div>
  </div>;
}

export function InspectionProgress({
  session,
  preparedSlots,
  submitting,
  onSubmitAll,
  onStartNew,
  testMode = false,
}: InspectionProgressProps) {
  const readySlots = new Set([...session.completed_slots, ...preparedSlots]);
  const isComplete = session.status === "capture_completed";
  const nextSlot = session.required_slots.find((slot) => !readySlots.has(slot));
  const activeAnalyses = session.photos.filter(
    (photo) => ["queued", "processing"].includes(photo.analysis_status),
  ).length;
  const reviewAnalyses = session.photos.filter(
    (photo) => ["needs_review", "failed"].includes(photo.analysis_status),
  ).length;
  const completedAnalyses = session.photos.filter(
    (photo) => photo.analysis_status === "completed",
  ).length;
  if (!isComplete && nextSlot) return null;
  return (
    <section className={`inspection-progress ${isComplete ? "inspection-progress--complete" : ""}`}>
      {(isComplete || !nextSlot) && <div className="inspection-progress__footer">
        <p>
          {testMode
            ? isComplete ? "本次辨識測試完成，照片與結果未送入後台。" : "必拍角度已完成，可以完成測試；不會儲存或送出照片。"
            : isComplete
            ? session.stage === "return"
              ? `已送出；系統正以同租單、同車、同視角的借車照自動比對。背景處理 ${activeAnalyses} 張、完成 ${completedAnalyses} 張、待後台覆核 ${reviewAnalyses} 張。`
              : "借車照片已保存；完成還車後才會連同還車照片一併送入後台辨識。"
            : session.stage === "return"
              ? "必拍位置已通過；按下完成後，借車與還車照片才會一併送入後台辨識。"
              : "必拍位置已通過；借車照片只保存，完成還車後才送入後台。"}
        </p>
        {isComplete ? (
          <button type="button" onClick={onStartNew} disabled={submitting}>{submitting ? "正在開始…" : testMode ? "重新測試" : session.stage === "return" ? "重新開始借車" : "開始新的流程"}</button>
        ) : !nextSlot ? (
          <button type="button" onClick={onSubmitAll} disabled={submitting}>
            {testMode ? "完成測試（不送後台）" : submitting ? "正在送出…" : "完成流程／重試送出"}
          </button>
        ) : null}
      </div>}
    </section>
  );
}
