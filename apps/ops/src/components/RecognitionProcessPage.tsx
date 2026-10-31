import { useEffect, useMemo, useState } from "react";

import {
  authenticatedAssetUrl, getInspectionPhotoRecords, getRecognitionProgress,
  type InspectionPhotoRecord, type RecognitionJobProgress, type RecognitionProgress,
} from "../lib/api";
import { ZoomableImage } from "./ImageViewer";
import { PhotoListControls, usePhotoListPager } from "./PhotoListPager";
import { jobDurationMs, jobSlotLabel, RecognitionJobList } from "./RecognitionJobList";

const slotLabels: Record<string, string> = {
  left_front: "左前方", right_front: "右前方", left_rear: "左後方", right_rear: "右後方",
  interior_front: "前座", interior_rear: "後座", damage_detail: "車損特寫",
};

function formatTime(value: string): string {
  const time = Date.parse(value);
  return Number.isFinite(time) ? new Date(time).toLocaleString("zh-TW", { hour12: false }) : "—";
}

function groupPhotos(jobs: RecognitionJobProgress[]) {
  const photos = new Map<string, RecognitionJobProgress[]>();
  for (const job of jobs) photos.set(job.photo_id, [...(photos.get(job.photo_id) ?? []), job]);
  return [...photos.entries()];
}

export function RecognitionProcessPage({ onRecordsViewed }: { onRecordsViewed?: (flows: RecognitionProgress[]) => void }) {
  const [flows, setFlows] = useState<RecognitionProgress[]>([]);
  const [photos, setPhotos] = useState<InspectionPhotoRecord[]>([]);
  const [filter, setFilter] = useState<"all" | "active" | "finished">("all");
  const [query, setQuery] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [photoError, setPhotoError] = useState(false);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    let stopped = false;
    let timer: number;
    async function poll() {
      const [progress, records] = await Promise.allSettled([getRecognitionProgress(50), getInspectionPhotoRecords()]);
      if (stopped) return;
      if (progress.status === "fulfilled") {
        setFlows(progress.value); setLoaded(true); setError(null);
        onRecordsViewed?.(progress.value);
      } else setError("辨識進度讀取失敗，正在重試。");
      if (records.status === "fulfilled") { setPhotos(records.value); setPhotoError(false); }
      else setPhotoError(true);
      timer = window.setTimeout(() => void poll(), 2500);
    }
    void poll();
    const clock = window.setInterval(() => setNow(Date.now()), 1000);
    return () => { stopped = true; window.clearTimeout(timer); window.clearInterval(clock); };
  }, [onRecordsViewed]);

  const photoMap = useMemo(() => new Map(photos.map((photo) => [photo.id, photo])), [photos]);
  const totals = flows.reduce((sum, flow) => ({
    queued: sum.queued + flow.queued, processing: sum.processing + flow.processing,
    completed: sum.completed + flow.completed, failed: sum.failed + flow.failed,
  }), { queued: 0, processing: 0, completed: 0, failed: 0 });
  const needle = query.trim().toLocaleLowerCase();
  const visible = flows.filter((flow) => {
    const active = flow.queued + flow.processing > 0;
    return (filter === "all" || (filter === "active" ? active : !active))
      && (!needle || [flow.rental_id, flow.vehicle_id, flow.session_id, ...(flow.jobs ?? []).map((job) => job.filename)]
        .join(" ").toLocaleLowerCase().includes(needle));
  });
  // Page by rental: the three most recent rentals show every flow (借車、還車、比對); the rest are behind "展開".
  const rentals = [...visible.reduce((groups, flow) =>
    groups.set(flow.rental_id, [...(groups.get(flow.rental_id) ?? []), flow]), new Map<string, RecognitionProgress[]>()).values()];
  const pager = usePhotoListPager(rentals, 10, false, 3);

  return <section className="recognition-process-page ops-section" id="recognition-progress">
    <div className="recognition-process-heading">
      <div><p className="eyebrow">辨識紀錄</p><h1>辨識過程與紀錄</h1>
        <p>查看正式照片的排隊、辨識狀態與耗時。每 2.5 秒更新；辨識秒數不含排隊等待。</p></div>
      <span className="recognition-process-update">{error ? "連線重試中" : loaded ? "自動更新中" : "讀取中…"}</span>
    </div>
    <div className="recognition-process-stats">
      {([["queued", "排隊中"], ["processing", "辨識中"], ["completed", "已完成"], ["failed", "失敗"]] as const)
        .map(([key, label]) => <div key={key}><span>{label}</span><strong>{totals[key]}</strong></div>)}
    </div>
    <div className="recognition-process-controls">
      <div className="photo-record-filters" role="group" aria-label="辨識狀態篩選">
        {([["all", "全部"], ["active", "處理中"], ["finished", "已結束"]] as const).map(([value, label]) =>
          <button key={value} type="button" className={filter === value ? "active" : ""}
            aria-pressed={filter === value} onClick={() => { setFilter(value); pager.reset(); }}>{label}</button>)}
      </div>
      <input aria-label="搜尋辨識紀錄" placeholder="搜尋車輛、租單或檔名" value={query} onChange={(event) => { setQuery(event.target.value); pager.reset(); }} />
    </div>
    {error && <p className="error-box" role="status">{error}</p>}
    {photoError && <p role="status">照片資訊暫時讀取失敗；辨識狀態仍會持續更新。</p>}
    {loaded && !visible.length && <div className="empty-state">
      {flows.length ? "沒有符合篩選條件的辨識紀錄。" : "目前沒有辨識工作；正式照片送出後會顯示在這裡。測試照片不送入後台。"}
    </div>}
    <div className="recognition-process-flows">
      {pager.visibleItems.flat().map((flow) => {
        const active = flow.queued + flow.processing > 0;
        const percent = Math.round((flow.completed + flow.failed) / Math.max(1, flow.total) * 100);
        const jobs = flow.jobs ?? [];
        const duration = jobs.length ? jobs.reduce((sum, job) => sum + jobDurationMs(job, now), 0)
          : Math.max(0, (flow.finished_at ? Date.parse(flow.finished_at) : now) - Date.parse(flow.started_at));
        return <article className="recognition-process-flow" key={flow.session_id}>
          <header><div><h2>{flow.vehicle_id} · {flow.stage === "pickup" ? "借車" : flow.stage === "return" ? "還車" : "車損比對"}</h2>
            <p>租單 {flow.rental_id} · {formatTime(flow.started_at)}</p></div>
            <strong>{active ? flow.processing ? "辨識中" : "排隊中" : flow.failed ? "有工作失敗" : "辨識完畢"}</strong></header>
          <div className="recognition-process-flow-meta"><span>完成 {flow.completed + flow.failed} / {flow.total} 項 · {percent}%</span>
            <span>{jobs.length ? "辨識累計" : "流程耗時"} {(duration / 1000).toFixed(1)}s</span></div>
          <div className="recognition-notice-track" role="progressbar" aria-label={`${flow.vehicle_id} 辨識進度`}
            aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><i style={{ width: `${percent}%` }} /></div>
          <ul className="recognition-process-photos" aria-label="照片辨識紀錄">
            {groupPhotos(jobs).map(([photoId, photoJobs]) => {
              const photo = photoMap.get(photoId);
              const first = photoJobs[0];
              return <li className="recognition-process-photo" key={photoId}>
                {photo?.image_available && photo.image_url && <ZoomableImage className="recognition-process-image"
                  src={authenticatedAssetUrl(photo.image_url)} alt={first.filename} title={first.filename} />}
                <div><div className="recognition-process-photo-meta"><strong>{jobSlotLabel(first, slotLabels)}</strong>
                  <span className="recognition-process-filename" title={first.filename}>{first.filename}</span>
                  {photo?.recognized_plate && <span>車牌：{photo.recognized_plate}</span>}</div>
                  <RecognitionJobList jobs={photoJobs} /></div>
              </li>;
            })}
          </ul>
        </article>;
      })}
    </div>
    <PhotoListControls collapsedItemCount={pager.collapsedItemCount} currentPage={pager.currentPage} expanded={pager.expanded}
      itemLabel="租單" onExpandedChange={pager.setExpanded} onPageChange={pager.setPage} totalItems={rentals.length}
      totalPages={pager.totalPages} visibleEnd={pager.visibleEnd} visibleStart={pager.visibleStart} />
    {loaded && flows.length > 0 && <p className="recognition-process-footnote">顯示最近 {flows.length} 個流程，處理中的工作優先。更早的耗時保留在「租還車照片紀錄」。</p>}
  </section>;
}
