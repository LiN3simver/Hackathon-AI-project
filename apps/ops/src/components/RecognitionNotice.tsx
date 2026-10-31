import { useEffect, useState } from "react";

import { getRecognitionProgress, type RecognitionProgress } from "../lib/api";
import { jobDurationMs, RecognitionJobList } from "./RecognitionJobList";

const COLLAPSED_KEY = "guardian-recognition-notice-collapsed";

function duration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  return minutes ? `${minutes} 分 ${String(seconds % 60).padStart(2, "0")} 秒` : `${seconds} 秒`;
}

export function RecognitionNotice() {
  const [items, setItems] = useState<RecognitionProgress[]>([]);
  const [collapsed, setCollapsed] = useState(() => window.localStorage.getItem(COLLAPSED_KEY) === "true");
  const [now, setNow] = useState(Date.now());
  const [error, setError] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);

  useEffect(() => {
    let stopped = false;
    let timer: number;
    async function poll() {
      try {
        const progress = await getRecognitionProgress();
        if (!stopped) { setItems(progress); setError(null); }
      } catch {
        if (!stopped) setError("辨識進度讀取失敗，正在重試。");
      } finally {
        if (!stopped) timer = window.setTimeout(() => void poll(), 2500);
      }
    }
    void poll();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const item = items.find((item) => item.session_id === selectedSession) ?? items[0];
  if (!item) return <aside className="recognition-notice" aria-label="辨識過程通知" role="status">
    <strong>辨識過程通知</strong><p>{error ?? "目前沒有辨識工作；正式照片送出後會在此顯示。"}</p>
  </aside>;

  const active = item.queued + item.processing > 0;
  const percent = Math.round(((item.completed + item.failed) / Math.max(1, item.total)) * 100);
  const elapsedEnd = !active && item.finished_at ? Date.parse(item.finished_at) : now;
  const elapsed = Math.max(0, Math.floor((elapsedEnd - Date.parse(item.started_at)) / 1000));
  const title = active
    ? item.processing > 0 ? "辨識中" : "開始辨識"
    : item.failed > 0 ? "辨識完畢（有失敗）" : "辨識完畢";

  function toggle() {
    setCollapsed((current) => {
      window.localStorage.setItem(COLLAPSED_KEY, String(!current));
      return !current;
    });
  }

  return <aside className={`recognition-notice ${collapsed ? "recognition-notice--collapsed" : ""}`}
    aria-label="辨識過程通知" aria-live="polite">
    <div className="recognition-notice-heading">
      <span className={`recognition-notice-indicator ${active ? "recognition-notice-indicator--active" : ""}`} aria-hidden="true" />
      <strong>{title}</strong>
      <span className="recognition-notice-percent">{percent}%</span>
      <button type="button" onClick={toggle} aria-expanded={!collapsed}
        aria-label={collapsed ? "展開辨識通知" : "縮小辨識通知"}>{collapsed ? "▾" : "▴"}</button>
    </div>
    {!collapsed && <div className="recognition-notice-details">
      {error && <p role="status">{error}</p>}
      {items.length > 1 && <select aria-label="選擇辨識流程" value={item.session_id} onChange={(event) => setSelectedSession(event.target.value)}>
        {items.map((flow) => <option key={flow.session_id} value={flow.session_id}>
          {flow.vehicle_id} · {flow.stage === "pickup" ? "借車" : flow.stage === "return" ? "還車" : "車損"} · {flow.queued + flow.processing ? "處理中" : "已完成"}
        </option>)}
      </select>}
      <p>{item.vehicle_id} · {item.stage === "damage_pair" ? "車損比對" : `${item.stage === "pickup" ? "取車" : "還車"}辨識`}
        {items.length > 1 && <span> · 另有 {items.length - 1} 個流程</span>}</p>
      <div className="recognition-notice-track" role="progressbar" aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={percent} aria-label="辨識進度"><i style={{ width: `${percent}%` }} /></div>
      <div className="recognition-notice-meta">
        <span>完成 {item.completed + item.failed} / {item.total} 項
          {item.queued > 0 ? ` · 排隊 ${item.queued}` : ""}
          {item.processing > 0 ? ` · 處理 ${item.processing}` : ""}</span>
        <span>{item.jobs?.length ? `辨識累計 ${(item.jobs.reduce((sum, job) => sum + jobDurationMs(job, now), 0) / 1000).toFixed(1)}s` : `流程時間 ${duration(elapsed)}`}</span>
      </div>
      <RecognitionJobList jobs={item.jobs ?? []} showSlot />
      {!active && <small>耗時已保留於租還車照片紀錄。</small>}
    </div>}
  </aside>;
}
