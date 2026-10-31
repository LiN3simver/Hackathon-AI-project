import { useEffect, useState } from "react";
import type { RecognitionJobProgress } from "../lib/api";

const taskLabels: Record<string, string> = {
  plate_recognition: "車牌 OCR", interior_cleanliness: "整潔辨識", damage_comparison: "車損比對",
};
const slotLabels: Record<string, string> = {
  left_front: "左前方", right_front: "右前方", left_rear: "左後方", right_rear: "右後方",
  interior_front: "前座", interior_rear: "後座", damage_detail: "特寫",
};
const statusLabels: Record<string, string> = {
  queued: "排隊中", processing: "辨識中", completed: "完成", needs_review: "待覆核", resolved: "已覆核", failed: "失敗",
};

export function jobSlotLabel(job: RecognitionJobProgress, labels: Record<string, string> = slotLabels): string {
  const label = labels[job.slot] ?? job.slot;
  return job.damage_view ? `${label}·${labels[job.damage_view] ?? job.damage_view}` : label;
}

export function jobDurationMs(job: RecognitionJobProgress, now: number): number {
  const accumulated = job.duration_ms ?? 0;
  if (job.status !== "processing" || !job.started_at) return accumulated;
  if (!job.heartbeat_at) return accumulated;
  const heartbeat = Date.parse(job.heartbeat_at);
  // Stop projecting once the worker misses its next heartbeat.
  return accumulated + (Number.isFinite(heartbeat) ? Math.min(5000, Math.max(0, now - heartbeat)) : 0);
}

export function RecognitionJobList({ jobs, showSlot = false }: { jobs: RecognitionJobProgress[]; showSlot?: boolean }) {
  const [now, setNow] = useState(Date.now());
  const active = jobs.some((job) => job.status === "processing");
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active]);
  if (!jobs.length) return null;
  return <ul className="recognition-job-list" aria-label="各項辨識狀態與耗時">
    {jobs.map((job) => <li key={job.id}>
      <span>{showSlot && `${jobSlotLabel(job)} · `}{taskLabels[job.task] ?? job.task}</span>
      <strong>{job.model_pending ? "尚未啟用" : statusLabels[job.status] ?? job.status}</strong>
      <span>{job.model_pending ? "未執行模型" : job.status === "queued"
        ? job.duration_ms != null ? `已辨識 ${(job.duration_ms / 1000).toFixed(1)}s` : "等待辨識"
        : job.status === "processing" || job.duration_ms != null
          ? `${(jobDurationMs(job, now) / 1000).toFixed(1)}s` : "未紀錄耗時"}</span>
    </li>)}
  </ul>;
}
