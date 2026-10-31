import { useEffect, useState } from "react";

export type RecognitionTiming = { start: number; durationMs?: number; filename: string };

export function RecognitionTimer({ timing, durationMs }: { timing?: RecognitionTiming; durationMs?: number | null }) {
  const [now, setNow] = useState(performance.now());
  const running = timing && timing.durationMs == null;
  useEffect(() => {
    if (!running) return;
    setNow(performance.now());
    const timer = window.setInterval(() => setNow(performance.now()), 100);
    return () => window.clearInterval(timer);
  }, [running, timing?.start]);
  const measured = timing?.durationMs ?? durationMs;
  if (!timing && measured == null) return null;
  const seconds = Math.max(0, running ? now - timing.start : measured ?? 0) / 1000;
  return <span className={`recognition-timer ${running ? "recognition-timer--running" : ""}`}>
    {running ? "辨識中" : "辨識耗時"} <strong>{seconds.toFixed(1)}s</strong>
  </span>;
}
