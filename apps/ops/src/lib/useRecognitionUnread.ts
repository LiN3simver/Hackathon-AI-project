import { useCallback, useEffect, useMemo, useState } from "react";

import { getRecognitionProgress, type RecognitionProgress } from "./api";

function recordIds(flows: RecognitionProgress[]): string[] {
  return flows.flatMap((flow) => flow.jobs?.length
    ? flow.jobs.map((job) => `job:${job.id}`)
    : [`flow:${flow.session_id}`]);
}

export function useRecognitionUnread(subject: string) {
  const storageKey = `guardian-recognition-seen:${subject}`;
  const [flows, setFlows] = useState<RecognitionProgress[]>([]);
  const [seen, setSeen] = useState<Set<string>>(() => {
    try {
      const saved: unknown = JSON.parse(window.localStorage.getItem(storageKey) ?? "[]");
      return new Set(Array.isArray(saved) ? saved.filter((id): id is string => typeof id === "string") : []);
    } catch { return new Set(); }
  });

  useEffect(() => {
    let stopped = false;
    let timer: number;
    async function poll() {
      try {
        const progress = await getRecognitionProgress(50);
        if (!stopped) setFlows(progress);
      } catch {
        // Keep the last known count during a temporary connection failure.
      } finally {
        if (!stopped) timer = window.setTimeout(() => void poll(), 2500);
      }
    }
    void poll();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, []);

  const markViewed = useCallback((records: RecognitionProgress[]) => {
    const ids = recordIds(records);
    setSeen((current) => {
      if (ids.every((id) => current.has(id))) return current;
      return new Set([...current, ...ids]);
    });
  }, []);

  useEffect(() => {
    try { window.localStorage.setItem(storageKey, JSON.stringify([...seen])); }
    catch { /* In-memory tracking remains available when storage is disabled. */ }
  }, [seen, storageKey]);

  const count = useMemo(() => new Set(recordIds(flows).filter((id) => !seen.has(id))).size, [flows, seen]);
  return { count, markViewed };
}
