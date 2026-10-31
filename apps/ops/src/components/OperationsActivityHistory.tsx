import { useEffect, useState } from "react";
import {
  clearExpiredOperationsActivity,
  downloadOperationsExcel,
  getOperationsActivity,
  type OperationsActivity,
} from "../lib/api";
import { confirmDeletion } from "../lib/confirmDeletion";

// "all" is the admin audit trail; "mine" is a staff member's own operations (the API scopes it by account).
export function OperationsActivityHistory({ scope, onNotice }: {
  scope: "all" | "mine"; onNotice?: (message: string) => void;
}) {
  const isAll = scope === "all";
  const [records, setRecords] = useState<OperationsActivity[]>([]);
  const [archived, setArchived] = useState(false);
  const [more, setMore] = useState(false);
  const [busy, setBusy] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setBusy(true); setError(null);
    void getOperationsActivity(0, archived).then((items) => {
      if (!cancelled) { setRecords(items); setMore(items.length === 30); }
    }).catch((caught: unknown) => {
      if (!cancelled) setError(caught instanceof Error ? caught.message : "讀取操作紀錄失敗");
    }).finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [archived, reloadKey]);

  async function loadMore() {
    setBusy(true); setError(null);
    try {
      const items = await getOperationsActivity(0, archived, records.at(-1));
      setRecords((current) => {
        const ids = new Set(current.map((item) => item.id));
        return [...current, ...items.filter((item) => !ids.has(item.id))];
      });
      setMore(items.length === 30);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "讀取紀錄失敗"); }
    finally { setBusy(false); }
  }

  async function exportAll() {
    setExporting(true); setError(null);
    try { await downloadOperationsExcel(); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "匯出失敗"); }
    finally { setExporting(false); }
  }

  async function clearOldActivity() {
    if (!confirmDeletion(
      "超過 180 天的營運操作時間軸紀錄會先封存再清除；之後仍可查看封存紀錄或匯出 Excel。"
    )) return;
    setClearing(true); setError(null);
    try {
      const result = await clearExpiredOperationsActivity();
      onNotice?.(`已封存並清除 ${result.deleted_activity_events} 筆舊操作紀錄，可繼續查核與匯出。`);
      setReloadKey((current) => current + 1);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "清除舊操作紀錄失敗"); }
    finally { setClearing(false); }
  }

  return <section className="activity-panel" id="operations-activity">
    <div className="ops-title">
      <div><p className="eyebrow">{isAll ? "AUDIT TRAIL" : "MY ACTIVITY"}</p><h2>{isAll ? "營運操作時間軸" : "我的操作紀錄"}</h2></div>
      {isAll && <div className="ops-title-actions">
        <button className="clear-test-data" type="button" disabled={clearing} onClick={() => void clearOldActivity()}>
          {clearing ? "清除中…" : "清除超過 180 天紀錄"}
        </button>
      </div>}
    </div>
    <div className="activity-history-controls">
      <span>{archived ? "含歷史封存" : "最近紀錄"} · 已顯示 {records.length} 筆</span>
      <label><input type="checkbox" checked={archived} disabled={busy}
        onChange={(event) => setArchived(event.target.checked)} />查看封存紀錄</label>
      <button type="button" disabled={exporting} onClick={() => void exportAll()}>
        {exporting ? "匯出中…" : isAll ? "匯出全部 Excel" : "匯出我的紀錄 Excel"}
      </button>
    </div>
    <p className="activity-history-note">{isAll ? "" : "只顯示你本人帳號的營運操作；"}預設顯示最近 30 筆；180 天前的紀錄清除前會封存，匯出包含全部現存及封存紀錄。歷史資料快照僅代表現存資料，無法還原未保存的歷次操作。</p>
    {error && <p className="error-box" role="alert">{error}</p>}
    {!records.length && !busy && <div className="empty-state">尚無操作紀錄</div>}
    <ol className="activity-list">
      {records.map((item) => <li key={item.id}><i /><div>
        <strong>{item.summary}</strong>
        <small>{item.actor} · {item.entity_type} · {item.entity_id} · {item.action}</small>
        <details><summary>查核詳細資料</summary><pre>{JSON.stringify(item.metadata, null, 2)}</pre></details>
      </div><time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString("zh-TW")}</time></li>)}
    </ol>
    {more && <button type="button" disabled={busy} onClick={() => void loadMore()}>{busy ? "讀取中…" : "載入更多紀錄"}</button>}
  </section>;
}
