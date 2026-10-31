import { useCallback, useEffect, useMemo, useState } from "react";

import {
  authenticatedAssetUrl,
  clearInspectionPhotoRecords,
  deleteInspectionPhotoRecord,
  getInspectionPhotoRecords,
  type InspectionPhotoRecord,
} from "../lib/api";
import { ZoomableImage } from "./ImageViewer";
import { RecognitionJobList } from "./RecognitionJobList";
import { confirmDeletion } from "../lib/confirmDeletion";

const slotLabels: Record<string, string> = {
  left_front: "左前方",
  right_front: "右前方",
  left_rear: "左後方",
  right_rear: "右後方",
  interior_front: "前座車內",
  interior_rear: "後座車內",
  damage_detail: "車損特寫",
};

function recordSlotLabel(record: InspectionPhotoRecord): string {
  const label = slotLabels[record.slot] ?? record.slot;
  return record.damage_view ? `${label} · ${slotLabels[record.damage_view] ?? record.damage_view}` : label;
}

function formatRecordTime(value: string): string {
  return new Intl.DateTimeFormat("zh-TW", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function InspectionPhotoRecords({
  refreshKey = 0,
  canDelete = false,
}: {
  refreshKey?: number;
  canDelete?: boolean;
}) {
  const [records, setRecords] = useState<InspectionPhotoRecord[]>([]);
  const [filter, setFilter] = useState<"all" | "pickup" | "return">("all");
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 1500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const refresh = useCallback(async () => {
    try {
      setRecords(await getInspectionPhotoRecords());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "讀取租還車照片紀錄失敗");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2500);
    return () => window.clearInterval(timer);
  }, [refresh, refreshKey]);

  const visibleRecords = useMemo(
    () => filter === "all" ? records : records.filter((record) => record.stage === filter),
    [filter, records],
  );
  const displayedRecords = expanded ? visibleRecords : visibleRecords.slice(0, 3);

  async function remove(record: InspectionPhotoRecord) {
    if (!confirmDeletion(
      `將永久刪除租單 ${record.rental_id} 的「${record.filename}」照片紀錄與私有原圖，並移除警示對這張照片的連結。其他照片不受影響。此操作無法復原。`
    )) return;
    setBusy(record.id);
    setNotice(null);
    setError(null);
    try {
      const result = await deleteInspectionPhotoRecord(record.id);
      setNotice(`已刪除 ${result.deleted_records} 筆照片。`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "刪除照片紀錄失敗");
    } finally {
      setBusy(null);
    }
  }

  async function clearAll() {
    if (records.length === 0 || !confirmDeletion(
      `將永久刪除全部 ${records.length} 筆租還車照片紀錄及其私有原圖，並移除警示中的照片連結。包含借車與還車，與目前篩選條件無關。此操作無法復原。`
    )) return;
    setBusy("clear-all");
    setNotice(null);
    setError(null);
    try {
      const result = await clearInspectionPhotoRecords();
      setNotice(`已清除 ${result.deleted_records} 筆照片。`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "一鍵清除照片紀錄失敗");
    } finally {
      setBusy(null);
    }
  }

  return <section className="inspection-photo-records" id="inspection-photos">
    <div className="photo-record-heading">
      <div>
        <p className="eyebrow">PRIVATE INSPECTION ARCHIVE</p>
        <h2>租還車照片紀錄</h2>
        <p>兩種營運帳號皆可查看；只有管理員可刪除。點擊照片可放大、縮小與拖曳檢查。</p>
      </div>
      <div className="photo-record-actions">
        <span>共 {records.length} 筆</span>
        {canDelete && <button
          type="button"
          className="danger-outline"
          disabled={records.length === 0 || busy === "clear-all"}
          onClick={() => void clearAll()}
        >{busy === "clear-all" ? "清除中…" : "清除全部租還車照片"}</button>}
      </div>
    </div>
    <div className="photo-record-filters" role="group" aria-label="照片階段篩選">
      {([
        ["all", "全部"],
        ["pickup", "借車"],
        ["return", "還車"],
      ] as const).map(([value, label]) => <button
        key={value}
        type="button"
        className={filter === value ? "active" : ""}
        aria-pressed={filter === value}
        onClick={() => { setFilter(value); setExpanded(false); }}
      >{label}</button>)}
    </div>
    {notice && <div className="ops-notice"><strong>照片紀錄</strong><span>{notice}</span></div>}
    {error && <div className="error-box"><strong>照片紀錄操作失敗</strong><span>{error}</span></div>}
    {visibleRecords.length === 0
      ? <div className="empty-state">目前沒有{filter === "pickup" ? "借車" : filter === "return" ? "還車" : "租還車"}照片紀錄</div>
      : <div className="photo-record-grid">
        {displayedRecords.map((record) => <article className="photo-record-card" key={record.id}>
          {record.image_available && record.image_url
            ? <ZoomableImage
              className="photo-record-image"
              src={authenticatedAssetUrl(record.image_url)}
              alt={`${record.vehicle_id} ${recordSlotLabel(record)}`}
              title={record.filename}
            />
            : <div className="photo-record-image-empty">無保留圖片<br /><small>品質檢查未通過</small></div>}
          <div className="photo-record-body">
            <div className="card-top">
              <strong>{record.stage === "pickup" ? "借車" : "還車"} · {recordSlotLabel(record)}</strong>
              <span>{record.quality_status === "fail" ? "已阻擋 · 不評分" : `基礎品質 ${record.quality_score} 分`}</span>
            </div>
            <h3>{record.vehicle_id}</h3>
            <p title={record.filename}>{record.filename}</p>
            <dl>
              <div><dt>租單</dt><dd>{record.rental_id}</dd></div>
              <div><dt>檢查單</dt><dd>{record.session_id}</dd></div>
              {(record.expected_plate || record.recognized_plate || record.human_verified_plate || record.plate_model_uncertain) && <>
                <div><dt>租單車牌</dt><dd>{record.expected_plate ?? "未設定"}</dd></div>
                <div><dt>模型辨識</dt><dd>{record.recognized_plate ?? (record.plate_model_uncertain ? "模型不確定，待人工檢查" : "未辨識")}</dd></div>
                {record.human_verified_plate && <div><dt>人工確認</dt><dd>{record.human_verified_plate}</dd></div>}
              </>}
            </dl>
            <RecognitionJobList jobs={record.analysis_jobs ?? []} />
            <small>{formatRecordTime(record.created_at)} · {({ queued: "排隊中", processing: "辨識中", completed: "辨識完成", needs_review: "待覆核", failed: "辨識失敗", not_requested: "無背景工作" } as Record<string, string>)[record.analysis_status] ?? record.analysis_status}</small>
            {canDelete && <button
              type="button"
              className="danger-outline photo-record-delete"
              disabled={busy === record.id || busy === "clear-all"}
              onClick={() => void remove(record)}
            >{busy === record.id ? "刪除中…" : "刪除照片"}</button>}
          </div>
        </article>)}
      </div>}
    {visibleRecords.length > 3 && <button
      type="button"
      className="photo-record-expand"
      aria-expanded={expanded}
      onClick={() => setExpanded((current) => !current)}
    >{expanded ? "收合照片紀錄" : `展開其餘 ${visibleRecords.length - 3} 筆照片紀錄`}</button>}
  </section>;
}
