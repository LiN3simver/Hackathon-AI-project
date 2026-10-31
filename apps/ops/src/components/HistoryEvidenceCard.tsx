import { useEffect, useState } from "react";
import { damageCaseImageUrl, getHistoryEventDetail, ragSeedImageUrl, type HistoryEventDetail } from "../lib/api";

const damageTypes: Record<string, string> = {
  scratch: "刮痕", dent: "凹陷", crack: "裂痕", broken_part: "破損", paint_transfer: "掉漆",
  deformation: "變形", panel_misalignment: "板件錯位", other: "其他損傷",
};
const sources: Record<HistoryEventDetail["source"], string> = {
  damage_case: "AI 照片比對與人工覆核", record_text: "紀錄文字判讀", none: "沒有照片判讀資料",
};

function presence(value: boolean | null) {
  return value === true ? "有此損傷" : value === false ? "沒有此損傷" : "未記錄";
}

function PhotoPane({ src, label, present, regions }: {
  src: string | null; label: string; present: boolean | null; regions: HistoryEventDetail["regions"];
}) {
  const [failed, setFailed] = useState(false);
  const shown = src && !failed;
  return <div className="answer-evidence__pane">
    <div className="answer-evidence__frame">
      {shown ? <>
        <img src={src} alt={label} loading="lazy" onError={() => setFailed(true)} />
        {regions.map((region, index) => region.bounding_box && <span key={index} className="answer-evidence__box"
          title={region.description} style={{
            left: `${region.bounding_box[0] * 100}%`, top: `${region.bounding_box[1] * 100}%`,
            width: `${(region.bounding_box[2] - region.bounding_box[0]) * 100}%`,
            height: `${(region.bounding_box[3] - region.bounding_box[1]) * 100}%`,
          }} />)}
      </> : <span className="answer-evidence__missing">沒有照片</span>}
    </div>
    <span className={`answer-evidence__presence${present ? " is-present" : ""}`}>{label}：{presence(present)}</span>
    {shown && <a href={src} target="_blank" rel="noreferrer">開啟原圖</a>}
  </div>;
}

/** Pickup and return photos side by side, with the AI regions and the deterministic conclusion. */
export function HistoryEvidenceCard({ eventId }: { eventId: string }) {
  const [detail, setDetail] = useState<HistoryEventDetail | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setDetail(null);
    setFailed(false);
    getHistoryEventDetail(eventId, controller.signal).then(setDetail).catch(() => {
      if (!controller.signal.aborted) setFailed(true);
    });
    return () => controller.abort();
  }, [eventId]);
  if (failed) return <p className="answer-evidence__error" role="status">照片判讀暫時無法載入。</p>;
  if (!detail) return <div className="answer-evidence answer-evidence--loading" aria-busy="true">載入照片判讀…</div>;
  const url = (slot: "pickup" | "return") => detail.photo_source === "rag_seed" ? ragSeedImageUrl(detail.event_id, slot)
    : detail.photo_source === "damage_case" && detail.damage_case_id ? damageCaseImageUrl(detail.damage_case_id, slot) : null;
  return <figure className="answer-evidence">
    <figcaption>
      <strong>照片判讀</strong>
      <span>{detail.renter_name}{detail.license_plate ? `（${detail.license_plate}）` : ""}{detail.occurred_on ? ` · ${detail.occurred_on}` : ""}</span>
    </figcaption>
    <div className="answer-evidence__photos">
      {(detail.photo_slots ?? ["pickup", "return"]).includes("pickup") && <PhotoPane src={url("pickup")} label={detail.photo_slots?.length === 1 ? "事件照片" : "借車照片"} present={detail.present_at_pickup} regions={[]} />}
      {(detail.photo_slots ?? ["pickup", "return"]).includes("return") && <PhotoPane src={url("return")} label={detail.photo_slots?.length === 1 ? "事件照片" : "還車照片"} present={detail.present_at_return} regions={detail.regions} />}
    </div>
    <dl className="answer-evidence__facts">
      <div><dt>部位</dt><dd>{detail.location ?? detail.view_label ?? "未記錄"}</dd></div>
      <div><dt>類型</dt><dd>{detail.damage_type ? damageTypes[detail.damage_type] ?? detail.damage_type : "未記錄"}</dd></div>
      <div><dt>依據</dt><dd>{sources[detail.source]}</dd></div>
      {detail.ai_summary && <div><dt>AI 說明</dt><dd>{detail.ai_summary}{detail.ai_score != null ? `（風險分數 ${detail.ai_score.toFixed(2)}）` : ""}</dd></div>}
      {detail.reviewer_notes && <div><dt>覆核備註</dt><dd>{detail.reviewer_notes}</dd></div>}
      {detail.regions.length > 0 && <div><dt>標記區域</dt><dd>{detail.regions.map((region) => region.description).join("；")}</dd></div>}
    </dl>
    <p className="answer-evidence__conclusion">{detail.conclusion}</p>
  </figure>;
}
