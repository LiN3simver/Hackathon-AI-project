import { useId, useState, type FormEvent, type ReactNode } from "react";
import {
  damageCaseImageUrl, downloadHistoryExcel, ragSeedImageUrl, type CapabilityItem, type EvidenceRow,
  type HistoryQueryResponse, type PersonRow,
  type PriorDamageCheck, type RiskRow, type ScopeChip,
} from "../lib/api";
import { HistoryEvidenceCard } from "./HistoryEvidenceCard";

const chipKinds: Record<ScopeChip["kind"], string> = { scope: "對象", period: "期間", topic: "主題" };
const responsibility: Record<EvidenceRow["responsibility"], string> = {
  confirmed: "責任已確認", pending: "待確認", disputed: "爭議中", ai_only: "AI 偵測未確認", not_applicable: "",
};
const verdicts: Record<PriorDamageCheck["verdict"], string> = {
  already_present: "可能早已存在", new_damage: "本次新增", unknown: "無法判斷",
};

export function money(value: string | null | undefined): string {
  const amount = Number(value ?? 0);
  return `NT$${amount.toLocaleString("zh-TW", { maximumFractionDigits: amount % 1 ? 2 : 0 })}`;
}

function ScopeChips({ chips, busy, onAsk }: { chips: ScopeChip[]; busy: boolean; onAsk: (question: string) => void }) {
  const [open, setOpen] = useState<string | null>(null);
  const menuId = useId();
  return <div className="answer-chips" role="group" aria-label="查詢條件，可點選修改">
    {chips.map((chip) => <div className="answer-chip" key={chip.kind}
      onKeyDown={(event) => { if (event.key === "Escape") setOpen(null); }}>
      <button type="button" aria-expanded={open === chip.kind} aria-controls={`${menuId}-${chip.kind}`}
        disabled={busy || !chip.options.length} onClick={() => setOpen(open === chip.kind ? null : chip.kind)}>
        <small>{chipKinds[chip.kind]}</small>{chip.label || "—"}
        {chip.options.length > 0 && <svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true"><path d="M2 4l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>}
      </button>
      {open === chip.kind && <div className="answer-chip__menu" id={`${menuId}-${chip.kind}`}>
        {chip.options.map((option) => <button type="button" key={option.question}
          onClick={() => { setOpen(null); onAsk(option.question); }}>{option.label}</button>)}
      </div>}
    </div>)}
  </div>;
}

function PeopleTable({ people, columns = ["damage", "litter"] }: { people: PersonRow[]; columns?: Array<"damage" | "litter"> }) {
  // Only the topics the question asked about; a damage-only answer has no litter column.
  const damage = columns.includes("damage"), litter = columns.includes("litter");
  return <div className="answer-table-wrap"><table className="answer-table">
    <thead><tr><th scope="col">使用者</th>
      {damage && <><th scope="col">責任車損</th><th scope="col">待確認</th><th scope="col">爭議</th></>}
      {litter && <th scope="col">垃圾</th>}
      {damage && <th scope="col">未賠</th>}</tr></thead>
    <tbody>{people.map((person, index) => <tr key={`${person.name}-${index}`}>
      <th scope="row">{person.name}{person.user_id && <small>編號 {person.user_id}</small>}{person.license_plates.length > 0 && <small>{person.license_plates.join("、")}</small>}</th>
      {damage && <><td>{person.caused_damage_count}</td><td>{person.unconfirmed_damage_count}</td><td>{person.disputed_count}</td></>}
      {litter && <td>{person.litter_count}</td>}
      {damage && <td>{Number(person.outstanding_amount) > 0 ? money(person.outstanding_amount) : "—"}</td>}
    </tr>)}</tbody>
  </table></div>;
}

function RiskRadar({ rows }: { rows: RiskRow[] }) {
  return <ol className="answer-risk" aria-label="風險排名">
    {rows.map((row) => <li key={row.rank} className="answer-risk__row">
      <span className="answer-risk__rank" aria-hidden="true">{row.rank}</span>
      <div className="answer-risk__main">
        <div className="answer-risk__name"><strong>{row.name}</strong>
          {row.license_plates.length > 0 && <small>{row.license_plates.join("、")}</small>}
          <span className={`answer-badge answer-badge--risk-${row.level}`}>{row.level_label}</span>
          {row.blacklisted && <span className="answer-badge answer-badge--disputed">黑名單</span>}
        </div>
        <div className="answer-risk__bar" role="img" aria-label={`風險分數 ${Math.round(row.score * 100)}`}>
          <span style={{ width: `${Math.max(4, row.score * 100)}%` }} />
        </div>
        <small>責任車損 {row.caused_damage_count} 件・垃圾 {row.litter_count} 次・未賠 {money(row.outstanding_amount)}
          ・爭議 {row.disputed_count} 件・租借 {row.rental_count} 次</small>
        {row.reasons.length > 0 && <small className="answer-risk__reason">{row.reasons.join("；")}</small>}
      </div>
    </li>)}
  </ol>;
}

function PriorDamage({ check }: { check: PriorDamageCheck }) {
  return <div className={`answer-prior answer-prior--${check.verdict}`}>
    <strong>{verdicts[check.verdict]}</strong>
    {check.earlier_rentals.length > 0 ? <ul>{check.earlier_rentals.map((rental, index) => <li key={index}>
      <span>{rental.renter_name}・{rental.rental_date ?? "?"} → {rental.return_date ?? "?"}</span>
      <span>相同部位紀錄：{rental.same_location ? "有" : "無"}{rental.visual_score != null ? `・影像相似分數 ${rental.visual_score.toFixed(2)}` : ""}</span>
      {rental.records.length > 0 && <small>{rental.records.join("；")}</small>}
    </li>)}</ul> : <small>這台車沒有更早的租借紀錄可以比對。</small>}
    {!check.visual_checked && <small>未進行影像比對（缺少照片或比對模型）。</small>}
  </div>;
}

function SourcePhotos({ row }: { row: EvidenceRow }) {
  const [hidden, setHidden] = useState<string[]>([]);
  const url = (slot: "pickup" | "return") => row.photo_source === "rag_seed" && row.event_id ? ragSeedImageUrl(row.event_id, slot)
    : row.photo_source === "damage_case" && row.damage_case_id ? damageCaseImageUrl(row.damage_case_id, slot) : null;
  if (!row.photo_source) return null;
  return <div className="rag-seed-photos">{(row.photo_slots ?? ["pickup", "return"] as const).filter((slot) => !hidden.includes(slot)).map((slot) => {
    const src = url(slot);
    return src && <a key={slot} href={src} target="_blank" rel="noreferrer">
      <img src={src} alt={row.photo_slots?.length === 1 ? "事件照片" : slot === "pickup" ? "借車照片" : "還車照片"} loading="lazy" onError={() => setHidden((items) => [...items, slot])} />
      <span>{row.photo_slots?.length === 1 ? "事件照片" : slot === "pickup" ? "借車照片" : "還車照片"}</span>
    </a>;
  })}</div>;
}

/** Records with photos appear directly in the chat, up to the card's photo_limit pairs. */
function PhotoGallery({ rows, anchor }: { rows: EvidenceRow[]; anchor: (ref: number) => string }) {
  return <div className="answer-gallery" aria-label="照片">
    {rows.map((row) => <figure key={row.ref} className="answer-gallery__item">
      <figcaption><a href={`#${anchor(row.ref)}`} className="answer-source__ref">[{row.ref}]</a>
        {row.occurred_on ?? "日期未填"} · {row.renter_name}{row.license_plate ? `（${row.license_plate}）` : ""}
        {responsibility[row.responsibility] && <span className={`answer-badge answer-badge--${row.responsibility}`}>{responsibility[row.responsibility]}</span>}
      </figcaption>
      <SourcePhotos row={row} />
    </figure>)}
  </div>;
}

function ExcelDownload({ result, note }: { result: HistoryQueryResponse; note: string | null }) {
  const [state, setState] = useState<"idle" | "busy" | "error">("idle");
  async function download() {
    setState("busy");
    try {
      await downloadHistoryExcel({ question: result.question, context: result.context, license_plate: result.license_plate });
      setState("idle");
    } catch { setState("error"); }
  }
  return <div className={`answer-download${note ? " is-truncated" : ""}`}>
    {note && <span>{note}</span>}
    <button type="button" onClick={() => void download()} disabled={state === "busy"}>
      <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></svg>
      {state === "busy" ? "產生中…" : "下載完整 Excel"}
    </button>
    {state === "error" && <small role="alert">下載失敗，請再試一次。</small>}
  </div>;
}

function SourceRow({ row, anchor }: { row: EvidenceRow; anchor: string }) {
  return <div className={`answer-source${row.cited ? " is-cited" : ""}`} id={anchor}>
    <div className="answer-source__head">
      <span className="answer-source__ref">[{row.ref}]</span>
      <strong>{row.occurred_on ?? "日期未填"}{row.return_date ? ` → ${row.return_date}` : ""} · {row.renter_name}
        {row.license_plate ? `（${row.license_plate}）` : ""}</strong>
      {responsibility[row.responsibility] && <span className={`answer-badge answer-badge--${row.responsibility}`}>{responsibility[row.responsibility]}</span>}
    </div>
    <div className="answer-source__body">{row.kind_label}｜{row.title}：{row.summary}</div>
    {row.compensation_amount != null && <small>應賠 {money(row.compensation_amount)}、已賠 {money(row.paid_amount)}{row.counted ? "" : "（不計入）"}</small>}
    {row.photo_source && <small>照片見上方［{row.ref}］或完整 Excel</small>}
  </div>;
}

function Feedback({ value, onSend }: { value?: 1 | -1; onSend: (rating: 1 | -1, expected: string) => Promise<void> }) {
  const [writing, setWriting] = useState(false);
  const [note, setNote] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  async function send(rating: 1 | -1, expected = "") {
    setSending(true);
    setError("");
    try { await onSend(rating, expected); setWriting(false); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "回饋送出失敗"); }
    finally { setSending(false); }
  }
  if (value) return <small className="answer-feedback__done">{value === 1 ? "已回饋：有幫助，謝謝！" : "已回饋：需要改進，謝謝！"}</small>;
  return <div className="answer-feedback">
    <span>這個回答有幫助嗎？</span>
    <button type="button" aria-label="有幫助" title="有幫助" disabled={sending} onClick={() => void send(1)}>
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden="true"><path d="M7 11v9H4v-9h3zm0 0l4-7a2 2 0 0 1 3 2l-1 4h5a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 16.8 20H7" /></svg>
    </button>
    <button type="button" aria-label="需要改進" title="需要改進" aria-expanded={writing} disabled={sending} onClick={() => setWriting((open) => !open)}>
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" aria-hidden="true"><path d="M17 13V4h3v9h-3zm0 0l-4 7a2 2 0 0 1-3-2l1-4H6a2 2 0 0 1-2-2.3l1.2-6A2 2 0 0 1 7.2 4H17" /></svg>
    </button>
    {writing && <form className="answer-feedback__form" onSubmit={(event: FormEvent) => { event.preventDefault(); void send(-1, note.trim()); }}>
      <textarea aria-label="正確答案應該是" placeholder="正確答案應該是…（可留空）" maxLength={2000} rows={2} value={note}
        onChange={(event) => setNote(event.target.value)} />
      <div><button type="submit" disabled={sending}>送出回饋</button><button type="button" onClick={() => setWriting(false)}>取消</button></div>
    </form>}
    {error && <small role="alert">{error}</small>}
  </div>;
}

const capabilityTone: Record<CapabilityItem["status"], string> = { ready: "risk-low", limited: "risk-watch", off: "off" };

/** 「你可以做什麼？」: what works now, built by the server from live data and settings. */
function Capabilities({ items, busy, onAsk }: { items: CapabilityItem[]; busy: boolean; onAsk?: (question: string) => void }) {
  return <ul className="answer-capabilities" aria-label="助理功能">
    {items.map((item) => <li key={item.key} className={`answer-capability answer-capability--${item.status}`}>
      <div className="answer-capability__head">
        <strong>{item.title}</strong>
        <span className={`answer-badge answer-badge--${capabilityTone[item.status]}`}>{item.status_label}</span>
      </div>
      <p>{item.description}</p>
      {item.examples.length > 0 && <div className="answer-card__followups" aria-label={`${item.title}範例問題`}>
        {item.examples.map((example) => onAsk
          ? <button type="button" key={example} disabled={busy} onClick={() => onAsk(example)}>{example}</button>
          : <span key={example}>{example}</span>)}
      </div>}
    </li>)}
  </ul>;
}

function withCitations(text: string, anchor: (ref: number) => string, onJump: (ref: number) => void): ReactNode[] {
  return text.split(/(\[\d+\])/).map((part, index) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) return part;
    const ref = Number(match[1]);
    return <a key={index} href={`#${anchor(ref)}`} className="answer-cite" onClick={(event) => { event.preventDefault(); onJump(ref); }}>{ref}</a>;
  });
}

function notice(result: HistoryQueryResponse): string | null {
  const warnings = result.warnings ?? [];
  if (warnings.includes("llm_generation_disabled")) return "AI 摘要已停用；上方為系統直接計算的結果。";
  if (warnings.includes("ollama_unverified") && result.generated_by !== "ollama_summary") return "AI 摘要未通過資料核對，已隱藏；上方為系統直接計算的結果。";
  if (warnings.includes("ollama_partial")) return "AI 摘要因時間限制只顯示部分內容。";
  if (warnings.includes("ollama_unverified")) return "AI 摘要後段未通過資料核對，已截斷。";
  if (result.generated_by !== "ollama_summary" && warnings.some((warning) => warning.startsWith("ollama_")))
    return "AI 摘要暫時無法使用；上方為系統直接計算的結果。";
  return null;
}

/** The deterministic answer card, then the streamed AI summary, sources, follow-ups and feedback. */
export function HistoryAnswerView({ result, summary, streaming, queued, stopped, busy, onAsk, feedback, onFeedback }: {
  result: HistoryQueryResponse; summary: string; streaming?: boolean; queued?: boolean; stopped?: boolean; busy: boolean;
  onAsk?: (question: string) => void; feedback?: 1 | -1; onFeedback?: (rating: 1 | -1, expected: string) => Promise<void>;
}) {
  const card = result.answer_card!;
  const prefix = useId().replace(/:/g, "");
  const anchor = (ref: number) => `${prefix}-source-${ref}`;
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const cited = card.rows.filter((row) => row.cited);
  const primary = cited.length ? cited : card.rows;
  const others = cited.length ? card.rows.filter((row) => !row.cited) : [];
  const showDetail = card.intent === "event_detail" || card.intent === "prior_damage";
  const message = !streaming ? notice(result) : null;
  const photoRows = showDetail ? [] : card.rows.filter((row) => row.photo_source);
  const shownPhotos = photoRows.slice(0, card.photo_limit ?? 10);
  const hiddenRecords = Math.max(0, card.total_rows - card.rows.length);
  const hiddenPhotos = photoRows.length - shownPhotos.length;
  const shown = [hiddenPhotos > 0 ? `前 ${shownPhotos.length} 組照片` : "",
    hiddenRecords > 0 ? `前 ${card.rows.length} 筆紀錄（共 ${card.total_rows} 筆）` : ""].filter(Boolean).join("與");
  const downloadNote = shown ? `對話中只顯示${shown}；其餘請下載完整 Excel 查看。` : null;
  function jump(ref: number) {
    setSourcesOpen(true);
    requestAnimationFrame(() => document.getElementById(anchor(ref))?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
  }
  return <div className="answer-card">
    {onAsk && card.chips.length > 0 && <ScopeChips chips={card.chips} busy={busy} onAsk={onAsk} />}
    <p className="answer-card__headline">{card.headline}</p>
    {card.metrics.length > 0 && <div className="answer-card__metrics">{card.metrics.map((metric) =>
      <span key={metric.label} className={`answer-metric answer-metric--${metric.tone}`}><small>{metric.label}</small><strong>{metric.value}</strong></span>)}</div>}
    {card.people.length > 1 && <PeopleTable people={card.people} columns={card.people_columns} />}
    {card.risk.length > 0 && <RiskRadar rows={card.risk} />}
    {card.prior_damage && <PriorDamage check={card.prior_damage} />}
    {card.capabilities && card.capabilities.length > 0 && <Capabilities items={card.capabilities} busy={busy || Boolean(streaming)} onAsk={onAsk} />}
    {showDetail && card.detail_event_ids.map((id) => <HistoryEvidenceCard key={id} eventId={id} />)}
    {shownPhotos.length > 0 && <PhotoGallery rows={shownPhotos} anchor={anchor} />}
    {(summary || streaming) && <p className="answer-card__summary" aria-busy={streaming || undefined}>
      <span className="answer-card__summary-label">AI 摘要</span>
      {summary ? withCitations(summary, anchor, jump) : queued ? "等待 AI 模型中…" : "AI 摘要產生中…"}
    </p>}
    {stopped && <small role="status" className="answer-card__notice">已停止產生摘要。</small>}
    {message && <small role="status" className="answer-card__notice">{message}</small>}
    {card.rows.length > 0 && <details className="history-chat__evidence" open={sourcesOpen} onToggle={(event) => setSourcesOpen(event.currentTarget.open)}>
      <summary>依據（{primary.length} 筆）{card.total_rows > card.rows.length ? `；共 ${card.total_rows} 筆符合` : ""}</summary>
      {primary.map((row) => <SourceRow key={row.ref} row={row} anchor={anchor(row.ref)} />)}
      {others.length > 0 && <details className="answer-card__more"><summary>其他相關紀錄（{others.length} 筆）</summary>
        {others.map((row) => <SourceRow key={row.ref} row={row} anchor={anchor(row.ref)} />)}
      </details>}
    </details>}
    {card.rows.length > 0 && !streaming && <ExcelDownload result={result} note={downloadNote} />}
    {onAsk && !streaming && card.followups.length > 0 && <div className="answer-card__followups" aria-label="建議追問">
      {card.followups.map((item) => <button type="button" key={item} disabled={busy} onClick={() => onAsk(item)}>{item}</button>)}
    </div>}
    {onFeedback && !streaming && <Feedback value={feedback} onSend={onFeedback} />}
  </div>;
}
