import { FormEvent, useEffect, useState } from "react";
import {
  getDamageHistoryRecords, getLitterHistoryRecords, saveDamageHistoryRecord,
  saveLitterHistoryRecord, setHistoryBlacklist, deleteHistoryRecord,
  type RentalEvent, type UserHistoryProfile,
} from "../lib/api";

type RecordDraft = {
  rental_id: string; record_type: "damage" | "litter"; description: string;
  occurred_on: string | null; damage_case_id: string | null; cleanliness_alert_id: string | null;
  claim_enabled: boolean; compensation_amount: string; paid_amount: string; responsibility_confirmed: boolean; disputed: boolean;
};
const blankRecord: RecordDraft = {
  rental_id: "", record_type: "damage", description: "", occurred_on: null,
  damage_case_id: null, cleanliness_alert_id: null,
  claim_enabled: false, compensation_amount: "0", paid_amount: "0", responsibility_confirmed: false,
  disputed: false,
};

export function HistoryRecordManager({ profile, onChange, defaultOpen = false }: {
  profile: UserHistoryProfile; onChange: () => Promise<void>; defaultOpen?: boolean;
}) {
  const [record, setRecord] = useState({ ...blankRecord, rental_id: profile.rental_ids[0] ?? "" });
  const [editId, setEditId] = useState<string>();
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [records, setRecords] = useState(profile.events.filter((event) => ["damage_liability_recorded", "litter_recorded"].includes(event.event_type)));

  useEffect(() => {
    let active = true;
    void Promise.all([getDamageHistoryRecords(profile.user_id), getLitterHistoryRecords(profile.user_id)]).then(([damage, litter]) => {
      if (active) setRecords([...damage, ...litter].sort((left, right) => right.created_at.localeCompare(left.created_at)));
    }).catch((caught) => {
      if (active) setError(caught instanceof Error ? caught.message : "紀錄讀取失敗");
    });
    return () => { active = false; };
  }, [profile]);

  function edit(event: RentalEvent) {
    setEditId(event.id);
    setRecord({
      rental_id: event.rental_id, description: event.summary,
      record_type: event.event_type === "litter_recorded" ? "litter" : "damage",
      occurred_on: typeof event.metadata.occurred_on === "string" ? event.metadata.occurred_on : null,
      damage_case_id: typeof event.metadata.damage_case_id === "string" ? event.metadata.damage_case_id : null,
      cleanliness_alert_id: typeof event.metadata.cleanliness_alert_id === "string" ? event.metadata.cleanliness_alert_id : null,
      claim_enabled: event.metadata.claim_enabled === true,
      compensation_amount: (Number(event.metadata.compensation_cents ?? 0) / 100).toFixed(2),
      paid_amount: (Number(event.metadata.paid_cents ?? 0) / 100).toFixed(2),
      responsibility_confirmed: event.verified, disputed: event.disputed,
    });
    setError("");
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!editId) return;
    setBusy(true); setError("");
    try {
      const common = { rental_id: record.rental_id, description: record.description,
        occurred_on: record.occurred_on, responsibility_confirmed: record.responsibility_confirmed,
        disputed: record.disputed };
      if (record.record_type === "litter") {
        await saveLitterHistoryRecord({ ...common, cleanliness_alert_id: record.cleanliness_alert_id, claim_enabled: record.claim_enabled, compensation_amount: record.claim_enabled ? record.compensation_amount : "0", paid_amount: record.claim_enabled ? record.paid_amount : "0" }, editId);
      } else {
        await saveDamageHistoryRecord({ ...common, record_type: "damage", damage_case_id: record.damage_case_id,
          compensation_amount: record.compensation_amount, paid_amount: record.paid_amount }, editId);
      }
      setEditId(undefined);
      setRecord({ ...blankRecord, rental_id: profile.rental_ids[0] ?? "" });
      await onChange();
    } catch (caught) { setError(caught instanceof Error ? caught.message : "儲存失敗"); }
    finally { setBusy(false); }
  }

  async function blacklist(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      await setHistoryBlacklist(profile.user_id, !profile.blacklisted, reason);
      await onChange();
    } catch (caught) { setError(caught instanceof Error ? caught.message : "黑名單更新失敗"); }
    finally { setBusy(false); }
  }

  async function remove(recordEvent: RentalEvent) {
    if (busy || !window.confirm(`確定刪除「${recordEvent.title}」（租單 ${recordEvent.rental_id}）？刪除後將重新計算違規次數與賠款統計。`)) return;
    setBusy(true); setError("");
    try {
      await deleteHistoryRecord(recordEvent);
      setRecords((current) => current.filter((item) => item.id !== recordEvent.id));
      if (editId === recordEvent.id) setEditId(undefined);
      await onChange();
    } catch (caught) { setError(caught instanceof Error ? caught.message : "刪除失敗"); }
    finally { setBusy(false); }
  }

  return <details className="history-record-manager" open={defaultOpen}>
    <summary>{profile.display_name}（{profile.user_id}）的車損、垃圾紀錄與黑名單</summary>
    <p>可查看、修改與刪除既有車損、垃圾紀錄；新增模擬紀錄請到「模擬資料區」。尚待確認或爭議中的紀錄不計入已確認次數。刪除紀錄後，已列入的黑名單可依需要手動移出。</p>
    {error && <div className="error-box" role="alert">{error}</div>}
    {editId && <form className="history-record-form" onSubmit={(event) => void save(event)}>
      <h3>修改紀錄</h3>
      <label>紀錄類型<select disabled={!!editId} value={record.record_type} onChange={(event) => setRecord({ ...record, record_type: event.target.value as "damage" | "litter", damage_case_id: null, cleanliness_alert_id: null })}>
        <option value="damage">車損賠償</option><option value="litter">留下垃圾</option>
      </select></label>
      <label>租單<select required disabled={!!editId} value={record.rental_id} onChange={(event) => setRecord({ ...record, rental_id: event.target.value })}>
        {profile.rental_ids.map((id) => <option key={id}>{id}</option>)}
      </select></label>
      <label>事件日期（留白優先使用還車日期）<input type="date" value={record.occurred_on ?? ""} onChange={(event) => setRecord({ ...record, occurred_on: event.target.value || null })} /></label>
      {record.record_type === "damage"
        ? <label>車損案件編號（選填）<input maxLength={80} disabled={!!editId} value={record.damage_case_id ?? ""} placeholder="連結已人工確認的車損案件" onChange={(event) => setRecord({ ...record, damage_case_id: event.target.value.trim() || null })} /></label>
        : <label>垃圾警示編號（選填）<input maxLength={80} disabled={!!editId} value={record.cleanliness_alert_id ?? ""} placeholder="連結已人工確認且有垃圾證據的警示" onChange={(event) => setRecord({ ...record, cleanliness_alert_id: event.target.value.trim() || null })} /></label>}
      <label>事件說明<textarea required maxLength={1000} value={record.description} onChange={(event) => setRecord({ ...record, description: event.target.value })} /></label>
      {record.record_type === "litter" && <label className="history-record-checkbox"><input type="checkbox" checked={record.claim_enabled} onChange={(event) => setRecord({ ...record, claim_enabled: event.target.checked })} />新增索賠金額</label>}
      {(record.record_type === "damage" || record.claim_enabled) && <>
        <label>應賠金額（NT$）<input required type="number" min="0" max="100000000" step="0.01" value={record.compensation_amount} onChange={(event) => setRecord({ ...record, compensation_amount: event.target.value })} /></label>
        <label>已賠金額（NT$）<input required type="number" min="0" max={record.compensation_amount || "0"} step="0.01" value={record.paid_amount} onChange={(event) => setRecord({ ...record, paid_amount: event.target.value })} /></label>
      </>}
      <label className="history-record-checkbox"><input type="checkbox" checked={record.responsibility_confirmed} onChange={(event) => setRecord({ ...record, responsibility_confirmed: event.target.checked })} />{record.record_type === "damage" ? "已確認由此使用者造成車損" : "已確認此使用者留下垃圾"}</label>
      <label className="history-record-checkbox"><input type="checkbox" checked={record.disputed} onChange={(event) => setRecord({ ...record, disputed: event.target.checked })} />爭議中（暫不列入已確認紀錄）</label>
      <button disabled={busy || !record.rental_id}>{busy ? "儲存中…" : "儲存紀錄"}</button>
      {editId && <button type="button" disabled={busy} onClick={() => { setEditId(undefined); setRecord({ ...blankRecord, rental_id: profile.rental_ids[0] ?? "" }); }}>取消修改</button>}
    </form>}
    <div className="history-record-list">
      {records.map((event) => <article key={event.id}>
        <div><strong>{event.occurred_on ?? ""} · {event.title}{event.disputed ? "（爭議中）" : ""}</strong>
          <p>{event.rental_id}｜{event.summary}</p>
          {(event.event_type === "damage_liability_recorded" || event.metadata.claim_enabled === true) && <small>應賠 NT$ {(Number(event.metadata.compensation_cents ?? 0) / 100).toFixed(2)} · 已賠 NT$ {(Number(event.metadata.paid_cents ?? 0) / 100).toFixed(2)}</small>}</div>
        <button type="button" disabled={busy} onClick={() => edit(event)}>{event.event_type === "litter_recorded" ? "修改垃圾紀錄" : "修改／登錄賠款"}</button>
        <button type="button" className="danger" disabled={busy} onClick={() => void remove(event)}>刪除紀錄</button>
      </article>)}
      {!records.length && <p>尚未建立車損或垃圾紀錄。</p>}
    </div>
    <form className="history-record-form" onSubmit={(event) => void blacklist(event)}>
      <h3>黑名單：{profile.blacklisted ? "已列入" : "未列入"}</h3>
      {profile.blacklist_reason && <p>最近設定原因：{profile.blacklist_reason}</p>}
      <label>{profile.blacklisted ? "移出原因" : "列入原因"}<input required maxLength={1000} value={reason} onChange={(event) => setReason(event.target.value)} /></label>
      <button disabled={busy}>{profile.blacklisted ? "移出黑名單" : "列入黑名單"}</button>
    </form>
  </details>;
}
