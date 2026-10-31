import { useState, type FormEvent } from "react";
import { saveCleanlinessHistoryRecord, saveDamageHistoryRecord, saveLitterHistoryRecord, type RentalEvent } from "../lib/api";

export function SimulationRecordEditor({ record, busy, setBusy, onSaved, onCancel, onError }: {
  record: RentalEvent; busy: boolean; setBusy: (value: boolean) => void;
  onSaved: (event: RentalEvent) => void; onCancel: () => void; onError: (message: string | null) => void;
}) {
  const isDamage = record.event_type === "damage_liability_recorded";
  const [claimEnabled, setClaimEnabled] = useState(record.metadata.claim_enabled === true);
  const [description, setDescription] = useState(record.summary);
  const [date, setDate] = useState(String(record.metadata.occurred_on ?? record.occurred_on ?? ""));
  const [amount, setAmount] = useState((Number(record.metadata.compensation_cents ?? 0) / 100).toFixed(2));
  const [paid, setPaid] = useState((Number(record.metadata.paid_cents ?? 0) / 100).toFixed(2));
  const [confirmed, setConfirmed] = useState(record.verified);
  const [disputed, setDisputed] = useState(record.disputed);

  async function save(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    setBusy(true); onError(null);
    try {
      const common = { rental_id: record.rental_id, description, occurred_on: date || null,
        responsibility_confirmed: confirmed, disputed };
      const claim = { claim_enabled: claimEnabled, compensation_amount: claimEnabled ? amount : "0", paid_amount: claimEnabled ? paid : "0" };
      const saved = record.event_type === "cleanliness_confirmed"
        ? await saveCleanlinessHistoryRecord({ ...common, ...claim }, record.id)
        : isDamage
        ? await saveDamageHistoryRecord({ ...common, record_type: "damage",
          damage_case_id: typeof record.metadata.damage_case_id === "string" ? record.metadata.damage_case_id : null,
          compensation_amount: amount, paid_amount: paid }, record.id)
        : await saveLitterHistoryRecord({ ...common, ...claim,
          cleanliness_alert_id: typeof record.metadata.cleanliness_alert_id === "string" ? record.metadata.cleanliness_alert_id : null }, record.id);
      onSaved(saved);
    } catch (error) { onError(error instanceof Error ? error.message : "修改失敗"); }
    finally { setBusy(false); }
  }

  return <form onSubmit={(event) => void save(event)}>
    <fieldset className="rag-seed-fields" disabled={busy}>
      <legend>修改模擬資料事件</legend>
      <label>事件日期<input type="date" value={date} onChange={(event) => setDate(event.target.value)} /></label>
      <label>事件說明<textarea required maxLength={1000} rows={3} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
      {!isDamage && <label className="rag-seed-check"><input type="checkbox" checked={claimEnabled} onChange={(event) => setClaimEnabled(event.target.checked)} />新增索賠金額</label>}
      {(isDamage || claimEnabled) && <div className="rag-seed-grid">
        <label>{isDamage ? "車損應賠金額（NT$）" : "索賠金額（NT$）"}<input required type="number" min="0" max="100000000" step="0.01" value={amount} onChange={(event) => setAmount(event.target.value)} /></label>
        <label>車損已賠金額（NT$）<input required type="number" min="0" max={amount || "0"} step="0.01" value={paid} onChange={(event) => setPaid(event.target.value)} /></label>
      </div>}
      <label className="rag-seed-check"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />模擬已確認使用者責任</label>
      <label className="rag-seed-check"><input type="checkbox" checked={disputed} onChange={(event) => setDisputed(event.target.checked)} />模擬爭議中</label>
      <button className="primary" type="submit">{busy ? "儲存中…" : "儲存修改"}</button>
      <button type="button" onClick={onCancel}>取消修改</button>
    </fieldset>
  </form>;
}
