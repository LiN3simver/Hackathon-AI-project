import { useEffect, useRef, useState, type FormEvent } from "react";
import { createRagSeed, deleteHistoryRecord, getRagSeeds, getUserHistory, type RentalEvent } from "../lib/api";
import { BLACKLIST_OFFENCES, offences } from "./BlacklistManager";
import { RagSeedPhotos } from "./RagSeedPhotos";
import { SimulationRecordEditor } from "./SimulationRecordEditor";

function sampleRental() {
  const suffix = crypto.randomUUID().slice(0, 8).toUpperCase();
  const today = new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Taipei", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date());
  return { rental_id: `RAG-${suffix}`, customer_id: `RAG-USER-${suffix}`, display_name: "模擬資料使用者",
    rental_date: today, return_date: today, vehicle_label: "", license_plate: "", password: "" };
}

export function PhotoInput({ title, file, onChange }: { title: string; file: File | null; onChange: (file: File | null) => void }) {
  const [preview, setPreview] = useState("");
  useEffect(() => {
    if (!file) { setPreview(""); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  return <label className="rag-photo-input"><strong>{title}</strong>
    <input required type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => onChange(event.target.files?.[0] ?? null)} />
    {preview && <img src={preview} alt={`${title}預覽`} />}
  </label>;
}

export function RagSeedManager() {
  const createForm = useRef<HTMLFormElement>(null);
  const [editId, setEditId] = useState<string | null>(null);
  const [rental, setRental] = useState(sampleRental);
  const [kind, setKind] = useState("damage");
  const [description, setDescription] = useState("還車時新增右前車門刮痕，借車照片無此車損。");
  const [amount, setAmount] = useState("3000");
  const [paid, setPaid] = useState("0");
  const [claimEnabled, setClaimEnabled] = useState(false);
  const [claimAmount, setClaimAmount] = useState("0");
  const [claimPaid, setClaimPaid] = useState("0");
  const needsPair = ["damage", "both"].includes(kind);
  const [confirmed, setConfirmed] = useState(true);
  const [disputed, setDisputed] = useState(false);
  const [pickup, setPickup] = useState<File | null>(null);
  const [returned, setReturned] = useState<File | null>(null);
  const [records, setRecords] = useState<RentalEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [fileKey, setFileKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getRagSeeds().then((rows) => { if (!cancelled) setRecords(rows); })
      .catch((err: unknown) => { if (!cancelled) setError(err instanceof Error ? err.message : "無法載入模擬資料"); });
    return () => { cancelled = true; };
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!returned || (needsPair && !pickup) || busy) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      const record = { rental_id: rental.rental_id, description, responsibility_confirmed: confirmed, disputed };
      const claim = { claim_enabled: claimEnabled, compensation_amount: claimEnabled ? claimAmount : "0", paid_amount: claimEnabled ? claimPaid : "0" };
      const created = await createRagSeed({ rental,
        ...(["damage", "both"].includes(kind) ? { damage: { ...record, compensation_amount: amount, paid_amount: paid } } : {}),
        ...(["litter", "both"].includes(kind) ? { litter: { ...record, ...(kind === "litter" ? claim : {}) } } : {}),
        ...(kind === "cleanliness" ? { cleanliness: { ...record, ...claim } } : {}),
      }, needsPair ? pickup : null, returned);
      setRecords((current) => [...created, ...current].slice(0, 100));
      const profile = await getUserHistory(rental.customer_id).catch(() => null);
      const count = profile ? offences(profile) : 0;
      const status = !profile ? "" : profile.blacklisted
        ? `此使用者已確認不良紀錄 ${count} 次，已列入黑名單。`
        : `此使用者已確認不良紀錄 ${count} 次，${count >= BLACKLIST_OFFENCES ? "曾被人工移出黑名單，不再自動列入。" : `再 ${BLACKLIST_OFFENCES - count} 次會自動列入黑名單。`}`;
      setNotice(`已建立模擬資料租單 ${rental.rental_id}，可到借還車AI助理查詢使用者 ${rental.customer_id}。${status}照片已保存，未執行辨識。`);
      setRental((current) => ({ ...current, rental_id: sampleRental().rental_id }));
      setPickup(null); setReturned(null); setFileKey((current) => current + 1);
    } catch (err) { setError(err instanceof Error ? err.message : "建立失敗"); }
    finally { setBusy(false); }
  }

  async function remove(record: RentalEvent) {
    if (busy || !window.confirm(`確定刪除「${record.title}」（租單 ${record.rental_id}）？此事件會從歷程及統計中移除。`)) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      await deleteHistoryRecord(record);
      setRecords((current) => current.filter((item) => item.id !== record.id));
      if (editId === record.id) setEditId(null);
      setNotice("已刪除模擬資料事件。");
    } catch (err) { setError(err instanceof Error ? err.message : "刪除失敗"); }
    finally { setBusy(false); }
  }

  return <section className="rag-seed-manager">
    <div className="settings-group-heading"><p className="eyebrow">RAG SAMPLE DATA</p><h2>模擬資料區</h2>
      <p>建立造成車損、留下垃圾或造成髒汙的模擬租單，供借還車AI助理檢索。垃圾與髒汙只需一張事件照片，可選擇新增索賠金額。所有紀錄標示為模擬資料，略過辨識與模型訓練。</p>
    </div>
    {error && <p role="alert" className="rag-seed-error">{error}</p>}
    {notice && <p role="status" className="rag-seed-notice">{notice}</p>}
    <form ref={createForm} onSubmit={(event) => void submit(event)}>
      <fieldset disabled={busy} className="rag-seed-fields">
        <legend>租單與使用者</legend>
        <button type="button" onClick={() => setRental(sampleRental())}>自動填入一組新模擬資料</button>
        <div className="rag-seed-grid">
          {([
            ["rental_id", "租單編號", "text", 80], ["customer_id", "使用者編號", "text", 80],
            ["display_name", "使用者姓名", "text", 120], ["rental_date", "借車日期", "date", 10],
            ["return_date", "還車日期", "date", 10], ["vehicle_label", "車輛名稱", "text", 120],
            ["license_plate", "車牌號碼", "text", 10], ["password", "租單密碼（4 位數字）", "text", 4],
          ] as const).map(([key, label, type, maxLength]) => <label key={key}><span>{label}</span>
            <input required type={type} maxLength={maxLength} value={rental[key]}
              min={key === "return_date" ? rental.rental_date : undefined}
              pattern={key === "password" ? "[0-9]{4}" : undefined}
              onChange={(event) => setRental({ ...rental, [key]: event.target.value })} />
          </label>)}
        </div><small>同一人的多張租單請使用相同使用者編號，建立後會自動保留此編號方便繼續新增。</small>
      </fieldset>
      <fieldset disabled={busy} className="rag-seed-fields"><legend>事件內容</legend>
        <label>案例類型<select value={kind} onChange={(event) => {
          setKind(event.target.value);
          setDescription(event.target.value === "cleanliness" ? "還車時座椅出現飲料污漬，借車時座椅乾淨，沒有留下垃圾。" : event.target.value === "litter" ? "還車時在後座留下飲料杯與包裝垃圾，借車時車內乾淨。" : event.target.value === "both" ? "還車時新增右前車門刮痕，並在後座留下飲料杯與包裝垃圾。" : "還車時新增右前車門刮痕，借車照片無此車損。");
        }}><option value="damage">造成車損</option><option value="litter">留下垃圾</option><option value="cleanliness">造成髒汙</option><option value="both">車損與垃圾</option></select></label>
        <label>事件說明<textarea required maxLength={1000} rows={3} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
        {["damage", "both"].includes(kind) && <div className="rag-seed-grid">
          <label>車損應賠金額（NT$）<input required type="number" min="0" max="100000000" step="0.01" value={amount} onChange={(event) => setAmount(event.target.value)} /></label>
          <label>車損已賠金額（NT$）<input required type="number" min="0" max={amount} step="0.01" value={paid} onChange={(event) => setPaid(event.target.value)} /></label>
        </div>}
        {!needsPair && <>
          <label className="rag-seed-check"><input type="checkbox" checked={claimEnabled} onChange={(event) => setClaimEnabled(event.target.checked)} />新增索賠金額</label>
          {claimEnabled && <div className="rag-seed-grid">
            <label>索賠金額（NT$）<input required type="number" min="0" max="100000000" step="0.01" value={claimAmount} onChange={(event) => setClaimAmount(event.target.value)} /></label>
            <label>已賠金額（NT$）<input required type="number" min="0" max={claimAmount || "0"} step="0.01" value={claimPaid} onChange={(event) => setClaimPaid(event.target.value)} /></label>
          </div>}
        </>}
        <label className="rag-seed-check"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} />模擬已確認使用者責任</label>
        <label className="rag-seed-check"><input type="checkbox" checked={disputed} onChange={(event) => setDisputed(event.target.checked)} />模擬爭議中</label>
        <small>事件日期採還車日期；待確認或爭議中的事件不計入已確認責任統計。同一使用者編號已確認的車損、垃圾與髒污合計達 {BLACKLIST_OFFENCES} 次（同一張租單兩者都有算 2 次）會自動列入黑名單。</small>
        {kind === "cleanliness" && <small>髒汙紀錄供整潔歷程查詢，不計入垃圾次數或車損賠款。</small>}
      </fieldset>
      <fieldset disabled={busy} className="rag-seed-fields"><legend>{needsPair ? "借還車照片" : "事件照片"}</legend>
        <p>{needsPair ? "每組需一張借車照片及一張還車照片。" : "上傳一張垃圾或髒汙的事件照片即可。"}接受 JPEG、PNG、WebP，不執行辨識。</p>
        <div className="rag-seed-grid" key={`${fileKey}-${needsPair}`}>{needsPair && <PhotoInput title="借車照片" file={pickup} onChange={setPickup} />}<PhotoInput title={needsPair ? "還車照片" : "事件照片"} file={returned} onChange={setReturned} /></div>
      </fieldset>
      <button className="primary" disabled={busy || (needsPair && !pickup) || !returned}>{busy ? "建立並保存照片中…" : "建立 RAG 模擬資料租單"}</button>
    </form>
    <div className="settings-group-heading"><h3>最近建立的模擬資料事件</h3><p>最多顯示 100 筆事件；車損與垃圾同時建立時各有一筆，共用同一組照片。</p>
      <button type="button" disabled={busy} onClick={() => {
        createForm.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        createForm.current?.querySelector("input")?.focus({ preventScroll: true });
      }}>新增模擬資料</button>
    </div>
    {!records.length && <p>尚未建立模擬資料。</p>}
    <div className="rag-seed-records">{records.map((record) => <article key={record.id}>
      <strong>{record.title} · {record.verified ? "責任已確認" : "待確認"}{record.disputed ? " · 爭議中" : ""}</strong>
      <p>{record.rental_id} · 使用者 {record.user_id} · {String(record.metadata.occurred_on ?? "")}</p>
      <p>{record.summary}</p><RagSeedPhotos event={record} />
      {(record.event_type === "damage_liability_recorded" || record.metadata.claim_enabled === true) && <p>應賠 NT$ {(Number(record.metadata.compensation_cents ?? 0) / 100).toFixed(2)} · 已賠 NT$ {(Number(record.metadata.paid_cents ?? 0) / 100).toFixed(2)}</p>}
      {editId === record.id ? <SimulationRecordEditor key={record.id} record={record} busy={busy} setBusy={setBusy}
        onError={setError} onCancel={() => setEditId(null)} onSaved={(saved) => {
          setRecords((current) => current.map((item) => item.id === saved.id ? saved : item));
          setEditId(null); setNotice("已修改模擬資料事件。");
        }} /> : <button type="button" disabled={busy} onClick={() => { setEditId(record.id); setError(null); setNotice(null); }}>修改事件</button>}
      <button type="button" className="danger" disabled={busy} onClick={() => void remove(record)}>刪除事件</button>
    </article>)}</div>
  </section>;
}
