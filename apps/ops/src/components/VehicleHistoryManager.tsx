import { useEffect, useId, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { getSavedVehicleHistories, uploadVehicleHistory, type VehicleHistoryProfile } from "../lib/api";
import { BLACKLIST_OFFENCES } from "./BlacklistManager";
import { RagSeedPhotos } from "./RagSeedPhotos";
import { isHistoryPhoto, mergeHistoryPhotos } from "../lib/vehicleHistoryPhotos";

function newRental() {
  return { key: crypto.randomUUID(), rental_id: `HIST-${crypto.randomUUID().slice(0, 8).toUpperCase()}`,
    customer_id: "", display_name: "", rental_date: "", return_date: "", kind: "none", description: "",
    compensation_amount: "0", paid_amount: "0", confirmed: true, disputed: false,
    pickup: [] as File[], returned: [] as File[], evidence: [] as File[] };
}
type RentalDraft = ReturnType<typeof newRental>;

function PhotoUploadInput({ title, files, onChange }: { title: string; files: File[]; onChange: (files: File[]) => void }) {
  const [skipped, setSkipped] = useState(0);
  function addFiles(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    if (!selected.length) return;
    const images = selected.filter(isHistoryPhoto)
      .sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
    setSkipped(selected.length - images.length);
    onChange(mergeHistoryPhotos(files, images));
    event.target.value = "";
  }
  return <div><strong>{title}</strong>
    <label>選擇多張圖片<input type="file" multiple accept=".jpg,.jpeg,.png,.webp" aria-label={`${title}：選擇多張圖片`} onChange={addFiles} /></label>
    <label>選擇整份資料夾<input type="file" multiple {...{ webkitdirectory: "", directory: "" }} aria-label={`${title}：選擇整份資料夾`} onChange={addFiles} /></label>
    <small>{files.length ? `已選擇 ${files.length} 張照片` : "可選擇多張圖片或整份資料夾（含子資料夾），支援 JPG、PNG、WebP；可繼續加入照片"}
      {skipped > 0 && `；已略過 ${skipped} 個非支援檔案`}</small>
    {files.length > 0 && <button type="button" onClick={() => { onChange([]); setSkipped(0); }}>清除{title}</button>}
  </div>;
}

function CombinedPhotoFolderInput({ rental, onChange }: { rental: RentalDraft; onChange: (changes: Partial<RentalDraft>) => void }) {
  const [summary, setSummary] = useState<string | null>(null);
  function selectFolder(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (!files.length) return;
    const images = files.filter(isHistoryPhoto);
    const evidence = mergeHistoryPhotos(rental.evidence, images);
    onChange({ evidence });
    setSummary(`已加入 ${evidence.length - rental.evidence.length} 張歷程照片。`
      + (files.length > images.length ? `已略過 ${files.length - images.length} 個非支援檔案。` : ""));
    event.target.value = "";
  }
  return <div>
    <label>一次上傳歷程資料夾<input type="file" multiple {...{ webkitdirectory: "", directory: "" }}
      aria-label="一次上傳歷程資料夾" onChange={selectFolder} /></label>
    <small>直接加入本次租借資料夾內的所有 JPG、PNG、WebP 照片（含子資料夾），不需分類，也不需完整借車、還車照片。已選擇 {rental.evidence.length} 張歷程照片。</small>
    {summary && <p role="status">{summary}</p>}
    {rental.evidence.length > 0 && <button type="button" onClick={() => { onChange({ evidence: [] }); setSummary(null); }}>清除一次上傳照片</button>}
  </div>;
}

export function VehicleHistoryManager() {
  const [plate, setPlate] = useState("");
  const [vehicleLabel, setVehicleLabel] = useState("");
  const [rentals, setRentals] = useState<RentalDraft[]>(() => [newRental()]);
  const [profiles, setProfiles] = useState<VehicleHistoryProfile[] | null>(null);
  const [historyExpanded, setHistoryExpanded] = useState(false);
  const historySectionId = useId();
  const [historyLoading, setHistoryLoading] = useState(false);
  const historySection = useRef<HTMLElement>(null);
  const historyRequest = useRef(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => {
    if (profiles && historyExpanded) historySection.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [profiles, historyExpanded]);
  function update(key: string, changes: Partial<RentalDraft>) {
    setRentals((current) => current.map((rental) => rental.key === key ? { ...rental, ...changes } : rental));
  }
  async function loadHistory() {
    const requestId = ++historyRequest.current;
    setHistoryLoading(true);
    setError(null);
    try {
      const result = await getSavedVehicleHistories();
      if (requestId === historyRequest.current) {
        setProfiles(result);
        setHistoryExpanded(true);
      }
    } catch (err) {
      if (requestId !== historyRequest.current) return;
      setError(err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError")
        ? "讀取歷程逾時，請再點選一次查詢。" : err instanceof Error ? err.message : "讀取歷程失敗");
    } finally {
      if (requestId === historyRequest.current) setHistoryLoading(false);
    }
  }
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busy || rentals.some((rental) => !rental.evidence.length && (!rental.pickup.length || !rental.returned.length))) return;
    setBusy(true); setError(null); setNotice(null);
    try {
      const records = rentals.map((rental) => {
        const record = { rental_id: rental.rental_id, description: rental.description,
          responsibility_confirmed: rental.confirmed, disputed: rental.disputed };
        return { data_kind: "vehicle_history", description: rental.description,
          rental: { rental_id: rental.rental_id, customer_id: rental.customer_id, display_name: rental.display_name,
            rental_date: rental.rental_date, return_date: rental.return_date,
            vehicle_label: vehicleLabel, license_plate: plate, password: String(crypto.getRandomValues(new Uint32Array(1))[0] % 10_000).padStart(4, "0") },
          ...(rental.kind === "damage" || rental.kind === "both" ? { damage: { ...record,
            compensation_amount: rental.compensation_amount, paid_amount: rental.paid_amount } } : {}),
          ...(rental.kind === "litter" || rental.kind === "both" ? { litter: record } : {}),
        };
      });
      await uploadVehicleHistory(records, rentals.map((rental) => ({ pickup: rental.pickup, returned: rental.returned, evidence: rental.evidence })));
      setNotice(`已保存 ${plate} 的 ${rentals.length} 筆租還紀錄與照片，可至借還車AI助理的「同車牌租借歷程」查詢。`);
      setRentals([newRental()]);
      void loadHistory();
    } catch (err) { setError(err instanceof Error ? err.message : "上傳失敗"); }
    finally { setBusy(false); }
  }
  return <section className="rag-seed-manager vehicle-history-manager">
    <div className="settings-group-heading"><p className="eyebrow">VEHICLE RENTAL HISTORY</p><h2>同車牌租借歷程</h2>
      <p>將同一台車的每次借還紀錄與照片上傳，指定當時的承租人，供 AI 助理依車牌查詢。可一次上傳現有歷程資料夾，不需分類或完整借還車照片；也可分別選擇借車、還車的多張圖片或資料夾。照片直接保存，略過辨識。</p><small>匯入歷程不會提供登入密碼；如需啟用租單登入，請由管理員在帳號管理重設密碼。</small></div>
    {error && <p className="rag-seed-error" role="alert">{error}</p>}
    {notice && <p className="rag-seed-notice" role="status">{notice}</p>}
    <form onSubmit={(event) => void submit(event)}>
      <fieldset className="rag-seed-fields" disabled={busy}><legend>同一車輛</legend>
        <div className="rag-seed-grid">
          <label>車牌號碼<input required minLength={5} maxLength={10} placeholder="請輸入新增租單的車牌" value={plate} onChange={(event) => setPlate(event.target.value.toUpperCase())} /></label>
          <label>車輛名稱<input required maxLength={120} placeholder="Toyota Yaris" value={vehicleLabel} onChange={(event) => setVehicleLabel(event.target.value)} /></label>
        </div>
      </fieldset>
      <button type="button" onClick={() => {
        if (historyExpanded) setHistoryExpanded(false);
        else if (profiles) setHistoryExpanded(true);
        else void loadHistory();
      }} disabled={historyLoading} aria-busy={historyLoading} aria-expanded={historyExpanded} aria-controls={historySectionId}>{historyLoading ? "讀取歷程中…" : historyExpanded ? "收合已建立的歷程" : "查看目前已建立的歷程"}</button>
      <small>直接顯示所有已上傳的歷程與照片，不需填寫車牌或新租單資料。</small>
      {profiles && historyExpanded && <section id={historySectionId} ref={historySection} aria-label="目前已建立的歷程">
        <h3>目前已建立 {profiles.reduce((total, profile) => total + profile.rental_count, 0)} 筆歷程 · {profiles.length} 個車牌</h3>
        {profiles.length === 0 && <p>目前尚未建立租借歷程。</p>}
        {profiles.map((profile) => <section key={profile.license_plate}>
          <h4>{profile.license_plate} · {profile.vehicle_label} · {profile.rental_count} 筆歷程</h4>
          <div className="rag-seed-records">{profile.rentals.map((rental) => <article key={rental.rental_id}>
            <strong>{rental.renter_name}（{rental.renter_id}）</strong><p>{rental.rental_date ?? "未填"} 借車 → {rental.return_date ?? "未填"} 還車</p><p>租單 {rental.rental_id}</p>
            {profile.events.filter((event) => event.rental_id === rental.rental_id && event.event_type === "rental_history_uploaded").map((event) => <RagSeedPhotos key={event.id} event={event} />)}
          </article>)}</div>
        </section>)}
      </section>}
      {rentals.map((rental, index) => <fieldset key={rental.key} className="rag-seed-fields vehicle-history-draft" disabled={busy}>
        <legend>第 {index + 1} 次租借</legend>
        <div className="rag-seed-grid">{([
          ["rental_id", "租單編號", "text", 80], ["customer_id", "當時承租人編號", "text", 80],
          ["display_name", "當時承租人姓名", "text", 120], ["rental_date", "借車日期", "date", 10],
          ["return_date", "還車日期", "date", 10],
        ] as const).map(([key, label, type, maxLength]) => <label key={key}>{label}<input required type={type} maxLength={maxLength}
          value={rental[key]} min={key === "return_date" ? rental.rental_date || undefined : undefined}
          onChange={(event) => update(rental.key, { [key]: event.target.value })} /></label>)}</div>
        <small>同一承租人的不同租單請填相同編號；借還日期與承租人會保存為當次歷程。</small>
        <label>當次車況<select aria-label="當次車況" value={rental.kind} onChange={(event) => update(rental.key, { kind: event.target.value })}>
          <option value="none">一般租借紀錄</option><option value="damage">造成車損</option><option value="litter">留下垃圾</option><option value="both">車損與垃圾</option>
        </select></label>
        <label>當次說明<textarea rows={2} maxLength={1000} required={rental.kind !== "none"} value={rental.description} onChange={(event) => update(rental.key, { description: event.target.value })} placeholder="例如：借車時乾淨，還車時後座留下飲料杯" /></label>
        {(rental.kind === "damage" || rental.kind === "both") && <div className="rag-seed-grid">
          <label>車損應賠金額<input required type="number" min="0" max="100000000" step="0.01" value={rental.compensation_amount} onChange={(event) => update(rental.key, { compensation_amount: event.target.value })} /></label>
          <label>車損已賠金額<input required type="number" min="0" max={rental.compensation_amount} step="0.01" value={rental.paid_amount} onChange={(event) => update(rental.key, { paid_amount: event.target.value })} /></label>
        </div>}
        {rental.kind !== "none" && <>
          <label className="rag-seed-check"><input type="checkbox" checked={rental.confirmed} onChange={(event) => update(rental.key, { confirmed: event.target.checked })} />已確認當時承租人責任</label>
          <label className="rag-seed-check"><input type="checkbox" checked={rental.disputed} onChange={(event) => update(rental.key, { disputed: event.target.checked })} />爭議中</label>
        </>}
        <CombinedPhotoFolderInput rental={rental} onChange={(changes) => update(rental.key, changes)} />
        <div className="rag-seed-grid"><PhotoUploadInput title="借車照片" files={rental.pickup} onChange={(files) => update(rental.key, { pickup: files })} /><PhotoUploadInput title="還車照片" files={rental.returned} onChange={(files) => update(rental.key, { returned: files })} /></div>
        {rentals.length > 1 && <button type="button" onClick={() => setRentals((current) => current.filter((item) => item.key !== rental.key))}>移除此筆待上傳租單</button>}
      </fieldset>)}
      <div className="vehicle-history-actions"><button type="button" disabled={busy || rentals.length >= 50} onClick={() => setRentals((current) => [...current, newRental()])}>＋ 加入下一次租借</button>
        <button className="primary" disabled={busy || rentals.some((rental) => !rental.evidence.length && (!rental.pickup.length || !rental.returned.length))}>{busy ? "保存整批歷程中…" : `上傳 ${rentals.length} 筆租借歷程`}</button></div>
      <small>每批最多 50 筆，可分批繼續上傳同一車牌。租單編號需唯一；若有重複或失敗，整批都不會寫入。同一承租人已確認的車損、垃圾與髒污合計達 {BLACKLIST_OFFENCES} 次會自動列入黑名單，可在「黑名單管理」查看。</small>
    </form>
  </section>;
}
