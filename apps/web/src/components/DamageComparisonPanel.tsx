import { ChangeEvent, useEffect, useMemo, useState } from "react";

import {
  compareDamagePair,
  type DamageCase,
  type ExpectedView,
} from "../lib/api";
import { ZoomableImage } from "./ImageViewer";
import { confirmRemoval } from "../lib/confirmRemoval";

const MAX_DAMAGE_PHOTOS = 5;
const exteriorViews: Array<{ value: ExpectedView; label: string }> = [
  { value: "left_front", label: "左前方" },
  { value: "right_front", label: "右前方" },
  { value: "left_rear", label: "左後方" },
  { value: "right_rear", label: "右後方" },
];
const defaultPairViews = () => Array<ExpectedView>(MAX_DAMAGE_PHOTOS).fill("left_front");

type PhotoSlotProps = {
  label: string;
  hint: string;
  files: File[];
  disabled: boolean;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onRemove: (index: number) => void;
  onClear: () => void;
};

function PhotoSlot({
  label,
  hint,
  files,
  disabled,
  onChange,
  onRemove,
  onClear,
}: PhotoSlotProps) {
  const previews = useMemo(() => files.map((file) => URL.createObjectURL(file)), [files]);

  useEffect(() => () => previews.forEach((preview) => URL.revokeObjectURL(preview)), [previews]);

  return (
    <article className={`damage-upload-slot ${files.length ? "damage-upload-slot--ready" : ""}`}>
      <div className="damage-upload-slot__heading">
        <div><strong>{label}</strong><span>{hint}</span></div>
        <div className="damage-upload-slot__count">
          <span>{files.length} / {MAX_DAMAGE_PHOTOS} 張</span>
          {files.length > 0 && <button type="button" onClick={onClear} disabled={disabled}>全部移除</button>}
        </div>
      </div>

      {files.length > 0 && <div className="damage-upload-slot__previews">
        {files.map((file, index) => <div key={`${file.name}-${file.size}-${file.lastModified}-${index}`}>
          <ZoomableImage
            src={previews[index]}
            alt={`${label}第 ${index + 1} 張預覽`}
            title={`${label} ${index + 1} · ${file.name}`}
            overlay={<span>第 {index + 1} 張 · 點擊放大</span>}
          />
          <button type="button" onClick={() => onRemove(index)} disabled={disabled}>移除</button>
        </div>)}
      </div>}

      {files.length < MAX_DAMAGE_PHOTOS && <label className={`damage-upload-slot__picker ${files.length ? "damage-upload-slot__picker--more" : ""}`}>
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          capture="environment"
          multiple
          onChange={onChange}
          disabled={disabled}
          hidden
        />
        <span>
          <strong>＋ {files.length ? "繼續新增照片" : "選擇照片"}</strong>
          <small>可一次選取多張 · 每區最多 {MAX_DAMAGE_PHOTOS} 張</small>
        </span>
      </label>}
    </article>
  );
}

function resultTitle(result: DamageCase): string {
  if (result.comparison.status === "no_change") return "通過：未發現新增車損";
  if (result.comparison.status === "review_required") return "不通過：發現新增車損，已送人工覆核";
  return "不通過：照片無法比對，請依同角度重拍";
}

function verificationLabel(decision: NonNullable<DamageCase["comparison"]["verification"]>["decision"]): string {
  if (decision === "confirmed_new_damage") return "確認新增物理損傷";
  if (decision === "no_new_damage") return "未確認新增損傷";
  return "照片不足以判定";
}

export function DamageComparisonPanel() {
  const [pairViews, setPairViews] = useState<ExpectedView[]>(defaultPairViews);
  const [pickupFiles, setPickupFiles] = useState<File[]>([]);
  const [returnFiles, setReturnFiles] = useState<File[]>([]);
  const [results, setResults] = useState<DamageCase[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [submittedCount, setSubmittedCount] = useState(0);

  function chooseFiles(slot: "pickup" | "return", event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    const current = slot === "pickup" ? pickupFiles : returnFiles;
    const remaining = MAX_DAMAGE_PHOTOS - current.length;
    const additions = selected.slice(0, remaining);
    const next = [...current, ...additions];
    if (slot === "pickup") setPickupFiles(next);
    else setReturnFiles(next);
    setResults([]);
    setError(selected.length > remaining
      ? `每個上傳區最多 ${MAX_DAMAGE_PHOTOS} 張；已加入前 ${remaining} 張。`
      : null);
    event.target.value = "";
  }

  function removeFile(slot: "pickup" | "return", index: number) {
    const label = slot === "pickup" ? "取車" : "還車";
    if (!confirmRemoval(`只會移除目前選取的第 ${index + 1} 張${label}比對照片並清除本頁比對結果；電腦原檔及已保存的資料不受影響。`)) return;
    if (slot === "pickup") setPickupFiles((current) => current.filter((_, itemIndex) => itemIndex !== index));
    else setReturnFiles((current) => current.filter((_, itemIndex) => itemIndex !== index));
    setPairViews((current) => current.map((view, viewIndex) =>
      viewIndex >= index ? "left_front" : view));
    setResults([]);
    setError(null);
  }

  function clearFiles(slot: "pickup" | "return") {
    const label = slot === "pickup" ? "取車" : "還車";
    const count = slot === "pickup" ? pickupFiles.length : returnFiles.length;
    if (!confirmRemoval(`只會移除目前選取的全部 ${count} 張${label}比對照片並清除本頁比對結果；另一區、電腦原檔及已保存的資料不受影響。`)) return;
    if (slot === "pickup") setPickupFiles([]);
    else setReturnFiles([]);
    setPairViews(defaultPairViews());
    setResults([]);
    setError(null);
  }

  async function compare() {
    if (!pickupFiles.length || !returnFiles.length) {
      setError("請至少各選擇 1 張取車照與還車照。");
      return;
    }
    if (pickupFiles.length !== returnFiles.length) {
      setError("取車照與還車照張數必須相同，系統會依選取順序一對一比對。");
      return;
    }
    const submittedPickupFiles = pickupFiles;
    const submittedReturnFiles = returnFiles;
    const submittedPairViews = pairViews;
    setSubmittedCount(submittedPickupFiles.length);
    setLoading(true);
    setError(null);
    setResults([]);
    setPickupFiles([]);
    setReturnFiles([]);
    setPairViews(defaultPairViews());
    const completed: DamageCase[] = [];
    try {
      for (let index = 0; index < submittedPickupFiles.length; index += 1) {
        try {
          completed.push(await compareDamagePair(
            submittedPickupFiles[index],
            submittedReturnFiles[index],
            submittedPairViews[index],
          ));
          setResults([...completed]);
        } catch (caught) {
          const message = caught instanceof Error ? caught.message : "車損比對失敗，請稍後再試。";
          throw new Error(`第 ${index + 1} 組比對失敗：${message}`);
        }
      }
    } catch (caught) {
      setPickupFiles(submittedPickupFiles);
      setReturnFiles(submittedReturnFiles);
      setPairViews(submittedPairViews);
      setError(caught instanceof Error ? caught.message : "車損比對失敗，請稍後再試。");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="damage-comparison-section" aria-labelledby="damage-comparison-title">
      <div className="damage-comparison-section__intro">
        <p className="eyebrow">取車 × 還車</p>
        <div className="test-section-title-row">
          <h2 id="damage-comparison-title">車損比對專區</h2>
          <span className="development-test-badge">開發測試用</span>
        </div>
        <p>
          每批最多可比對五組取還車照片，並為每組選擇拍攝角度；系統會依選取順序一對一比對，
          分開辨識物理損傷與陰影、反光、雨痕及車身原有線條。結果只供車況覆核，不會自動判責。
        </p>
      </div>

      <div className="damage-comparison-card">
        <div className="damage-comparison-toolbar">
          <span>每區最多 5 張 · 張數相同 · 依順序配對 · 每組須同車、同側、同角度</span>
        </div>

        <div className="damage-upload-grid">
          <PhotoSlot
            label="取車基準照"
            hint="車損發生前 · 第 1 張會與還車第 1 張比對"
            files={pickupFiles}
            disabled={loading}
            onChange={(event) => chooseFiles("pickup", event)}
            onRemove={(index) => removeFile("pickup", index)}
            onClear={() => clearFiles("pickup")}
          />
          <PhotoSlot
            label="還車比對照"
            hint="車損發生後 · 請依取車照順序選取"
            files={returnFiles}
            disabled={loading}
            onChange={(event) => chooseFiles("return", event)}
            onRemove={(index) => removeFile("return", index)}
            onClear={() => clearFiles("return")}
          />
        </div>

        {Math.max(pickupFiles.length, returnFiles.length) > 0 && <div className="damage-pair-views">
          {Array.from({ length: Math.max(pickupFiles.length, returnFiles.length) }, (_, index) => (
            <label key={index}>
              <span>第 {index + 1} 組比對角度</span>
              <select
                value={pairViews[index]}
                onChange={(event) => {
                  setPairViews((current) => current.map((view, viewIndex) =>
                    viewIndex === index ? event.target.value as ExpectedView : view));
                  setResults([]);
                }}
                disabled={loading}
              >
                {exteriorViews.map((view) => (
                  <option key={view.value} value={view.value}>{view.label}</option>
                ))}
              </select>
            </label>
          ))}
        </div>}

        <button
          className="damage-compare-button"
          type="button"
          onClick={() => void compare()}
          disabled={loading || !pickupFiles.length || !returnFiles.length}
        >
          {loading
            ? `正在逐組比對（已完成 ${results.length} / ${submittedCount}）…`
            : pickupFiles.length && returnFiles.length
              ? `開始比對 ${Math.min(pickupFiles.length, returnFiles.length)} 組車損`
              : "選取照片後開始比對"}
        </button>
        {error && <div className="error-box"><strong>無法完成全部比對</strong><span>{error}</span></div>}

        {results.length > 0 && <div className="damage-comparison-results">
          {results.map((result, index) => {
            const verification = result.comparison.verification;
            return <div className={`damage-comparison-result damage-comparison-result--${result.comparison.status}`} key={result.id}>
              <div className="damage-comparison-result__copy">
                <span>第 {index + 1} 組 · {exteriorViews.find((view) => view.value === result.expected_view)?.label} · 案件 {result.id}</span>
                <h3>{resultTitle(result)}</h3>
                <p>{result.comparison.summary}</p>
                <dl>
                  <div><dt>像素候選區</dt><dd>{result.comparison.candidates.length} 處</dd></div>
                  <div><dt>照片對位</dt><dd>{Math.round(result.comparison.alignment.score * 100)}%</dd></div>
                  {verification && <div>
                    <dt>物理車損覆核</dt>
                    <dd>{`${verificationLabel(verification.decision)} · ${Math.round(verification.confidence * 100)}%`}</dd>
                  </div>}
                  {result.comparison.specialist && <div>
                    <dt>車損比對模型</dt>
                    <dd>{result.comparison.specialist.detected ? "偵測到新增車損" : "未達門檻"} · {Math.round(result.comparison.specialist.score * 100)}%</dd>
                  </div>}
                </dl>
                <small>像素變化包含光影與拍攝位移；上方百分比不代表肇責或索賠機率。</small>
              </div>
              {result.comparison.annotated_return_image && (
                <figure>
                  <ZoomableImage
                    src={result.comparison.annotated_return_image}
                    alt={`第 ${index + 1} 組還車照片疑似變化區標記`}
                    title={`第 ${index + 1} 組車損比對結果`}
                    overlay={<span>⌕ 點擊放大結果</span>}
                  />
                  <figcaption>{result.comparison.candidates[0]?.id.startsWith("AI-")
                    ? "橘框為車損比對模型標示的區域，需人工覆核"
                    : "紅框僅代表像素變化候選，仍須物理損傷覆核"}</figcaption>
                </figure>
              )}
            </div>;
          })}
        </div>}
      </div>
    </section>
  );
}
