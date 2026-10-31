import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";

import {
  checkManualTruthDuplicates,
  ManualTruthBusyError,
  uploadManualTruths,
  type LearningTask,
  type ManualTruthDestination,
  type ManualTruthUploadResponse,
} from "../lib/api";

// The server accepts at most this many photos per request; larger selections are split.
const BATCH_SIZE = 30;
// Batches sent at the same time; the server answers 429 beyond its own limit.
const BATCH_CONCURRENCY = 2;
const BATCH_MAX_ATTEMPTS = 5;
const MAX_SELECTION = 600;
const PREVIEW_LIMIT = 5;
const IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const IMAGE_EXTENSIONS = /\.(jpe?g|png|webp)$/i;

export type ManualTruthTaskOption = {
  task: LearningTask;
  label: string;
  labels: { value: string; text: string }[];
  // Region-annotated datasets default to the pending list so boxes can be drawn.
  regionReview: boolean;
};

type SelectedPhoto = { file: File; hash: string };
type UploadResultItem = ManualTruthUploadResponse["results"][number];
type UploadProgress = { doneBatches: number; totalBatches: number; created: number; waiting: boolean };

const interiorSlots = [
  { value: "interior_front", text: "前座車內" },
  { value: "interior_rear", text: "後座車內" },
];

function usePreviewUrls(files: File[]): string[] {
  const urls = useMemo(() => files.map((file) => URL.createObjectURL(file)), [files]);
  useEffect(() => () => urls.forEach((url) => URL.revokeObjectURL(url)), [urls]);
  return urls;
}

function isImageFile(file: File): boolean {
  return IMAGE_TYPES.has(file.type) || (!file.type && IMAGE_EXTENSIONS.test(file.name));
}

function displayName(file: File): string {
  return file.webkitRelativePath || file.name;
}

// SubtleCrypto only exists in secure contexts; elsewhere the server still rejects duplicates.
async function photoHash(file: File): Promise<string> {
  if (!globalThis.crypto?.subtle) return `file:${file.name}:${file.size}:${file.lastModified}`;
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

const isContentHash = (hash: string) => /^[0-9a-f]{64}$/.test(hash);

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function uploadWithRetry(
  send: () => Promise<ManualTruthUploadResponse>,
  onWaiting: (waiting: boolean) => void,
): Promise<ManualTruthUploadResponse> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      const response = await send();
      onWaiting(false);
      return response;
    } catch (caught) {
      const busy = caught instanceof ManualTruthBusyError;
      const network = caught instanceof TypeError;
      if ((!busy && !network) || attempt >= BATCH_MAX_ATTEMPTS) throw caught;
      onWaiting(true);
      const backoff = busy ? caught.retryAfterMs : 1000 * 2 ** (attempt - 1);
      await wait(backoff + Math.random() * 500);
    }
  }
}

export function ManualTruthUpload({
  options,
  disabled,
  onUploaded,
}: {
  options: ManualTruthTaskOption[];
  disabled: boolean;
  onUploaded: (message: string) => Promise<void> | void;
}) {
  const initial = options.find((item) => item.task === "interior_cleanliness") ?? options[0];
  const [task, setTask] = useState<LearningTask>(initial?.task ?? "interior_cleanliness");
  const option = options.find((item) => item.task === task) ?? initial;
  const [label, setLabel] = useState(initial?.labels[0]?.value ?? "");
  const [slot, setSlot] = useState("interior_front");
  const [destination, setDestination] = useState<ManualTruthDestination>(
    initial?.regionReview ? "pending" : "truth",
  );
  const [photos, setPhotos] = useState<SelectedPhoto[]>([]);
  const [secondary, setSecondary] = useState<File | null>(null);
  const [notes, setNotes] = useState("");
  const [checking, setChecking] = useState<string | null>(null);
  const [selectionNotice, setSelectionNotice] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [skippedResults, setSkippedResults] = useState<UploadResultItem[]>([]);
  const folderInput = useRef<HTMLInputElement>(null);
  const files = useMemo(() => photos.map((photo) => photo.file), [photos]);
  const previewFiles = useMemo(() => files.slice(0, PREVIEW_LIMIT), [files]);
  const previews = usePreviewUrls(previewFiles);
  const secondaryFiles = useMemo(() => secondary ? [secondary] : [], [secondary]);
  const [secondaryPreview] = usePreviewUrls(secondaryFiles);
  const isPair = task === "damage_pair";
  const isInterior = task === "interior_cleanliness";

  // React has no typed prop for folder pickers.
  useEffect(() => {
    folderInput.current?.setAttribute("webkitdirectory", "");
    folderInput.current?.setAttribute("directory", "");
  }, [isPair]);

  if (!option) return null;

  function resetFeedback() {
    setError(null);
    setSkippedResults([]);
    setSelectionNotice(null);
  }

  function changeTask(next: LearningTask) {
    const nextOption = options.find((item) => item.task === next);
    setTask(next);
    setLabel(nextOption?.labels[0]?.value ?? "");
    setDestination(nextOption?.regionReview ? "pending" : "truth");
    setPhotos([]);
    setSecondary(null);
    resetFeedback();
  }

  async function addFiles(event: ChangeEvent<HTMLInputElement>) {
    const chosen = Array.from(event.target.files ?? []);
    event.target.value = "";
    resetFeedback();
    const images = chosen.filter(isImageFile);
    const notices: string[] = [];
    if (images.length < chosen.length) notices.push(`${chosen.length - images.length} 個非圖片檔已略過`);
    if (images.length === 0) {
      setSelectionNotice(notices.join("；") || null);
      return;
    }
    if (isPair) {
      setPhotos([{ file: images[0], hash: await photoHash(images[0]) }]);
      return;
    }

    setChecking(`檢查重複照片中（共 ${images.length} 張）…`);
    try {
      const known = new Set(photos.map((photo) => photo.hash));
      const added: SelectedPhoto[] = [];
      let repeatedInSelection = 0;
      for (const [index, file] of images.entries()) {
        if (index % 20 === 0) setChecking(`檢查重複照片中… ${index} / ${images.length}`);
        const hash = await photoHash(file);
        if (known.has(hash)) { repeatedInSelection += 1; continue; }
        known.add(hash);
        added.push({ file, hash });
      }
      if (repeatedInSelection) notices.push(`${repeatedInSelection} 張與已選照片重複`);

      const hashes = added.map((photo) => photo.hash).filter(isContentHash);
      let inDataset = new Set<string>();
      if (hashes.length > 0) {
        try {
          inDataset = new Set(await checkManualTruthDuplicates(task, hashes));
        } catch {
          notices.push("暫時無法向伺服器確認重複，上傳時仍會自動排除");
        }
      }
      const fresh = added.filter((photo) => !inDataset.has(photo.hash));
      if (inDataset.size) notices.push(`${added.length - fresh.length} 張已在「${option!.label}」資料集中`);

      const merged = [...photos, ...fresh];
      if (merged.length > MAX_SELECTION) notices.push(`一次最多選 ${MAX_SELECTION} 張，已保留前 ${MAX_SELECTION} 張`);
      setPhotos(merged.slice(0, MAX_SELECTION));
      setSelectionNotice(notices.length ? `已阻擋：${notices.join("；")}。` : null);
    } finally {
      setChecking(null);
    }
  }

  function removeFile(index: number) {
    setPhotos((current) => current.filter((_, position) => position !== index));
  }

  const labelName = option.labels.find((item) => item.value === label)?.text ?? label;
  const slotName = interiorSlots.find((item) => item.value === slot)?.text;
  const targetName = `${option.label}${isInterior && slotName ? `・${slotName}` : ""}・${labelName}`;
  const ready = Boolean(label) && photos.length > 0 && (!isPair || secondary !== null) && !checking;
  const batchCount = Math.ceil(photos.length / BATCH_SIZE);

  async function submit() {
    if (!ready) return;
    setUploading(true);
    resetFeedback();
    const batches: SelectedPhoto[][] = [];
    for (let start = 0; start < photos.length; start += BATCH_SIZE) {
      batches.push(photos.slice(start, start + BATCH_SIZE));
    }
    const counter = { done: 0, created: 0, waiting: 0 };
    const report = () => setProgress({
      doneBatches: counter.done, totalBatches: batches.length, created: counter.created, waiting: counter.waiting > 0,
    });
    report();
    const results: UploadResultItem[] = [];
    const unsent: SelectedPhoto[] = [];
    let lastError: string | null = null;
    let next = 0;

    async function worker() {
      while (next < batches.length) {
        const batch = batches[next];
        next += 1;
        let isWaiting = false;
        try {
          const response = await uploadWithRetry(() => uploadManualTruths({
            task,
            correctedLabel: label,
            destination,
            images: batch.map((photo) => photo.file),
            secondaryImage: isPair ? secondary : null,
            inspectionSlot: isInterior ? slot : null,
            reviewerNotes: notes.trim(),
          }), (waiting) => {
            if (waiting !== isWaiting) { counter.waiting += waiting ? 1 : -1; isWaiting = waiting; report(); }
          });
          results.push(...response.results);
          counter.created += response.created;
        } catch (caught) {
          if (isWaiting) counter.waiting -= 1;
          lastError = caught instanceof Error ? caught.message : "上傳待標註真值失敗";
          unsent.push(...batch);
        }
        counter.done += 1;
        report();
      }
    }

    try {
      await Promise.all(Array.from({ length: Math.min(BATCH_CONCURRENCY, batches.length) }, worker));
      // Keep photos from failed batches selected so they can be retried.
      setPhotos(unsent);
      if (unsent.length === 0) setSecondary(null);
      setSkippedResults(results.filter((item) => item.status !== "created"));
      if (lastError) setError(`${unsent.length} 張未送出（${lastError}），已保留在清單中，可再按上傳重試。`);
      const skipped = results.length - counter.created;
      const where = destination === "truth" ? "已加入人工真值" : "已放入待處理，請到下方逐張框選";
      if (counter.created > 0 || results.length > 0) {
        await onUploaded(`${targetName}：${counter.created} ${isPair ? "組" : "張"}${where}${skipped ? `，${skipped} 張未加入` : ""}。`);
      }
    } finally {
      setUploading(false);
      setProgress(null);
    }
  }

  const busy = disabled || uploading || checking !== null;

  return <section className="manual-truth-upload" aria-labelledby="manual-truth-upload-title">
    <div className="manual-truth-upload__heading">
      <p className="eyebrow">MANUAL TRUTH UPLOAD</p>
      <h3 id="manual-truth-upload-title">上傳待標註真值</h3>
      <p>自己挑選的照片直接標註答案，不經過任何模型判斷。可選整個資料夾，超過 {BATCH_SIZE} 張會自動分批上傳；重複的照片會被阻擋。車內清潔需指定前座或後座；想框出灰塵、腳印、皮椅污漬等位置時，請選「先放入待處理」。</p>
    </div>

    <div className="manual-truth-upload__fields">
      <label>
        <span>資料集</span>
        <select value={task} disabled={busy} onChange={(event) => changeTask(event.target.value as LearningTask)}>
          {options.map((item) => <option key={item.task} value={item.task}>{item.label}</option>)}
        </select>
      </label>
      <label>
        <span>人工真值</span>
        <select value={label} disabled={busy} onChange={(event) => { setLabel(event.target.value); setSkippedResults([]); }}>
          {option.labels.map((item) => <option key={item.value} value={item.value}>{item.text}</option>)}
        </select>
      </label>
      {isInterior && <fieldset className="manual-truth-upload__choice">
        <legend>拍攝位置</legend>
        {interiorSlots.map((item) => <label key={item.value}>
          <input type="radio" name="manual-truth-slot" value={item.value} checked={slot === item.value} disabled={busy} onChange={() => setSlot(item.value)} />
          <span>{item.text}</span>
        </label>)}
      </fieldset>}
      <fieldset className="manual-truth-upload__choice">
        <legend>上傳後</legend>
        <label>
          <input type="radio" name="manual-truth-destination" checked={destination === "pending"} disabled={busy} onChange={() => setDestination("pending")} />
          <span>先放入待處理，逐張框選位置</span>
        </label>
        <label>
          <input type="radio" name="manual-truth-destination" checked={destination === "truth"} disabled={busy} onChange={() => setDestination("truth")} />
          <span>直接成為人工真值</span>
        </label>
      </fieldset>
    </div>

    <div className="manual-truth-upload__pickers manual-truth-upload__pickers--pair">
      <label className="manual-truth-upload__picker">
        <input type="file" accept="image/jpeg,image/png,image/webp" multiple={!isPair} disabled={busy} onChange={(event) => void addFiles(event)} hidden />
        <strong>{isPair ? "＋ 選擇取車照" : "＋ 選擇照片"}</strong>
        <small>{isPair ? "取車／基準照 1 張" : "可多選 JPG、PNG、WebP"}</small>
      </label>
      {isPair
        ? <label className="manual-truth-upload__picker">
          <input type="file" accept="image/jpeg,image/png,image/webp" disabled={busy} onChange={(event) => { setSecondary(event.target.files?.[0] ?? null); event.target.value = ""; resetFeedback(); }} hidden />
          <strong>＋ 選擇還車照</strong>
          <small>還車／對比照 1 張，角度需與取車照相同</small>
        </label>
        : <label className="manual-truth-upload__picker">
          <input ref={folderInput} type="file" multiple disabled={busy} onChange={(event) => void addFiles(event)} hidden />
          <strong>＋ 選擇整個資料夾</strong>
          <small>自動挑出資料夾內的圖片，最多 {MAX_SELECTION} 張</small>
        </label>}
    </div>

    {checking && <p className="manual-truth-upload__status" role="status">{checking}</p>}
    {selectionNotice && <p className="manual-truth-upload__status manual-truth-upload__status--warn" role="status">{selectionNotice}</p>}

    {(photos.length > 0 || secondary) && <>
      {!isPair && <div className="manual-truth-upload__summary">
        <span>已選 {photos.length} 張{batchCount > 1 ? `，將分 ${batchCount} 批上傳（每批 ${BATCH_SIZE} 張）` : ""}</span>
        <button type="button" disabled={busy} onClick={() => { setPhotos([]); resetFeedback(); }}>清除全部</button>
      </div>}
      <ul className="manual-truth-upload__files">
        {previewFiles.map((file, index) => <li key={`${photos[index].hash}-${index}`}>
          <img src={previews[index]} alt={file.name} />
          <span title={displayName(file)}>{isPair ? "取車照・" : ""}{displayName(file)}</span>
          <button type="button" disabled={busy} aria-label={`移除 ${file.name}`} onClick={() => removeFile(index)}>×</button>
        </li>)}
        {isPair && secondary && <li>
          <img src={secondaryPreview} alt={secondary.name} />
          <span>還車照・{secondary.name}</span>
          <button type="button" disabled={busy} aria-label={`移除 ${secondary.name}`} onClick={() => setSecondary(null)}>×</button>
        </li>}
      </ul>
      {photos.length > PREVIEW_LIMIT && <p className="manual-truth-upload__status">另有 {photos.length - PREVIEW_LIMIT} 張未顯示預覽，仍會一起上傳。</p>}
    </>}

    <label className="manual-truth-upload__notes">
      <span>備註（選填）</span>
      <input maxLength={500} value={notes} disabled={busy} placeholder="例如：Toyota Altis 後座腳踏墊泥沙" onChange={(event) => setNotes(event.target.value)} />
    </label>

    <button className="manual-truth-upload__submit" type="button" disabled={busy || !ready} onClick={() => void submit()}>
      {progress
        ? `上傳中… 第 ${Math.min(progress.doneBatches + 1, progress.totalBatches)} / ${progress.totalBatches} 批，已新增 ${progress.created} 張${progress.waiting ? "（伺服器忙碌，稍後自動重試）" : ""}`
        : ready
          ? `上傳${isPair ? " 1 組" : ` ${photos.length} 張`}為「${targetName}」`
          : isPair ? "請選擇取車照與還車照" : "請先選擇照片"}
    </button>
    {progress && <progress className="manual-truth-upload__progress" max={progress.totalBatches} value={progress.doneBatches} />}

    {error && <div className="error-box" role="alert"><strong>上傳未完成</strong><span>{error}</span></div>}
    {skippedResults.length > 0 && <ul className="manual-truth-upload__skipped" role="status">
      {skippedResults.map((item, index) => <li key={`${item.filename}-${index}`}>
        <strong>{item.status === "duplicate" ? "已存在" : "失敗"}</strong>
        <span>{item.filename}：{item.message}</span>
      </li>)}
    </ul>}
  </section>;
}
