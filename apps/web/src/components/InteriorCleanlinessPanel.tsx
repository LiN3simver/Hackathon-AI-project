import { ChangeEvent, useEffect, useMemo, useState } from "react";

import {
  analyzeInteriorCleanliness,
  type InteriorAnalysisResponse,
  type InteriorView,
} from "../lib/api";
import { CleanlinessFeedback } from "./CleanlinessPhoto";
import { ZoomableImage } from "./ImageViewer";
import { confirmRemoval } from "../lib/confirmRemoval";

type TestableInteriorView = Exclude<InteriorView, "unknown">;

type CleanlinessUploadCardProps = {
  view: TestableInteriorView;
  label: string;
  hint: string;
};

function CleanlinessUploadCard({ view, label, hint }: CleanlinessUploadCardProps) {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<InteriorAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const preview = useMemo(() => file ? URL.createObjectURL(file) : null, [file]);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    setResult(null);
    setError(null);
    event.target.value = "";
  }

  function removeFile() {
    if (!confirmRemoval(`只會移除目前選取的${label}測試照片與本頁分析結果；電腦原檔及已保存的資料不受影響。`)) return;
    setFile(null);
    setResult(null);
    setError(null);
  }

  async function analyze() {
    if (!file) {
      setError(`請先選擇 1 張${label}照片。`);
      return;
    }
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      setResult(await analyzeInteriorCleanliness(file, view));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : `${label}整潔辨識失敗，請稍後再試。`);
    } finally {
      setLoading(false);
    }
  }


  return (
    <article className={`cleanliness-upload-card ${preview ? "cleanliness-upload-card--ready" : ""}`}>
      <div className="cleanliness-upload-card__heading">
        <div><strong>{label}</strong><span>{hint}</span></div>
        {file && <button type="button" onClick={removeFile} disabled={loading}>移除</button>}
      </div>

      <label className="cleanliness-upload-card__picker">
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          capture="environment"
          onChange={chooseFile}
          disabled={loading}
          hidden
        />
        {preview ? (
          <img src={preview} alt={`${label}整潔辨識照片預覽`} />
        ) : (
          <span><strong>＋ 上傳{label}照片</strong><small>限 1 張 JPG、PNG 或 WebP</small></span>
        )}
      </label>
      {file && <small className="cleanliness-upload-card__filename">{file.name}</small>}

      <button
        className="cleanliness-analyze-button"
        type="button"
        onClick={() => void analyze()}
        disabled={loading || !file}
      >
        {loading ? `正在辨識${label}整潔度…` : `開始${label}整潔辨識`}
      </button>

      {error && <div className="error-box"><strong>無法完成整潔辨識</strong><span>{error}</span></div>}

      {result && (
        <div className={`cleanliness-result cleanliness-result--${result.assessment.level}`} role="status">
          <strong>整潔辨識結果</strong>
          {preview && <ZoomableImage src={preview} alt={`${label}疑似髒汙與雜物位置`}
            evidence={result.assessment.evidence} />}
          <CleanlinessFeedback assessment={result.assessment} />
        </div>
      )}
    </article>
  );
}

export function InteriorCleanlinessPanel() {
  return (
    <section className="cleanliness-section" aria-labelledby="cleanliness-section-title">
      <div className="cleanliness-section__intro">
        <p className="eyebrow">前座 × 後座</p>
        <div className="test-section-title-row">
          <h2 id="cleanliness-section-title">車內整潔辨識專區</h2>
          <span className="development-test-badge">開發測試用</span>
        </div>
        <p>
          前座與後座分別辨識疑似髒汙與雜物，並顯示可確認的框選位置。
        </p>
      </div>

      <div className="cleanliness-upload-grid">
        <CleanlinessUploadCard
          view="front"
          label="前座車內"
          hint="拍入駕駛座、副駕座與前座腳踏區"
        />
        <CleanlinessUploadCard
          view="rear"
          label="後座車內"
          hint="拍入後座椅面、椅背與後座腳踏區"
        />
      </div>
    </section>
  );
}
