import { useEffect, useState, type CSSProperties } from "react";
import type { CleanlinessEvidence, InteriorAnalysisResponse, PlatePrediction } from "../lib/api";

export type CleanlinessAssessment = Partial<InteriorAnalysisResponse["assessment"]>;

/** 通過 (null) or the one concrete reason the interior did not pass. */
export function cleanlinessFeedback(assessment?: CleanlinessAssessment, pending = false): string | null {
  if (pending) return "此座位辨識尚未啟用。";
  if (!assessment) return "整潔辨識失敗，請重拍。";
  if (assessment.image_usable === false || assessment.level === "cannot_assess") {
    return "看不清座椅與腳踏區，請重拍。";
  }
  const items = assessment.evidence?.filter((item) => item.visible && item.bounding_box) ?? [];
  if (items.length) return `發現 ${items.length} 處髒汙或雜物，請清理框選處。`;
  if (assessment.specialist?.suspected_dirty || assessment.level === "dirty") return "發現髒汙，請清理後重拍。";
  if (assessment.localization_status !== "completed") return "整潔辨識失敗，請重拍。";
  return null;
}

/** The box OCR read the plate from, labelled with the text it read. */
function plateRegion(plate: PlatePrediction | null | undefined) {
  const box = plate?.bounding_box;
  if (!plate?.plate_detected || !box || box.length !== 4 || !box.every(Number.isFinite)) return null;
  const [x1, y1, x2, y2] = box.map((value) => Math.max(0, Math.min(1, value)));
  if (x2 <= x1 || y2 <= y1) return null;
  const readable = plate.is_confident && Boolean(plate.normalized_plate) && !plate.timed_out;
  return { x1, y1, x2, y2, readable, label: plate.normalized_plate ?? plate.plate_text ?? "車牌" };
}

export function CleanlinessPhoto({ src, alt, evidence = [], plate, className = "", style, onError }: {
  src: string; alt: string; evidence?: CleanlinessEvidence[]; plate?: PlatePrediction | null;
  className?: string; style?: CSSProperties; onError?: () => void;
}) {
  const [size, setSize] = useState({ width: 1000, height: 750 });
  useEffect(() => {
    let active = true;
    const image = new Image();
    image.onload = () => { if (active) setSize({ width: image.naturalWidth, height: image.naturalHeight }); };
    image.src = src;
    return () => { active = false; };
  }, [src]);
  const regions = evidence.filter((item) => {
    const box = item.bounding_box;
    return item.visible && box && [box.x1, box.y1, box.x2, box.y2].every(Number.isFinite)
      && box.x1 >= 0 && box.y1 >= 0 && box.x2 <= 1 && box.y2 <= 1
      && box.x2 > box.x1 && box.y2 > box.y1;
  });
  const plateBox = plateRegion(plate);
  if (!regions.length && !plateBox) return <img src={src} alt={alt} className={className} style={style} onError={onError} draggable={false} />;
  const unit = Math.max(size.width, size.height) / 600;
  const plateColor = plateBox?.readable ? "#3ddc84" : "#f5b942";
  return <svg className={`cleanliness-photo ${className}`} style={style}
    viewBox={`0 0 ${size.width} ${size.height}`} role="img" aria-label={alt}>
    <title>{alt}{regions.length > 0 && `；框選 ${regions.length} 處疑似髒汙或雜物`}{plateBox && `；車牌 ${plateBox.label}`}</title>
    <image href={src} width={size.width} height={size.height} onError={onError} />
    {plateBox && <g>
      <title>OCR 讀到的車牌：{plateBox.label}</title>
      <rect x={plateBox.x1 * size.width} y={plateBox.y1 * size.height}
        width={(plateBox.x2 - plateBox.x1) * size.width} height={(plateBox.y2 - plateBox.y1) * size.height}
        fill="none" stroke={plateColor} strokeWidth={3 * unit} />
      <rect x={plateBox.x1 * size.width} y={Math.max(0, plateBox.y1 * size.height - 24 * unit)}
        width={(plateBox.label.length * 10 + 12) * unit} height={22 * unit} rx={3 * unit} fill={plateColor} />
      <text x={plateBox.x1 * size.width + 6 * unit} y={Math.max(0, plateBox.y1 * size.height - 24 * unit) + 16 * unit}
        fill="#111" fontSize={15 * unit} fontWeight="bold">{plateBox.label}</text>
    </g>}
    {regions.map((item, index) => {
      const box = item.bounding_box!;
      const x = box.x1 * size.width, y = box.y1 * size.height;
      const labelY = Math.max(0, y - 24 * unit);
      return <g key={index}>
        <title>{item.description}</title>
        <rect x={x} y={y} width={(box.x2 - box.x1) * size.width} height={(box.y2 - box.y1) * size.height}
          fill="rgba(255,62,62,.10)" stroke="#ff3e3e" strokeWidth={3 * unit} />
        <rect x={x} y={labelY} width={25 * unit} height={24 * unit} rx={3 * unit} fill="#b41622" />
        <text x={x + 12.5 * unit} y={labelY + 17 * unit} fill="white" textAnchor="middle"
          fontSize={16 * unit} fontWeight="bold">{index + 1}</text>
      </g>;
    })}
  </svg>;
}

export function Verdict({ reason }: { reason: string | null }) {
  return <p className={`result-verdict result-verdict--${reason ? "fail" : "pass"}`}>
    <b>{reason ? "不通過" : "通過"}</b>{reason && `：${reason}`}
  </p>;
}

export function CleanlinessFeedback({ assessment, pending }: { assessment?: CleanlinessAssessment; pending?: boolean }) {
  return <Verdict reason={cleanlinessFeedback(assessment, pending)} />;
}
