import { useEffect, useRef, useState } from "react";
import type { PointerEvent, ReactNode, WheelEvent } from "react";
import { createPortal } from "react-dom";

type Point = { x: number; y: number };
export type ImageHighlight = {
  x: number;
  y: number;
  width: number;
  height: number;
  kind: "issue" | "hard_negative" | "reference" | "model";
  label: string;
  points?: Point[];
};

export function ZoomableImage({
  src,
  alt,
  title,
  className = "",
  imageClassName = "",
  overlay,
  highlights,
  fallback,
}: {
  src: string;
  alt: string;
  title?: string;
  className?: string;
  imageClassName?: string;
  overlay?: ReactNode;
  highlights?: ImageHighlight[];
  fallback?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
  const [loadedSize, setLoadedSize] = useState<(Point & { src: string }) | null>(null);
  const dragRef = useRef<{ pointer: Point; offset: Point } | null>(null);

  function resetView() {
    setScale(1);
    setOffset({ x: 0, y: 0 });
    dragRef.current = null;
  }

  function close() {
    setOpen(false);
    resetView();
  }

  function markImageFailed() {
    setImageFailed(true);
    close();
  }

  function zoomTo(next: number) {
    const bounded = Math.min(5, Math.max(0.5, Number(next.toFixed(2))));
    setScale(bounded);
    if (bounded === 1) setOffset({ x: 0, y: 0 });
  }

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") close();
      if (event.key === "+" || event.key === "=") zoomTo(scale + 0.25);
      if (event.key === "-") zoomTo(scale - 0.25);
      if (event.key === "0") resetView();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open, scale]);

  useEffect(() => {
    setImageFailed(false);
  }, [src]);

  const hasHighlights = Boolean(highlights?.length);
  const highlightLabels = [...new Set((highlights ?? []).map((highlight) => highlight.label.trim()).filter(Boolean))];
  const naturalSize = loadedSize?.src === src ? loadedSize : null;
  const highlightLayer = hasHighlights && naturalSize ? <svg
    className="zoomable-image-highlights"
    viewBox={`0 0 ${naturalSize.x} ${naturalSize.y}`}
    preserveAspectRatio="xMidYMid meet"
    aria-hidden="true"
  >
    {highlights!.map((highlight, index) => highlight.points?.length ? <polygon
      key={`${highlight.label}-${index}`}
      className={`zoomable-image-highlight zoomable-image-highlight--${highlight.kind}`}
      points={highlight.points.map((point) => `${point.x * naturalSize.x},${point.y * naturalSize.y}`).join(" ")}
    /> : <rect
      key={`${highlight.label}-${index}`}
      className={`zoomable-image-highlight zoomable-image-highlight--${highlight.kind}`}
      x={highlight.x * naturalSize.x}
      y={highlight.y * naturalSize.y}
      width={highlight.width * naturalSize.x}
      height={highlight.height * naturalSize.y}
    />)}
  </svg> : null;

  function onWheel(event: WheelEvent<HTMLDivElement>) {
    event.preventDefault();
    zoomTo(scale + (event.deltaY < 0 ? 0.25 : -0.25));
  }

  function startDrag(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointer: { x: event.clientX, y: event.clientY },
      offset,
    };
  }

  function drag(event: PointerEvent<HTMLDivElement>) {
    if (!dragRef.current) return;
    setOffset({
      x: dragRef.current.offset.x + event.clientX - dragRef.current.pointer.x,
      y: dragRef.current.offset.y + event.clientY - dragRef.current.pointer.y,
    });
  }

  const viewer = open ? <div
    className="image-viewer-backdrop"
    role="dialog"
    aria-modal="true"
    aria-label={`${title ?? alt}圖片檢視器`}
    onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}
  >
    <div className="image-viewer-shell">
      <div className="image-viewer-toolbar">
        <strong>{title ?? alt}</strong>
        <div>
          <button type="button" onClick={() => zoomTo(scale - 0.25)} aria-label="縮小圖片">−</button>
          <output aria-live="polite">{Math.round(scale * 100)}%</output>
          <button type="button" onClick={() => zoomTo(scale + 0.25)} aria-label="放大圖片">＋</button>
          <button type="button" onClick={resetView}>重設</button>
          <button className="image-viewer-close" type="button" onClick={close} aria-label="關閉圖片檢視器">×</button>
        </div>
      </div>
      <div
        className="image-viewer-stage image-viewer-stage--draggable"
        onWheel={onWheel}
        onPointerDown={startDrag}
        onPointerMove={drag}
        onPointerUp={() => { dragRef.current = null; }}
        onPointerCancel={() => { dragRef.current = null; }}
        onDoubleClick={() => zoomTo(scale === 1 ? 2 : 1)}
      >
        {hasHighlights ? <div
          className="image-viewer-annotated-canvas"
          style={{ transform: `translate3d(${offset.x}px, ${offset.y}px, 0) scale(${scale})` }}
        >
          <img src={src} alt={alt} draggable={false} onError={markImageFailed} />
          {highlightLayer}
        </div> : <img
          src={src}
          alt={alt}
          draggable={false}
          onError={markImageFailed}
          style={{ transform: `translate3d(${offset.x}px, ${offset.y}px, 0) scale(${scale})` }}
        />}
      </div>
      <small>{highlightLabels.length > 0 ? `${highlightLabels.join(" · ")}｜` : ""}按住滑鼠左鍵即可拖動畫面；滾輪或按鈕可縮放。</small>
    </div>
  </div> : null;

  if (imageFailed) {
    return <div className={`zoomable-image-fallback ${className}`.trim()}>
      {fallback ?? <span>圖片無法載入</span>}
    </div>;
  }

  return <>
    <button
      type="button"
      className={`zoomable-image ${hasHighlights ? "zoomable-image--annotated" : ""} ${className}`.trim()}
      onClick={() => setOpen(true)}
      aria-label={`檢視並縮放：${title ?? alt}${highlightLabels.length > 0 ? `，標記：${highlightLabels.join("、")}` : ""}`}
    >
      <img
        className={imageClassName}
        src={src}
        alt={alt}
        onError={markImageFailed}
        onLoad={(event) => setLoadedSize({
          src,
          x: event.currentTarget.naturalWidth,
          y: event.currentTarget.naturalHeight,
        })}
      />
      {highlightLayer}
      {overlay ?? <span className="zoomable-image-hint">⌕ 點擊檢視</span>}
    </button>
    {viewer && createPortal(viewer, document.body)}
  </>;
}
