import { useEffect, useRef, useState } from "react";
import type { PointerEvent, ReactNode, WheelEvent } from "react";
import { createPortal } from "react-dom";
import { CleanlinessPhoto } from "./CleanlinessPhoto";
import type { CleanlinessEvidence } from "../lib/api";

type Point = { x: number; y: number };

export function ZoomableImage({
  src,
  alt,
  title,
  className = "",
  imageClassName = "",
  overlay,
  fallback,
  evidence,
}: {
  src: string;
  alt: string;
  title?: string;
  className?: string;
  imageClassName?: string;
  overlay?: ReactNode;
  fallback?: ReactNode;
  evidence?: CleanlinessEvidence[];
}) {
  const [open, setOpen] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
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

  useEffect(() => setImageFailed(false), [src]);

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
        <CleanlinessPhoto
          src={src}
          alt={alt}
          evidence={evidence}
          onError={markImageFailed}
          style={{ transform: `translate3d(${offset.x}px, ${offset.y}px, 0) scale(${scale})` }}
        />
      </div>
      <small>按住滑鼠左鍵即可拖動畫面；滾輪或按鈕可縮放，雙擊可快速切換。</small>
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
      className={`zoomable-image ${className}`.trim()}
      onClick={() => setOpen(true)}
      aria-label={`檢視並縮放：${title ?? alt}`}
    >
      <CleanlinessPhoto className={imageClassName} src={src} alt={alt} evidence={evidence} onError={markImageFailed} />
      {overlay ?? <span className="zoomable-image-hint">⌕ 點擊檢視</span>}
    </button>
    {viewer && createPortal(viewer, document.body)}
  </>;
}
