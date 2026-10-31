import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent } from "react";
import type { ImageHighlight } from "./ImageViewer";
import { RegionAnnotationOverlay } from "./RegionAnnotationOverlay";
import { polygonBounds } from "../lib/regionGeometry";

import type {
  LearningAnnotationImageSlot,
  LearningRegionAnnotation,
  LearningTask,
} from "../lib/api";

type Point = { x: number; y: number };
type View = { scale: number; x: number; y: number };

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const IDENTITY_VIEW: View = { scale: 1, x: 0, y: 0 };

// Keep the zoomed image covering the whole viewport so no empty edge shows.
function clampView(view: View, width: number, height: number): View {
  const scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, view.scale));
  return {
    scale,
    x: Math.min(0, Math.max(width - width * scale, view.x)),
    y: Math.min(0, Math.max(height - height * scale, view.y)),
  };
}

const hardNegativeLabels = new Set([
  "normal_body_line",
  "panel_gap",
  "reflection",
  "water_droplet",
  "surface_dirt",
  "shadow",
  "floor_pattern",
  "seat_pattern",
  "leather_texture",
  "normal_wear",
  "other_normal",
]);

const labelNames: Record<string, string> = {
  normal_vehicle_body: "正常車身輪廓",
  vehicle_body: "完整車身輪廓",
  body_panel: "車身板件輪廓",
  bumper: "保險桿輪廓",
  window: "車窗輪廓",
  wheel: "輪圈／輪胎輪廓",
  seat_cushion: "座椅坐墊輪廓",
  seat_backrest: "座椅椅背輪廓",
  headrest: "頭枕輪廓",
  floor_mat: "地墊輪廓",
  interior_floor: "車內地板輪廓",
  dashboard: "中控台／儀表板輪廓",
  // Earlier generic interior boxes; kept so saved truths still read correctly.
  dirt_floor: "地墊／腳踏區髒污",
  dirt_seat: "座椅髒污",
  stain: "潑灑／污漬",
  dust: "灰塵",
  footprint: "腳印／鞋印",
  mud_sand: "泥沙／砂石",
  leather_seat_stain: "皮椅污漬",
  fabric_seat_stain: "布椅污漬",
  food_crumbs: "食物殘渣／碎屑",
  liquid_spill: "飲料／液體潑灑",
  dashboard_dust: "中控台／儀表板灰塵",
  seatback_pocket: "椅背／置物袋髒污",
  trash: "垃圾／雜物",
  pet_hair: "寵物毛髮",
  other_dirt: "其他髒污",
  scratch: "刮痕",
  dent: "凹陷",
  crack: "裂痕",
  broken_part: "破裂／零件損壞",
  paint_transfer: "漆料轉移",
  deformation: "板件變形",
  panel_misalignment: "板件錯位",
  other_damage: "其他車損",
  normal_body_line: "正常車身折線",
  panel_gap: "原廠板件接縫",
  reflection: "反光",
  water_droplet: "雨滴／水痕",
  surface_dirt: "表面髒污（非車損）",
  shadow: "陰影",
  floor_pattern: "地墊原始紋路",
  seat_pattern: "座椅原始紋路",
  leather_texture: "皮椅皺褶／光澤",
  normal_wear: "正常使用磨損",
  other_normal: "其他正常區域",
};

export function regionAnnotationLabel(label: string): string {
  return labelNames[label] ?? label;
}

type LabelGroup = { title: string; labels: string[] };

const NORMAL_GROUP_TITLE = "正常區域（不應判為問題）";
const referenceLabels = new Set(["normal_vehicle_body", "vehicle_body", "body_panel", "bumper", "window", "wheel",
  "seat_cushion", "seat_backrest", "headrest", "floor_mat", "interior_floor", "dashboard"]);

function taskLabelGroups(task: LearningTask, inspectionSlot?: string | null): LabelGroup[] {
  if (task === "interior_cleanliness") {
    const rear = inspectionSlot === "interior_rear";
    return [
      { title: "部位輪廓（記錄範圍）", labels: ["seat_cushion", "seat_backrest", "headrest", "floor_mat", "interior_floor",
        ...(rear ? [] : ["dashboard"])] },
      { title: "腳踏區／地板", labels: ["dust", "footprint", "mud_sand"] },
      { title: "座椅", labels: ["leather_seat_stain", "fabric_seat_stain", "food_crumbs", "pet_hair"] },
      {
        title: "其他髒污",
        labels: [
          "liquid_spill",
          "trash",
          // Front rows have a dashboard; rear rows have seatback pockets.
          ...(rear ? ["seatback_pocket"] : inspectionSlot === "interior_front" ? ["dashboard_dust"] : ["dashboard_dust", "seatback_pocket"]),
          "other_dirt",
        ],
      },
      {
        title: NORMAL_GROUP_TITLE,
        labels: ["floor_pattern", "seat_pattern", "leather_texture", "normal_wear", "shadow", "reflection", "other_normal"],
      },
    ];
  }
  return [
    { title: "部位輪廓（記錄範圍）", labels: ["vehicle_body", "body_panel", "bumper", "window", "wheel"] },
    {
      title: "問題區域",
      labels: ["scratch", "dent", "crack", "broken_part", "paint_transfer", "deformation", "panel_misalignment", "other_damage"],
    },
    {
      title: NORMAL_GROUP_TITLE,
      labels: ["normal_vehicle_body", "normal_body_line", "panel_gap", "reflection", "water_droplet", "surface_dirt", "shadow", "other_normal"],
    },
  ];
}

function normalizedPoint(
  event: { clientX: number; clientY: number },
  element: HTMLDivElement,
): Point {
  const rect = element.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
  };
}

function outlinePoints(region: LearningRegionAnnotation): Point[] {
  return region.shape === "polygon" && region.points ? region.points : [
    { x: region.x, y: region.y }, { x: region.x + region.width, y: region.y },
    { x: region.x + region.width, y: region.y + region.height }, { x: region.x, y: region.y + region.height },
  ];
}

function editedPolygon(region: LearningRegionAnnotation, points: Point[]): LearningRegionAnnotation {
  return { ...region, ...polygonBounds(points), shape: "polygon", points };
}

export function RegionAnnotationEditor({
  src,
  task,
  inspectionSlot,
  imageSlot,
  title,
  annotations,
  modelHighlights = [],
  onChange,
  onDraftChange,
}: {
  src: string;
  task: LearningTask;
  inspectionSlot?: string | null;
  imageSlot: LearningAnnotationImageSlot;
  title: string;
  annotations: LearningRegionAnnotation[];
  modelHighlights?: ImageHighlight[];
  onChange: (annotations: LearningRegionAnnotation[]) => void;
  onDraftChange?: (active: boolean) => void;
}) {
  const groups = useMemo(() => taskLabelGroups(task, inspectionSlot), [task, inspectionSlot]);
  const [selectedLabel, setSelectedLabel] = useState("");
  const [shape, setShape] = useState<"polygon" | "rectangle">("polygon");
  const [vertices, setVertices] = useState<Point[]>([]);
  const [cursor, setCursor] = useState<Point | null>(null);
  const [geometryError, setGeometryError] = useState<string | null>(null);
  const [drag, setDrag] = useState<{ start: Point; current: Point } | null>(null);
  const [view, setView] = useState<View>(IDENTITY_VIEW);
  const [pan, setPan] = useState<{ pointerX: number; pointerY: number; origin: View } | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [insertVertex, setInsertVertex] = useState(false);
  const [undo, setUndo] = useState<LearningRegionAnnotation[]>([]);
  const [vertexDrag, setVertexDrag] = useState<{ index: number; vertex: number; original: LearningRegionAnnotation } | null>(null);
  const annotationsRef = useRef(annotations);
  annotationsRef.current = annotations;
  const vertexDragRef = useRef(vertexDrag);
  vertexDragRef.current = vertexDrag;
  const viewportRef = useRef<HTMLDivElement | null>(null);
  // The zoomed canvas; its transformed bounding rect maps pointers to image coordinates.
  const stageRef = useRef<HTMLDivElement | null>(null);
  const visible = annotations.filter((item) => item.image_slot === imageSlot);
  useEffect(() => onDraftChange?.(vertices.length > 0 || Boolean(drag) || Boolean(vertexDrag)),
    [vertices.length, drag, vertexDrag, onDraftChange]);
  useEffect(() => () => onDraftChange?.(false), [onDraftChange]);

  useEffect(() => {
    setView(IDENTITY_VIEW);
    setVertices([]);
    setCursor(null);
    setGeometryError(null);
    setDrag(null);
    setPan(null);
    setEditingIndex(null);
    setUndo([]);
    setInsertVertex(false);
  }, [src]);

  function emit(next: LearningRegionAnnotation[]) {
    annotationsRef.current = next;
    onChange(next);
  }

  function replace(index: number, next: LearningRegionAnnotation) {
    emit(annotationsRef.current.map((item, i) => i === index ? next : item));
  }

  function selectOutline(index: number) {
    setEditingIndex(index);
    setUndo([]);
    setInsertVertex(false);
    setGeometryError(null);
  }

  function appendAnnotation(annotation: LearningRegionAnnotation) {
    emit([...annotationsRef.current, annotation]);
  }

  function commitEdit(index: number, before: LearningRegionAnnotation, after: LearningRegionAnnotation) {
    replace(index, after);
    setUndo((current) => [...current.slice(-19), before]);
    setGeometryError(null);
  }

  function annotationKind() {
    return referenceLabels.has(selectedLabel) ? "reference" as const
      : hardNegativeLabels.has(selectedLabel) ? "hard_negative" as const : "issue" as const;
  }

  function completePolygon() {
    try {
      const bounds = polygonBounds(vertices);
      void appendAnnotation({ image_slot: imageSlot, kind: annotationKind(), label: selectedLabel,
        ...bounds, shape: "polygon", points: vertices, notes: "",
        ...(referenceLabels.has(selectedLabel) ? { normal_reference: selectedLabel === "normal_vehicle_body" } : {}) });
      setVertices([]);
      setCursor(null);
      setGeometryError(null);
    } catch (error) {
      setGeometryError(error instanceof Error ? error.message : "請檢查輪廓。");
    }
  }

  // React registers wheel listeners as passive, so preventDefault needs a native listener.
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    function zoom(event: WheelEvent) {
      if (!viewport) return;
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const pointerX = event.clientX - rect.left - viewport.clientLeft;
      const pointerY = event.clientY - rect.top - viewport.clientTop;
      const factor = Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0015));
      setView((current) => {
        const scale = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current.scale * factor));
        // Zoom around the cursor so the spot being framed stays under the mouse.
        const ratio = scale / current.scale;
        return clampView({
          scale,
          x: pointerX - (pointerX - current.x) * ratio,
          y: pointerY - (pointerY - current.y) * ratio,
        }, viewport.clientWidth, viewport.clientHeight);
      });
    }
    viewport.addEventListener("wheel", zoom, { passive: false });
    return () => viewport.removeEventListener("wheel", zoom);
  }, []);

  function begin(event: PointerEvent<HTMLDivElement>) {
    if (!stageRef.current) return;
    // Right/middle drag always pans; left drag pans only while no label is chosen.
    const panning = event.button === 1 || event.button === 2 || (event.button === 0 && !selectedLabel && editingIndex === null);
    if (panning) {
      if (view.scale <= MIN_ZOOM) return;
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      setPan({ pointerX: event.clientX, pointerY: event.clientY, origin: view });
      return;
    }
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = normalizedPoint(event, stageRef.current);
    if (editingIndex !== null) {
      if (!insertVertex) return;
      const original = annotationsRef.current[editingIndex];
      const points = outlinePoints(original);
      if (points.length >= 256) { setGeometryError("最多 256 個頂點。"); return; }
      const rect = stageRef.current.getBoundingClientRect();
      let closest = 0, shortest = Infinity;
      points.forEach((a, index) => {
        const b = points[(index + 1) % points.length];
        const dx = (b.x - a.x) * rect.width, dy = (b.y - a.y) * rect.height;
        const px = (point.x - a.x) * rect.width, py = (point.y - a.y) * rect.height;
        const t = Math.max(0, Math.min(1, (px * dx + py * dy) / (dx * dx + dy * dy || 1)));
        const distance = Math.hypot(px - t * dx, py - t * dy);
        if (distance < shortest) { shortest = distance; closest = index; }
      });
      const next = [...points.slice(0, closest + 1), point, ...points.slice(closest + 1)];
      try {
        const edited = editedPolygon(original, next);
        setInsertVertex(false);
        commitEdit(editingIndex, original, edited);
      } catch { setGeometryError("新增頂點會造成輪廓交叉，請換個位置。"); }
      return;
    }
    if (shape === "polygon") {
      const first = vertices[0];
      const rect = stageRef.current.getBoundingClientRect();
      if (vertices.length >= 3 && first
        && Math.hypot((point.x - first.x) * rect.width, (point.y - first.y) * rect.height) <= 10) {
        completePolygon();
      } else if (vertices.length < 256) {
        const vertex = { x: Number(point.x.toFixed(6)), y: Number(point.y.toFixed(6)) };
        if (!vertices.some((p) => p.x === vertex.x && p.y === vertex.y)) setVertices([...vertices, vertex]);
        setGeometryError(null);
      } else {
        setGeometryError("最多 256 個頂點，請完成或取消目前輪廓。");
      }
      return;
    }
    setDrag({ start: point, current: point });
  }

  function move(event: PointerEvent<HTMLDivElement>) {
    if (vertexDragRef.current && stageRef.current) {
      const active = vertexDragRef.current;
      const points = outlinePoints(active.original).map((p, i) => i === active.vertex ? normalizedPoint(event, stageRef.current!) : p);
      // Preview freely while dragging; validate the completed polygon on pointer release.
      const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
      replace(active.index, { ...active.original, shape: "polygon", points,
        x: Math.min(...xs), y: Math.min(...ys), width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) });
      return;
    }
    if (pan && viewportRef.current) {
      setView(clampView({
        scale: pan.origin.scale,
        x: pan.origin.x + event.clientX - pan.pointerX,
        y: pan.origin.y + event.clientY - pan.pointerY,
      }, viewportRef.current.clientWidth, viewportRef.current.clientHeight));
      return;
    }
    if (shape === "polygon" && stageRef.current) setCursor(normalizedPoint(event, stageRef.current));
    if (!drag || !stageRef.current) return;
    setDrag({ ...drag, current: normalizedPoint(event, stageRef.current) });
  }

  function finish(event: PointerEvent<HTMLDivElement>) {
    const active = vertexDragRef.current;
    if (active && stageRef.current) {
      const points = outlinePoints(active.original).map((p, i) => i === active.vertex ? normalizedPoint(event, stageRef.current!) : p);
      try {
        const edited = editedPolygon(active.original, points);
        commitEdit(active.index, active.original, edited);
      } catch {
        replace(active.index, active.original);
        setGeometryError("輪廓不能交叉或縮成直線，已復原這次拖曳。");
      }
      vertexDragRef.current = null;
      setVertexDrag(null);
      return;
    }
    if (pan) {
      setPan(null);
      return;
    }
    if (!drag || !stageRef.current || !selectedLabel) return;
    const end = normalizedPoint(event, stageRef.current);
    const x = Math.min(drag.start.x, end.x);
    const y = Math.min(drag.start.y, end.y);
    const width = Math.abs(drag.start.x - end.x);
    const height = Math.abs(drag.start.y - end.y);
    setDrag(null);
    if (width < 0.015 || height < 0.015) return;
    void appendAnnotation({
        image_slot: imageSlot,
        kind: annotationKind(),
        label: selectedLabel,
        x: Number(x.toFixed(6)),
        y: Number(y.toFixed(6)),
        width: Number(width.toFixed(6)),
        height: Number(height.toFixed(6)),
        notes: "",
        ...(referenceLabels.has(selectedLabel) ? { normal_reference: selectedLabel === "normal_vehicle_body" } : {}),
      });
  }

  function remove(target: LearningRegionAnnotation) {
    setEditingIndex(null);
    setUndo([]);
    let removed = false;
    onChange(annotations.filter((item) => {
      if (!removed && item === target) {
        removed = true;
        return false;
      }
      return true;
    }));
  }

  const draftStyle = drag ? {
    left: `${Math.min(drag.start.x, drag.current.x) * 100}%`,
    top: `${Math.min(drag.start.y, drag.current.y) * 100}%`,
    width: `${Math.abs(drag.start.x - drag.current.x) * 100}%`,
    height: `${Math.abs(drag.start.y - drag.current.y) * 100}%`,
  } : undefined;
  const editing = editingIndex === null ? null : annotations[editingIndex];

  return <div className="region-annotation-editor">
    <div className="region-annotation-heading">
      <strong>{title}</strong>
      <select aria-label="標記類型" value={selectedLabel} disabled={vertices.length > 0 || Boolean(drag) || Boolean(vertexDrag)} onChange={(event) => {
        setEditingIndex(null);
        setUndo([]);
        setSelectedLabel(event.target.value);
        if (event.target.value === "normal_vehicle_body") setShape("rectangle");
        else if (referenceLabels.has(event.target.value)) setShape("polygon");
        setGeometryError(null);
      }}>
        <option value="">先選擇要標記的類型</option>
        {groups.map((group) => <optgroup key={group.title} label={group.title}>
          {group.labels.map((label) => <option key={label} value={label}>{labelNames[label]}</option>)}
        </optgroup>)}
      </select>
    </div>
    <p>選擇類型後，沿外緣逐點畫多邊形，或選矩形按住拖曳。完成後按「微調」可拖曳頂點修改範圍。</p>
    <div className="region-annotation-tools">
      <label>標記方式 <select value={shape} disabled={vertices.length > 0 || Boolean(drag) || Boolean(editing)}
        onChange={(event) => setShape(event.target.value as "polygon" | "rectangle")}>
        <option value="polygon">多邊形輪廓（逐點）</option><option value="rectangle">矩形（拖曳）</option>
      </select></label>
      {shape === "polygon" && !editing && <>
        <span>已點 {vertices.length} 個頂點</span>
        <button type="button" disabled={vertices.length < 3} onClick={completePolygon}>完成輪廓</button>
        <button type="button" disabled={!vertices.length} onClick={() => { setVertices(vertices.slice(0, -1)); setGeometryError(null); }}>退回一點</button>
        <button type="button" disabled={!vertices.length} onClick={() => { setVertices([]); setCursor(null); setGeometryError(null); }}>取消輪廓</button>
      </>}
      {editing && <>
        <span>微調：{regionAnnotationLabel(editing.label)}</span>
        <button type="button" disabled={Boolean(vertexDrag)} onClick={() => setInsertVertex(!insertVertex)}>{insertVertex ? "取消新增頂點" : "新增頂點"}</button>
        <button type="button" disabled={Boolean(vertexDrag) || !undo.length} onClick={() => {
          const restored = undo[undo.length - 1];
          replace(editingIndex!, restored); setUndo(undo.slice(0, -1)); setGeometryError(null); setInsertVertex(false);
        }}>復原上一步</button>
        {editing.auto_refinement && <button type="button" disabled={Boolean(vertexDrag)} onClick={() => {
          setUndo((current) => [...current.slice(-19), editing]);
          replace(editingIndex!, { ...editing, ...editing.auto_refinement!.original, auto_refinement: null });
          setInsertVertex(false); setGeometryError(null);
        }}>還原原始框選</button>}
        <button type="button" disabled={Boolean(vertexDrag)} onClick={() => { setEditingIndex(null); setInsertVertex(false); }}>完成微調</button>
      </>}
    </div>
    <div className="region-annotation-zoombar">
      <span>滑鼠滾輪縮放；放大後按住右鍵拖曳可移動畫面{selectedLabel ? "" : "（未選類型時左鍵也可拖曳）"}</span>
      <button type="button" disabled={view.scale <= MIN_ZOOM} onClick={() => setView(IDENTITY_VIEW)}>
        {Math.round(view.scale * 100)}%・重設
      </button>
    </div>
    <div
      className={`region-annotation-stage${selectedLabel ? " region-annotation-stage--ready" : ""}${pan ? " region-annotation-stage--panning" : ""}`}
      ref={viewportRef}
      onPointerDown={begin}
      onPointerMove={move}
      onPointerUp={finish}
      onPointerCancel={() => {
        if (vertexDragRef.current) replace(vertexDragRef.current.index, vertexDragRef.current.original);
        vertexDragRef.current = null;
        setVertexDrag(null);
        setDrag(null);
        setPan(null);
      }}
      onContextMenu={(event) => event.preventDefault()}
    >
      <div
        className="region-annotation-canvas"
        ref={stageRef}
        style={{
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
          ["--region-zoom" as string]: view.scale,
        }}
      >
        <img src={src} alt={title} draggable={false} />
        {modelHighlights.map((highlight, index) => <span
          key={`model-${index}`}
          className="region-box region-box--model"
          style={{ left: `${highlight.x * 100}%`, top: `${highlight.y * 100}%`,
            width: `${highlight.width * 100}%`, height: `${highlight.height * 100}%` }}
        ><span className="region-box-tag">{highlight.label}</span></span>)}
        <RegionAnnotationOverlay annotations={visible} />
        {visible.map((annotation, index) => <span
          className={`region-box region-box--${annotation.kind} region-box--label-only`}
          key={`${annotation.label}-${annotation.x}-${annotation.y}-${index}`}
          style={{
            left: `${annotation.x * 100}%`,
            top: `${annotation.y * 100}%`,
            width: `${annotation.width * 100}%`,
            height: `${annotation.height * 100}%`,
          }}
        >
          <span className="region-box-tag">
            <span>{labelNames[annotation.label] ?? annotation.label}</span>
            <button type="button" disabled={vertices.length > 0 || Boolean(drag) || Boolean(vertexDrag)}
              aria-label={`微調：${labelNames[annotation.label] ?? annotation.label}`}
              onPointerDown={(event) => event.stopPropagation()} onClick={() => selectOutline(annotations.indexOf(annotation))}>微調</button>
            <button
              type="button"
              title={`移除：${labelNames[annotation.label] ?? annotation.label}`}
              aria-label={`移除：${labelNames[annotation.label] ?? annotation.label}`}
              disabled={Boolean(vertexDrag)}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => remove(annotation)}
            >×</button>
          </span>
        </span>)}
        {editing && outlinePoints(editing).map((point, index) => <button type="button" key={index}
          className="region-edit-handle"
          aria-label={`拖曳頂點 ${index + 1}`}
          style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
          onPointerDown={(event) => {
            if (event.button !== 0 || !viewportRef.current) return;
            event.preventDefault(); event.stopPropagation();
            viewportRef.current.setPointerCapture(event.pointerId);
            const active = { index: editingIndex!, vertex: index, original: editing };
            vertexDragRef.current = active; setVertexDrag(active); setInsertVertex(false);
          }} />)}
        {draftStyle && <span className="region-box region-box--draft" style={draftStyle} />}
        {vertices.length > 0 && <svg className="region-geometry-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
          <polyline className="region-geometry region-geometry--draft"
            points={[...vertices, ...(cursor ? [cursor] : [])].map((p) => `${p.x},${p.y}`).join(" ")} />
          {vertices.map((p, index) => <circle key={index} cx={p.x} cy={p.y} r={0.005 / view.scale}
            className={`region-vertex${index === 0 ? " region-vertex--first" : ""}`} />)}
        </svg>}
      </div>
    </div>
    {editing && <small>{insertVertex ? "點選想補回的位置，會在最近的輪廓邊插入頂點。" : "拖曳黃色頂點微調，需要更細的邊緣時可新增頂點。"}</small>}
    {geometryError && <p role="alert" className="region-annotation-error">{geometryError}</p>}
    {vertices.length > 0 && <small>目前輪廓尚未完成；請先完成或取消，再關閉標註視窗。</small>}
    {visible.length > 0 && <small>已標記 {visible.length} 區；按標籤上的 × 可移除。紅色為問題、藍色為正常細節、綠色為部位輪廓。</small>}
    {visible.filter((region) => region.kind === "reference").map((region, index) => <label className="region-reference-confirmation" key={index}>
      <input type="checkbox" disabled={Boolean(vertexDrag)} checked={region.normal_reference ?? false} onChange={(event) =>
        onChange(annotations.map((item) => item === region ? { ...item, normal_reference: event.target.checked } : item))} />
      {regionAnnotationLabel(region.label)}：我已確認此部位可作正常外觀參考（無車損／髒污／異物）
    </label>)}
    {visible.some((region) => region.kind === "reference") && <small>未勾選的部位輪廓只記錄範圍，不會當成乾淨或無車損真值。</small>}
    {modelHighlights.length > 0 && <small>黃色虛線為模型疑似區域，供人工覆核參考；請框選確認後的區域以建立人工真值。</small>}
  </div>;
}
