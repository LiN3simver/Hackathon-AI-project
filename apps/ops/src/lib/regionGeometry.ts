import type { LearningAnnotationPoint, LearningRegionAnnotation } from "./api";

function cross(a: LearningAnnotationPoint, b: LearningAnnotationPoint, c: LearningAnnotationPoint) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

function onSegment(a: LearningAnnotationPoint, b: LearningAnnotationPoint, p: LearningAnnotationPoint) {
  return Math.abs(cross(a, b, p)) <= 1e-12
    && p.x >= Math.min(a.x, b.x) - 1e-12 && p.x <= Math.max(a.x, b.x) + 1e-12
    && p.y >= Math.min(a.y, b.y) - 1e-12 && p.y <= Math.max(a.y, b.y) + 1e-12;
}

export function polygonBounds(points: LearningAnnotationPoint[]) {
  if (points.length < 3 || points.length > 256
    || new Set(points.map((p) => `${p.x},${p.y}`)).size !== points.length) {
    throw new Error("輪廓需要 3–256 個不同頂點。");
  }
  if (points.some((p) => !Number.isFinite(p.x) || !Number.isFinite(p.y) || p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1)) {
    throw new Error("輪廓頂點必須位於照片內。");
  }
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length], next = points[(i + 2) % points.length];
    area += a.x * b.y - b.x * a.y;
    if (Math.abs(cross(a, b, next)) <= 1e-12
      && (b.x - a.x) * (next.x - b.x) + (b.y - a.y) * (next.y - b.y) < 0) {
      throw new Error("輪廓邊線不能重疊或折返。");
    }
    for (let j = i + 1; j < points.length; j++) {
      if (j === i + 1 || (i === 0 && j === points.length - 1)) continue;
      const c = points[j], d = points[(j + 1) % points.length];
      if ((cross(a, b, c) * cross(a, b, d) < 0 && cross(c, d, a) * cross(c, d, b) < 0)
        || onSegment(a, b, c) || onSegment(a, b, d) || onSegment(c, d, a) || onSegment(c, d, b)) {
        throw new Error("輪廓邊線不能交叉，請沿外緣依序標記。");
      }
    }
  }
  if (Math.abs(area) <= 1e-10) throw new Error("輪廓必須形成有面積的區域。");
  const x = Math.min(...points.map((p) => p.x)), y = Math.min(...points.map((p) => p.y));
  return { x, y, width: Math.max(...points.map((p) => p.x)) - x, height: Math.max(...points.map((p) => p.y)) - y };
}

/** Reference outlines describe parts; only issue/normal evidence determines the truth. */
export function evidenceRegions(annotations: LearningRegionAnnotation[]) {
  return annotations.filter((region) => region.kind !== "reference");
}
