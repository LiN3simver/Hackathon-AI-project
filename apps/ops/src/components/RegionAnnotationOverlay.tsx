import type { LearningRegionAnnotation } from "../lib/api";

/** Draw real geometry in an image-sized container, never stretch a bounding box into a mask. */
export function RegionAnnotationOverlay({ annotations }: { annotations: LearningRegionAnnotation[] }) {
  return <svg className="region-geometry-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true">
    {annotations.map((region, index) => region.shape === "polygon" && region.points?.length
      ? <polygon key={index} className={`region-geometry region-geometry--${region.kind}`}
        points={region.points.map((point) => `${point.x},${point.y}`).join(" ")} />
      : <rect key={index} className={`region-geometry region-geometry--${region.kind}`}
        x={region.x} y={region.y} width={region.width} height={region.height} />)}
  </svg>;
}
