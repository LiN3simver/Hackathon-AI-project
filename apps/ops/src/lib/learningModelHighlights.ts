import type { LearningSample } from "./api";
import type { ImageHighlight } from "../components/ImageViewer";

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : {};
}

export function learningModelHighlights(
  sample: LearningSample,
  slot: "primary" | "secondary",
): ImageHighlight[] {
  const metadata = sample.metadata;
  let regions: unknown[] = [];
  if (sample.task === "damage_pair" && slot === "secondary") {
    const specialist = record(metadata.specialist);
    regions = [
      ...(Array.isArray(metadata.candidates) ? metadata.candidates : []),
      ...(Array.isArray(specialist.regions) ? specialist.regions : []),
    ];
  } else if (sample.task === "interior_cleanliness" && slot === "primary") {
    regions = (Array.isArray(metadata.evidence) ? metadata.evidence : [])
      .filter((value) => record(value).visible === true && Number(record(value).severity) > 0);
  }
  const highlights: ImageHighlight[] = [];
  for (const value of regions) {
    const region = record(value);
    const box = record(region.bounding_box);
    const coordinates = Array.isArray(region.bounding_box)
      ? region.bounding_box : [box.x1, box.y1, box.x2, box.y2];
    if (coordinates.length !== 4 || !coordinates.every((n) => typeof n === "number" && Number.isFinite(n))) continue;
    const [x1, y1, x2, y2] = coordinates as number[];
    if (x1 < 0 || y1 < 0 || x2 > 1 || y2 > 1 || x2 <= x1 || y2 <= y1) continue;
    const highlight: ImageHighlight = {
      x: x1, y: y1, width: x2 - x1, height: y2 - y1,
      kind: "model",
      label: sample.task === "interior_cleanliness" ? "模型疑似髒汙" : "模型疑似車損",
    };
    if (!highlights.some((item) => item.x === highlight.x && item.y === highlight.y
      && item.width === highlight.width && item.height === highlight.height)) highlights.push(highlight);
  }
  return highlights;
}
