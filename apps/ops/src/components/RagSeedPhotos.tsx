import { ragSeedImageUrl, type RentalEvent } from "../lib/api";

export function RagSeedPhotos({ event }: { event: RentalEvent }) {
  if (event.metadata.purpose !== "rag" || !event.metadata.evidence_photos) return null;
  const slots = Object.keys(event.metadata.evidence_photos as Record<string, string>);
  const label = (slot: string) => slot.startsWith("history") ? `歷程照片${slot.includes("-") ? ` ${slot.split("-")[1]}` : ""}`
    : slots.length === 1 ? "事件照片" : `${slot.startsWith("pickup") ? "借車" : "還車"}照片${slot.includes("-") ? ` ${slot.split("-")[1]}` : ""}`;
  return <div className="rag-seed-photos">{slots.map((slot) =>
    <a key={slot} href={ragSeedImageUrl(event.id, slot)} target="_blank" rel="noreferrer">
      <img src={ragSeedImageUrl(event.id, slot)} alt={`${label(slot)}（${event.metadata.synthetic ? "模擬資料" : "歷程"}）`} loading="lazy" />
      <span>{label(slot)}</span>
    </a>)}</div>;
}
