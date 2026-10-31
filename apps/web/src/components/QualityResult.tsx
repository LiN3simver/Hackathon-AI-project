import type { QualityResponse } from "../lib/api";

/** The single concrete reason a photo did not pass; empty when it passed. */
export function qualityAdvice(result: QualityResponse): string {
  if (result.quality_skipped || result.status !== "fail") return "";
  if (result.recognition_abandoned) return result.summary;
  // Re-uploading the pickup photo, then an incorrect angle, take priority over image quality reasons.
  const issue = result.checks.find((check) => check.key === "same_as_pickup" && check.status === "fail")
    ?? result.checks.find((check) => check.key === "vehicle_view" && check.status === "fail")
    ?? result.checks.find((check) => check.status === "fail");
  return issue?.message ?? "照片未通過檢查，請重拍。";
}

/** Every result is shown only as 照片角度通過 or 不通過 with one concrete reason. */
export function QualityResult({ result }: { result: QualityResponse }) {
  if (result.quality_skipped) return null;
  const passed = result.status !== "fail";
  return <div className={`quality-overlay quality-overlay--${passed ? "pass" : "fail"}`} role="status" aria-live="polite">
    <div className="quality-overlay__score quality-overlay__score--verdict" aria-label={passed ? "照片角度通過" : "不通過"}>
      <strong>{passed ? "照片角度通過" : "不通過"}</strong>
    </div>
    {!passed && <p>{qualityAdvice(result)}</p>}
  </div>;
}
