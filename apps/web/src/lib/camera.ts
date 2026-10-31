// Image Capture focus controls are not included in every TypeScript DOM lib.
type FocusCapabilities = MediaTrackCapabilities & { focusMode?: string[] };
type FocusSettings = MediaTrackSettings & { focusMode?: string; pointsOfInterest?: { x: number; y: number }[] };
type FocusConstraints = MediaTrackConstraints & {
  focusMode?: { exact: string };
  pointsOfInterest?: { x: number; y: number }[];
};

export type CameraFocus = { automatic: boolean; point: boolean };
export const noCameraFocus: CameraFocus = { automatic: false, point: false };

export function cameraFocusModes(track: MediaStreamTrack): string[] {
  try { return (track.getCapabilities?.() as FocusCapabilities)?.focusMode ?? []; }
  catch { return []; }
}

async function applyCameraFocus(track: MediaStreamTrack, mode: string, point?: { x: number; y: number }) {
  const applied = () => {
    const settings = track.getSettings() as FocusSettings;
    return settings.focusMode === mode && (!point || settings.pointsOfInterest?.some(
      (item) => Math.abs(item.x - point.x) < 0.01 && Math.abs(item.y - point.y) < 0.01,
    ));
  };
  try {
    await track.applyConstraints({ focusMode: { exact: mode }, ...(point ? { pointsOfInterest: [point] } : {}) } as FocusConstraints);
    if (applied()) return;
  } catch { /* Older Chromium versions require Image Capture settings in advanced. */ }
  const settings = { focusMode: mode, ...(point ? { pointsOfInterest: [point] } : {}) };
  await track.applyConstraints({ advanced: [settings] } as unknown as MediaTrackConstraints);
  if (!applied()) throw new Error("此鏡頭未套用對焦設定。");
}

export async function configureCameraFocus(track: MediaStreamTrack): Promise<CameraFocus> {
  const modes = cameraFocusModes(track);
  const mode = modes.includes("continuous") ? "continuous" : modes.includes("single-shot") ? "single-shot" : null;
  if (!mode) return noCameraFocus;
  try {
    await applyCameraFocus(track, mode);
    const automatic = (track.getSettings() as FocusSettings).focusMode === mode;
    const supported = navigator.mediaDevices.getSupportedConstraints() as MediaTrackSupportedConstraints & { pointsOfInterest?: boolean };
    return { automatic, point: automatic && supported.pointsOfInterest === true && Array.isArray((track.getSettings() as FocusSettings).pointsOfInterest) };
  } catch { return noCameraFocus; }
}

export async function requestCameraFocus(track: MediaStreamTrack, point?: { x: number; y: number }) {
  const modes = cameraFocusModes(track);
  const mode = modes.includes("single-shot") ? "single-shot" : modes.includes("continuous") ? "continuous" : null;
  if (!mode || track.readyState !== "live") throw new Error("請改用手機原生相機對焦拍攝。");
  await applyCameraFocus(track, mode, point);
  // There is no cross-browser autofocus-complete event. Allow the lens to
  // settle, without claiming that applying a setting proves the image is sharp.
  await new Promise<void>((resolve) => window.setTimeout(resolve, 650));
  if (mode === "single-shot" && modes.includes("continuous") && track.readyState === "live") {
    await applyCameraFocus(track, "continuous", point);
  }
}

export async function videoFrame(video: HTMLVideoElement, maxEdge: number, quality: number) {
  const scale = Math.min(1, maxEdge / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const context = canvas.getContext("2d");
  if (!context || !canvas.width || !canvas.height) throw new Error("相機畫面尚未就緒。");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
  if (!blob) throw new Error("拍攝失敗，請再試一次。");
  return blob;
}

export async function cameraPhoto(video: HTMLVideoElement, track?: MediaStreamTrack) {
  const ImageCaptureClass = (window as unknown as { ImageCapture?: new (track: MediaStreamTrack) => { takePhoto(): Promise<Blob> } }).ImageCapture;
  if (track?.readyState === "live" && ImageCaptureClass) {
    let timeout: number | undefined;
    try {
      const blob = await Promise.race([
        new ImageCaptureClass(track).takePhoto(),
        new Promise<never>((_, reject) => { timeout = window.setTimeout(() => reject(new Error("拍攝逾時")), 6000); }),
      ]);
      if (blob.size && ["image/jpeg", "image/png", "image/webp"].includes(blob.type)) return blob;
    } catch { /* Unsupported still capture must not prevent taking a photo. */ }
    finally { window.clearTimeout(timeout); }
  }
  return videoFrame(video, Math.max(video.videoWidth, video.videoHeight), 0.96);
}
