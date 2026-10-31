import { useEffect, useRef, useState, type PointerEvent } from "react";
import { detectCapturePreview, type CapturePreviewResponse, type ExpectedView, type QualityResponse } from "../lib/api";
import { cameraPhoto, configureCameraFocus, noCameraFocus, requestCameraFocus, videoFrame, type CameraFocus } from "../lib/camera";
import { CaptureGuide, captureViewLabels, isExteriorView } from "./CaptureGuide";
import { CleanlinessPhoto } from "./CleanlinessPhoto";
import { QualityResult, qualityAdvice } from "./QualityResult";

function DetectionBox({ box, label }: { box: number[] | null | undefined; label: string }) {
  if (!box || box.length !== 4 || !box.every(Number.isFinite)) return null;
  const [x1, y1, x2, y2] = box.map((value) => Math.max(0, Math.min(1, value)));
  if (x2 <= x1 || y2 <= y1) return null;
  return <div className="camera-detection" style={{
    left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%`,
  }}><span>{label}</span></div>;
}

export function CameraCapture({ expectedView, captureDisabled = false, onCapture, onClose, onChooseFile }: {
  expectedView: ExpectedView;
  captureDisabled?: boolean;
  onCapture: (file: File) => Promise<QualityResponse | null | undefined>;
  onClose: () => void;
  onChooseFile: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nativeInputRef = useRef<HTMLInputElement>(null);
  const focusPending = useRef(false);
  const focusTarget = useRef<{ x: number; y: number } | undefined>(undefined);
  const sceneVersion = useRef(0);
  const pendingPreview = useRef(false);
  const shutterPending = useRef(false);
  const [ready, setReady] = useState(false);
  const [aspectRatio, setAspectRatio] = useState(4 / 3);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<CapturePreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [lastPhoto, setLastPhoto] = useState<string | null>(null);
  const [lastPhotoName, setLastPhotoName] = useState("capture.jpg");
  const [lastQuality, setLastQuality] = useState<QualityResponse | null>(null);
  const [updatedAt, setUpdatedAt] = useState(0);
  const [now, setNow] = useState(Date.now());
  const [usingNative, setUsingNative] = useState(false);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const [activeDeviceId, setActiveDeviceId] = useState("");
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [focus, setFocus] = useState<CameraFocus>(noCameraFocus);
  const [focusing, setFocusing] = useState(false);
  const [focusPoint, setFocusPoint] = useState<{ x: number; y: number } | null>(null);
  const [focusMessage, setFocusMessage] = useState<string | null>(null);
  const exterior = isExteriorView(expectedView);

  useEffect(() => {
    const input = nativeInputRef.current;
    const cancel = () => setUsingNative(false);
    input?.addEventListener("cancel", cancel);
    return () => input?.removeEventListener("cancel", cancel);
  }, []);

  useEffect(() => () => { if (lastPhoto) URL.revokeObjectURL(lastPhoto); }, [lastPhoto]);
  useEffect(() => {
    setLastPhoto(null);
    setLastQuality(null);
    setError(null);
    setNotice(null);
  }, [expectedView]);

  useEffect(() => {
    setReady(false);
    setFocus(noCameraFocus);
    setFocusPoint(null);
    focusTarget.current = undefined;
    setFocusMessage(null);
    if (usingNative) return;
    let disposed = false;
    let stream: MediaStream | undefined;
    async function start() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("無法開啟相機，請使用 HTTPS 或改選照片。");
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: "environment" } }),
            width: { ideal: 2560 }, height: { ideal: 1440 }, frameRate: { ideal: 24, max: 30 },
          }, audio: false,
        });
        if (disposed) { stream.getTracks().forEach((track) => track.stop()); return; }
        streamRef.current = stream;
        const track = stream.getVideoTracks()[0];
        const cameraFocus = track ? await configureCameraFocus(track) : noCameraFocus;
        if (disposed) return;
        setFocus(cameraFocus);
        setActiveDeviceId(track?.getSettings().deviceId ?? "");
        setError(null);
        if (videoRef.current) videoRef.current.srcObject = stream;
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          if (!disposed) setCameras(devices.filter((device) => device.kind === "videoinput"));
        } catch { /* Device lists are optional; the current camera remains usable. */ }
      } catch (caught) {
        if (!disposed) setError(caught instanceof Error && caught.message.includes("HTTPS")
          ? caught.message : "無法使用相機，請允許相機權限或改選照片。");
      }
    }
    void start();
    return () => {
      disposed = true;
      stream?.getTracks().forEach((track) => track.stop());
      if (streamRef.current === stream) streamRef.current = null;
    };
  }, [usingNative, deviceId]);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    setPreview(null);
    setPreviewError(null);
    setUpdatedAt(0);
    if (!ready || !exterior || busy) return;
    let disposed = false;
    let timer: number | undefined;
    let controller: AbortController | undefined;
    async function scan() {
      if (disposed) return;
      if (document.hidden || pendingPreview.current || shutterPending.current || focusPending.current) {
        timer = window.setTimeout(() => void scan(), 1200);
        return;
      }
      const video = videoRef.current;
      if (!video || video.readyState < 2) {
        timer = window.setTimeout(() => void scan(), 1200);
        return;
      }
      pendingPreview.current = true;
      controller = new AbortController();
      const timeout = window.setTimeout(() => controller?.abort(), 15000);
      const frameVersion = sceneVersion.current;
      try {
        // The live preview only locates the vehicle; the plate is read once
        // from the captured photo, so a small frame is enough.
        const frame = await videoFrame(video, 1280, 0.82);
        if (disposed) return;
        const detection = await detectCapturePreview(frame, controller.signal);
        if (!disposed && frameVersion === sceneVersion.current) {
          setPreview(detection);
          // Start the display window when the result arrives; OCR itself can
          // take longer than five seconds, especially on the first frame.
          setUpdatedAt(Date.now());
          setPreviewError(null);
        }
      } catch {
        if (!disposed) {
          setPreview(null);
          setPreviewError("即時偵測暫時無法連線，請依框線對齊拍攝。");
        }
      } finally {
        window.clearTimeout(timeout);
        pendingPreview.current = false;
        if (!disposed) timer = window.setTimeout(() => void scan(), 1200);
      }
    }
    void scan();
    return () => { disposed = true; window.clearTimeout(timer); controller?.abort(); };
  }, [expectedView, ready, exterior, busy]);

  async function focusCamera(point?: { x: number; y: number }) {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track || !ready || busy || focusPending.current || !focus.automatic) return;
    focusPending.current = true;
    if (point) focusTarget.current = point;
    sceneVersion.current += 1;
    setPreview(null);
    setUpdatedAt(0);
    setFocusing(true);
    setFocusPoint(point ?? null);
    setFocusMessage("正在調整對焦，請保持手機穩定…");
    try {
      await requestCameraFocus(track, point);
      setFocusMessage("已送出對焦設定，請確認車牌清楚後拍攝。");
    } catch {
      setFocusMessage("此鏡頭無法調整對焦，請改用手機原生相機拍攝。");
      setFocus(noCameraFocus);
    } finally { focusPending.current = false; setFocusing(false); setFocusPoint(null); }
  }

  function tapFocus(event: PointerEvent<HTMLDivElement>) {
    if (!focus.point || event.target instanceof HTMLButtonElement) return;
    const rect = event.currentTarget.getBoundingClientRect();
    void focusCamera({ x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)) });
  }

  function openNativeCamera() {
    if (busy || focusPending.current || captureDisabled) return;
    // Release the browser's camera before handing the hardware to the OS app.
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setReady(false);
    setUsingNative(true);
    if (nativeInputRef.current) { nativeInputRef.current.value = ""; nativeInputRef.current.click(); }
  }

  async function capture(nativePhoto?: File) {
    const video = videoRef.current;
    if ((!nativePhoto && (!video || !ready)) || shutterPending.current || focusPending.current || captureDisabled) return;
    shutterPending.current = true;
    setBusy(true);
    setError(null);
    setNotice(null);
    setLastQuality(null);
    try {
      const track = streamRef.current?.getVideoTracks()[0];
      if (!nativePhoto && track && focus.automatic) {
        try { await requestCameraFocus(track, focus.point ? focusTarget.current : undefined); }
        catch { setFocusMessage("無法調整對焦；若車牌不清楚，請改用手機原生相機。"); }
      }
      const blob = nativePhoto ?? await cameraPhoto(video!, track);
      setLastPhoto(URL.createObjectURL(blob));
      const extension = blob.type === "image/png" ? "png" : blob.type === "image/webp" ? "webp" : "jpg";
      const photo = nativePhoto ?? new File([blob], `camera-${Date.now()}.${extension}`, { type: blob.type });
      setLastPhotoName(photo.name);
      const quality = await onCapture(photo);
      setLastQuality(quality ?? null);
      if (!quality) setError("不通過：照片分析失敗，請重拍。");
      else if (quality.status === "fail") setError(`不通過：${qualityAdvice(quality)}`);
      else setNotice(expectedView === "damage_detail" ? "通過，可繼續補拍或關閉相機。" : "通過，請拍下一個位置。");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "拍攝失敗，請再試一次。");
    } finally { shutterPending.current = false; setBusy(false); if (nativePhoto) setUsingNative(false); }
  }

  const fresh = ready && !busy && !focusing && now - updatedAt < 5000;
  const vehicle = fresh ? preview?.coverage_prediction : null;
  const framingReady = vehicle?.status === "pass";
  const livePaused = !ready || usingNative || busy;
  const guidance = !exterior ? (expectedView === "damage_detail" ? "自由拍攝損傷特寫，保留周邊板件以確認位置。"
    : expectedView === "interior_front" ? "從前座車門外斜拍，讓座椅、方向盤、中控與腳踏區入鏡。"
    : "從後座車門外斜拍，讓後排座椅、前座椅背與腳踏區入鏡。")
    : livePaused ? "即時車身偵測已暫停；拍攝照片後會檢查車身與車牌。"
    : !vehicle?.car_detected ? "尚未辨識到主要車輛，請對照範本保持車身完整入鏡。"
    : vehicle.status === "fail" || vehicle.status === "warning" ? (vehicle.area_ratio < 0.3 ? "車身太小，請靠近一些。" : "車身太靠近邊緣，請稍微退後。")
    : "車身完整；請讓車牌清楚入鏡，確認與範本角度一致後即可拍攝。";

  return <section className="camera-capture" aria-label="即時拍攝" onClick={(event) => event.stopPropagation()}>
    <div className="camera-capture__heading"><div><span className="camera-live-dot" />{exterior ? "即時尋找車身" : expectedView === "damage_detail" ? "自由拍攝特寫" : "拍攝對齊引導"}</div><strong>{captureViewLabels[expectedView]}</strong></div>
    {cameras.length > 1 && <label className="camera-capture__lens">拍攝鏡頭
      <select value={activeDeviceId} disabled={busy || focusing || usingNative} onChange={(event) => setDeviceId(event.target.value)}>
        {cameras.map((camera, index) => <option key={camera.deviceId} value={camera.deviceId}>{camera.label || `鏡頭 ${index + 1}`}</option>)}
      </select>
    </label>}
    <div className="camera-capture__view" onPointerDown={tapFocus} style={{ aspectRatio, width: `min(100%, ${360 * aspectRatio}px)` }}>
      <video ref={videoRef} autoPlay playsInline muted onLoadedData={() => setReady(true)} onResize={() => {
        const video = videoRef.current;
        if (video?.videoWidth && video.videoHeight) setAspectRatio(video.videoWidth / video.videoHeight);
      }} />
      <CaptureGuide view={expectedView} />
      {vehicle?.car_detected && <DetectionBox box={vehicle.bounding_box} label="已找到車身" />}
      <div className="camera-capture__angle">{captureViewLabels[expectedView]}{expectedView !== "damage_detail" && " · 對齊半透明輪廓"}</div>
      {exterior && framingReady && <div className="camera-capture__recognition" role="status" aria-live="polite">
        <strong>✓ 車身完整入鏡</strong>
        <span>確認範本角度後可拍攝；車牌會在拍照後辨識</span>
      </div>}
      {focusPoint && <div className="camera-capture__focus-ring" style={{ left: `${focusPoint.x * 100}%`, top: `${focusPoint.y * 100}%` }} />}
      {usingNative && !busy && <div className="camera-capture__notice"><div>已切換手機相機<button type="button" onClick={() => setUsingNative(false)}>返回即時相機</button></div></div>}
      {busy && <div className="camera-capture__notice" role="status">正在拍攝並檢查照片品質…</div>}
    </div>
    <div className="camera-capture__feedback" aria-live="polite">
      <p>{guidance}</p>
      {exterior && <div className="camera-capture__chips"><span className={vehicle?.car_detected ? "is-found" : ""}>{vehicle?.car_detected ? "✓ 車身已找到" : livePaused ? "即時車身辨識暫停" : "◌ 尋找車身"}</span><span>車牌拍照後辨識</span></div>}
      {(previewError || (fresh && preview?.messages.length)) ? <small>{previewError ?? preview?.messages.join(" ")}</small> : <small>{exterior ? "拍攝後會辨識車牌，並確認照片品質與角度。" : expectedView === "damage_detail" ? "可補拍全景、近照與斜角，最多 4 張。" : "比照後台範本，從車門外對齊輪廓後拍攝。"}</small>}
      {notice && <p className="camera-capture__success" role="status">{notice}</p>}
      {error && <p className="camera-capture__error" role="alert">{error}</p>}
    </div>
    <div className="camera-capture__focus-tools">
      <p role="status">{focusMessage ?? (focus.automatic ? focus.point ? "已啟用自動對焦，可點車牌位置調整焦點。" : "已啟用自動對焦，模糊時可重新對焦。" : "若車牌模糊，請使用手機原生相機對焦拍攝。")}</p>
      <div>{focus.automatic && <button type="button" disabled={!ready || busy || focusing} onClick={() => void focusCamera()}>{focusing ? "對焦中…" : "重新對焦"}</button>}
        <button type="button" disabled={busy || focusing || captureDisabled || usingNative} onClick={openNativeCamera}>使用手機原生相機</button></div>
      <input ref={nativeInputRef} className="camera-capture__native-input" type="file" accept="image/jpeg,image/png,image/webp" capture="environment" hidden
        onChange={(event) => {
          const photo = event.target.files?.[0];
          if (photo) void capture(photo); else setUsingNative(false);
        }} />
    </div>
    {lastPhoto && <div className="camera-capture__last-photo">
      <CleanlinessPhoto src={lastPhoto} alt={`${captureViewLabels[expectedView]}剛拍攝的照片`} plate={lastQuality?.plate_prediction} />
      {lastQuality && <QualityResult result={lastQuality} />}
    </div>}
    {lastQuality && exterior && <div className="camera-capture__photo-feedback" aria-live="polite">
      {lastPhoto && <a href={lastPhoto} download={lastPhotoName}>下載這張照片</a>}
    </div>}
    <div className="camera-capture__actions">
      <button type="button" disabled={busy || focusing} onClick={onClose}>關閉相機</button>
      <button className="camera-capture__shutter" type="button" disabled={!ready || usingNative || busy || focusing || captureDisabled}
        onClick={() => void capture()}><span aria-hidden="true">◎</span>{busy ? "檢查中…" : captureDisabled ? "已達照片上限" : "拍攝照片"}</button>
      <button type="button" disabled={busy || focusing || captureDisabled} onClick={onChooseFile}>選擇照片</button>
    </div>
  </section>;
}
