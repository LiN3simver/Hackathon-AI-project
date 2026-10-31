import { useRef, useState } from "react";
import { getMobileAccess } from "../lib/api";

export function MobileAccess() {
  const dialog = useRef<HTMLDialogElement>(null);
  const [data, setData] = useState<Awaited<ReturnType<typeof getMobileAccess>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function open() {
    if (!dialog.current?.open) dialog.current?.showModal();
    setLoading(true); setError(null); setData(null);
    try { setData(await getMobileAccess()); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "讀取 QR Code 失敗"); }
    finally { setLoading(false); }
  }

  return <>
    <button type="button" className="mobile-access-button" onClick={() => void open()}>Qrcode</button>
    <dialog ref={dialog} className="mobile-access-dialog" aria-labelledby="mobile-access-title"
      onClick={(event) => { if (event.target === dialog.current) dialog.current?.close(); }}>
      <div className="mobile-access-card">
        <button type="button" className="mobile-access-close" aria-label="關閉 QR Code" onClick={() => dialog.current?.close()}>×</button>
        <h2 id="mobile-access-title">用手機開啟拍攝前台</h2>
        {loading && <p role="status">正在讀取 QR Code…</p>}
        {error && <p role="alert">{error}</p>}
        {data?.qr_data_url && <img src={data.qr_data_url} alt="手機開啟前台網址的 QR Code" width="280" height="280" />}
        {data?.url && <a href={data.url} target="_blank" rel="noreferrer">{data.url}</a>}
        {data && <p>{data.message}</p>}
        <p>掃描後請在手機登入租單。若連線失敗，請確認 Wi-Fi、主機防火牆與網址。</p>
        <button type="button" onClick={() => void open()} disabled={loading}>重新讀取</button>
      </div>
    </dialog>
  </>;
}
