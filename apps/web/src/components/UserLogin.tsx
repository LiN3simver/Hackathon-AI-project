import { FormEvent, useState } from "react";

import { loginUser, type AuthSession } from "../lib/api";
import { MobileAccess } from "./MobileAccess";

export function UserLogin({ onLoggedIn }: { onLoggedIn: (session: AuthSession) => void }) {
  const [rentalId, setRentalId] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onLoggedIn(await loginUser({ rental_id: rentalId, password }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "登入失敗");
    } finally {
      setBusy(false);
    }
  }

  return <main className="login-page">
    <section className="login-card">
      <div className="login-brand"><span className="brand-mark">G</span><div><strong>Guardian</strong><small>租還車拍攝登入</small></div></div>
      <p className="eyebrow">RENTAL ACCESS</p>
      <div className="login-access-heading">
        <h1>確認目前租單</h1>
        <MobileAccess />
      </div>
      <p>請輸入營運人員提供的租單編號與密碼，登入後開始拍攝流程。</p>
      <form onSubmit={(event) => void submit(event)}>
        <label><span>租單編號</span><input required value={rentalId} onChange={(event) => setRentalId(event.target.value)} autoComplete="username" /></label>
        <label><span>密碼</span><input required type="password" inputMode="numeric" pattern="[0-9]{4}" minLength={4} maxLength={4} value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" /></label>
        {error && <div className="error-box"><strong>無法登入</strong><span>{error}</span></div>}
        <button className="login-submit" disabled={busy}>{busy ? "登入中…" : "登入"}</button>
      </form>
    </section>
  </main>;
}
