import { FormEvent, useState } from "react";

import { loginOps, type AuthSession } from "../lib/api";

export function OpsLogin({ onLoggedIn }: { onLoggedIn: (session: AuthSession) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onLoggedIn(await loginOps(username, password));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "登入失敗");
    } finally {
      setBusy(false);
    }
  }

  return <main className="ops-login-page">
    <section className="ops-login-card">
      <div className="login-brand"><span className="brand-mark">G</span><div><strong>Guardian OPS</strong><small>營運中心登入</small></div></div>
      <p className="eyebrow">ROLE-BASED ACCESS</p><h1>營運人員登入</h1>
      <p>請使用管理員建立的帳號登入；可用功能依帳號權限提供。</p>
      <form onSubmit={(event) => void submit(event)}>
        <label><span>帳號</span><input required value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" /></label>
        <label><span>密碼</span><input required type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" /></label>
        {error && <div className="error-box"><strong>無法登入</strong><span>{error}</span></div>}
        <button className="login-submit" disabled={busy}>{busy ? "登入中…" : "登入"}</button>
      </form>
    </section>
  </main>;
}
