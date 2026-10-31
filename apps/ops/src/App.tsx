import { useEffect, useRef, useState } from "react";

import { AdminSettings } from "./components/AdminSettings";
import { OpsLogin } from "./components/OpsLogin";
import { OpsDashboard } from "./components/OpsDashboard";
import { getStoredOpsSession, logoutOps, type AuthSession } from "./lib/api";

type Theme = "light" | "dark";

function initialTheme(): Theme {
  const saved = window.localStorage.getItem("guardian-theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [authSession, setAuthSession] = useState<AuthSession | null>(getStoredOpsSession);
  const [page, setPage] = useState<"operations" | "settings">(() => authSession?.principal.role === "admin" ? "settings" : "operations");
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [loginNotice, setLoginNotice] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const sidebarCloseTimer = useRef<number | null>(null);

  function cancelSidebarClose() {
    if (sidebarCloseTimer.current !== null) {
      window.clearTimeout(sidebarCloseTimer.current);
      sidebarCloseTimer.current = null;
    }
  }

  function closeSidebar() {
    cancelSidebarClose();
    setSidebarCollapsed(true);
  }

  function scheduleSidebarClose() {
    cancelSidebarClose();
    sidebarCloseTimer.current = window.setTimeout(() => {
      setSidebarCollapsed(true);
      sidebarCloseTimer.current = null;
    }, 1000);
  }

  useEffect(() => () => {
    if (sidebarCloseTimer.current !== null) window.clearTimeout(sidebarCloseTimer.current);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("guardian-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!loginNotice) return;
    const timer = window.setTimeout(() => setLoginNotice(null), 1500);
    return () => window.clearTimeout(timer);
  }, [loginNotice]);

  if (!authSession) return <OpsLogin onLoggedIn={(session) => {
    const isAdmin = session.principal.role === "admin";
    window.history.replaceState(null, "", isAdmin ? "#model-management" : "#ops-overview");
    setPage(isAdmin ? "settings" : "operations");
    setAuthSession(session);
    setLoginNotice("已登入營運中心。");
  }} />;

  return <div className="app-shell ops-app-shell">
    {loginNotice && <div className="login-success-toast" role="status" aria-live="polite">
      <span aria-hidden="true">✓</span>
      <div><strong>登入成功</strong><p>{loginNotice}</p></div>
      <button type="button" aria-label="關閉登入成功通知" onClick={() => setLoginNotice(null)}>×</button>
    </div>}
    <header className={`topbar ${authSession.principal.role === "admin" ? "topbar--admin" : "topbar--staff"}`}>
      <div className="topbar-brand-group">
      <button
        type="button"
        className="ops-sidebar-toggle"
        aria-label={sidebarCollapsed ? "展開快速導覽" : "收合快速導覽"}
        aria-expanded={!sidebarCollapsed}
        aria-controls="ops-sidebar-navigation"
        onPointerEnter={(event) => {
          if (event.pointerType === "mouse") {
            cancelSidebarClose();
            setSidebarCollapsed(false);
          }
        }}
        onPointerLeave={(event) => { if (event.pointerType === "mouse") scheduleSidebarClose(); }}
        onClick={() => {
          cancelSidebarClose();
          setSidebarCollapsed((current) => window.matchMedia("(hover: none)").matches ? !current : false);
        }}
      >
        <span className="ops-sidebar-hamburger" aria-hidden="true"><i /><i /><i /></span>
      </button>
      <a className="brand" href="#top" aria-label="Guardian 營運首頁">
        <span className="brand-mark">G</span>
        <span><strong>Guardian OPS</strong><small>車況營運與人工覆核</small></span>
      </a>
      </div>
      {authSession.principal.role === "admin" && <nav className="topnav" aria-label="營運功能">
        <button className={page === "operations" ? "active" : ""} onClick={() => { setPage("operations"); closeSidebar(); }}>營運中心</button>
        <button className={page === "settings" ? "active" : ""} onClick={() => { setPage("settings"); closeSidebar(); }}>管理員設定</button>
      </nav>}
      <div className="ops-account">
        <button
          type="button"
          className="theme-toggle"
          aria-label={`切換至${theme === "dark" ? "淺色" : "深色"}主題`}
          title={`切換至${theme === "dark" ? "淺色" : "深色"}主題`}
          aria-pressed={theme === "dark"}
          onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}
        >
          <span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span>
        </button>
        <span><strong>{authSession.principal.display_name}</strong><small>{authSession.principal.role === "admin" ? "可調整系統設定" : "營運作業權限"}</small></span>
        <button type="button" onClick={() => { void logoutOps(); setPage("operations"); setAuthSession(null); }}>登出</button>
      </div>
    </header>

    {page === "operations" || authSession.principal.role !== "admin"
      ? <OpsDashboard
        sidebarCollapsed={sidebarCollapsed}
        onCloseSidebar={closeSidebar}
        onEnterSidebar={cancelSidebarClose}
        onLeaveSidebar={scheduleSidebarClose}
        canDeleteInspectionPhotos={authSession.principal.role === "admin"}
        canManageNotifications={authSession.principal.role === "admin"}
        canDeleteWorkOrders={authSession.principal.role === "admin"}
        showMyActivity={authSession.principal.role !== "admin"}
      />
      : <AdminSettings currentPrincipal={authSession.principal} sidebarCollapsed={sidebarCollapsed} onCloseSidebar={closeSidebar} onEnterSidebar={cancelSidebarClose} onLeaveSidebar={scheduleSidebarClose} />}

    <footer><span>Guardian OPS · Hackathon Project</span><span>車況管理介面 · 人工覆核優先</span></footer>
  </div>;
}
