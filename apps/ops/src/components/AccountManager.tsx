import { FormEvent, useEffect, useRef, useState } from "react";

import {
  createOpsAccount,
  createRentalAccount,
  deleteManagedAccount,
  getManagedAccounts,
  updateOpsAccount,
  updateRentalAccount,
  type AccountManagementResponse,
  type AuthPrincipal,
  type ManagedAccount,
} from "../lib/api";

type Props = {
  currentPrincipal: AuthPrincipal;
  onNotice: (message: string) => void;
  onError: (message: string | null) => void;
};

const emptyRental = {
  rental_id: "",
  customer_id: "",
  display_name: "",
  rental_date: "",
  return_date: "",
  vehicle_label: "",
  license_plate: "",
  password: "",
};
const emptyOps = {
  username: "",
  display_name: "",
  kind: "staff" as "staff" | "admin",
  password: "",
};
const RENTAL_PREVIEW_COUNT = 4;

function formatTime(value: string) {
  return new Intl.DateTimeFormat("zh-TW", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Taipei",
  }).format(new Date(value));
}

function kindLabel(kind: ManagedAccount["kind"]) {
  if (kind === "rental") return "租單";
  if (kind === "admin") return "管理者";
  return "營運人員";
}

export function AccountManager({ currentPrincipal, onNotice, onError }: Props) {
  const [management, setManagement] = useState<AccountManagementResponse | null>(null);
  const [rentalsExpanded, setRentalsExpanded] = useState(false);
  const [rentalForm, setRentalForm] = useState(emptyRental);
  const [opsForm, setOpsForm] = useState(emptyOps);
  const [busy, setBusy] = useState<string | null>(null);
  const [editTarget, setEditTarget] = useState<ManagedAccount | null>(null);
  const [editCustomerId, setEditCustomerId] = useState("");
  const [editRentalDate, setEditRentalDate] = useState("");
  const [editReturnDate, setEditReturnDate] = useState("");
  const [editVehicleLabel, setEditVehicleLabel] = useState("");
  const [editLicensePlate, setEditLicensePlate] = useState("");
  const [editDisplayName, setEditDisplayName] = useState("");
  const [editKind, setEditKind] = useState<"staff" | "admin">("staff");
  const [editPassword, setEditPassword] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ManagedAccount | null>(null);
  const [deleteStep, setDeleteStep] = useState<1 | 2>(1);
  const [deleteIdentifier, setDeleteIdentifier] = useState("");
  const [adminPassword, setAdminPassword] = useState("");
  const [dialogError, setDialogError] = useState<string | null>(null);
  const initialLoadStarted = useRef(false);

  async function refresh(showError = true) {
    try {
      setManagement(await getManagedAccounts());
    } catch (caught) {
      if (showError) {
        onError(`帳號資料載入失敗：${caught instanceof Error ? caught.message : "未知錯誤"}`);
      }
    }
  }

  useEffect(() => {
    if (initialLoadStarted.current) return;
    initialLoadStarted.current = true;
    void refresh();
    const timer = window.setInterval(() => void refresh(false), 30_000);
    return () => window.clearInterval(timer);
  }, []);

  function updateLocal(account: ManagedAccount) {
    setManagement((current) => current ? {
      ...current,
      accounts: current.accounts.map((item) => item.subject === account.subject ? account : item),
    } : current);
  }

  async function submitRental(event: FormEvent) {
    event.preventDefault();
    setBusy("create-rental"); onError(null);
    try {
      const created = await createRentalAccount({ ...rentalForm, rental_date: rentalForm.rental_date || null, return_date: rentalForm.return_date || null });
      setManagement((current) => current ? {
        ...current, accounts: [...current.accounts, created],
      } : current);
      setRentalForm(emptyRental);
      onNotice(`租單 ${created.username} 已建立。`);
    } catch (caught) {
      onError(caught instanceof Error ? caught.message : "租單建立失敗");
    } finally { setBusy(null); }
  }

  async function submitOps(event: FormEvent) {
    event.preventDefault();
    setBusy("create-ops"); onError(null);
    try {
      const created = await createOpsAccount(opsForm);
      setManagement((current) => current ? {
        ...current, accounts: [...current.accounts, created],
      } : current);
      setOpsForm(emptyOps);
      onNotice(`${kindLabel(created.kind)}帳號 ${created.username} 已建立。`);
    } catch (caught) {
      onError(caught instanceof Error ? caught.message : "帳號建立失敗");
    } finally { setBusy(null); }
  }

  function openEdit(account: ManagedAccount) {
    setEditTarget(account);
    setEditCustomerId(account.customer_id ?? account.rental_id ?? "");
    setEditRentalDate(account.rental_date ?? "");
    setEditReturnDate(account.return_date ?? "");
    setEditVehicleLabel(account.vehicle_label ?? "");
    setEditLicensePlate(account.license_plate ?? "");
    setEditDisplayName(account.display_name);
    setEditKind(account.kind === "admin" ? "admin" : "staff");
    setEditPassword("");
    setDialogError(null);
  }

  async function saveEdit(event: FormEvent) {
    event.preventDefault();
    if (!editTarget) return;
    setBusy("edit"); setDialogError(null); onError(null);
    try {
      const updated = editTarget.kind === "rental"
        ? await updateRentalAccount(editTarget.username, {
          customer_id: editCustomerId,
          display_name: editDisplayName,
          rental_date: editRentalDate || null,
          return_date: editReturnDate || null,
          vehicle_label: editVehicleLabel,
          license_plate: editLicensePlate,
          ...(editPassword ? { password: editPassword } : {}),
        })
        : await updateOpsAccount(editTarget.username, {
          display_name: editDisplayName,
          kind: editKind,
          ...(editPassword ? { password: editPassword } : {}),
        });
      updateLocal(updated);
      setEditTarget(null);
      onNotice(`${kindLabel(updated.kind)} ${updated.username} 已更新。`);
    } catch (caught) {
      setDialogError(caught instanceof Error ? caught.message : "帳號更新失敗");
    } finally { setBusy(null); }
  }

  function openDelete(account: ManagedAccount) {
    setDeleteTarget(account);
    setDeleteStep(1);
    setDeleteIdentifier("");
    setAdminPassword("");
    setDialogError(null);
  }

  function closeDelete() {
    if (busy === "delete") return;
    setDeleteTarget(null);
    setDeleteStep(1);
    setDeleteIdentifier("");
    setAdminPassword("");
    setDialogError(null);
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    setBusy("delete"); setDialogError(null); onError(null);
    try {
      const response = await deleteManagedAccount(
        deleteTarget,
        deleteIdentifier,
        adminPassword,
      );
      setManagement(response.management);
      onNotice(`${kindLabel(deleteTarget.kind)} ${deleteTarget.username} 已刪除。`);
      setDeleteTarget(null);
      setDeleteStep(1);
      setDeleteIdentifier("");
      setAdminPassword("");
    } catch (caught) {
      setDialogError(caught instanceof Error ? caught.message : "帳號刪除失敗");
    } finally { setBusy(null); }
  }

  const rentals = (management?.accounts.filter((account) => account.kind === "rental") ?? [])
    .sort((a, b) => Number(b.is_active || b.is_current) - Number(a.is_active || a.is_current));
  const opsAccounts = management?.accounts.filter((account) => account.kind !== "rental") ?? [];

  return <section className="account-manager">
    <div className="account-manager__heading">
      <div>
        <p className="eyebrow">ACCOUNT CONTROL</p>
        <h2>租單與人員帳號管理</h2>
        <p>租單、管理者與營運人員皆使用帳號密碼登入。使用中的帳號必須先登出才能刪除。</p>
      </div>
      <div className={`account-live-summary ${management?.analysis_queue_at_capacity ? "account-live-summary--warning" : ""}`}>
        <div className="account-live-metrics">
          <div><strong>{management?.active_account_count ?? "—"}</strong><span>目前登入帳號</span></div>
          <div><strong>{management?.active_device_count ?? "—"}</strong><span>近 5 分鐘活躍裝置</span></div>
          <div><strong>{management?.active_inspection_count ?? "—"}</strong><span>進行中檢測流程</span></div>
          <div>
            <strong>{management ? management.queued_analysis_job_count + management.processing_analysis_job_count : "—"}</strong>
            <span>AI 待處理工作</span>
          </div>
        </div>
        <small>
          有效工作階段 {management?.active_session_count ?? "—"} 個；
          AI 排隊 {management?.queued_analysis_job_count ?? "—"}、處理中 {management?.processing_analysis_job_count ?? "—"}、
          上限 {management?.analysis_queue_capacity ?? "—"}。
        </small>
        {management?.analysis_queue_at_capacity && <div className="account-load-warning" role="alert">AI 佇列已滿，新照片會要求稍後再試。</div>}
        <small>本帳號：{currentPrincipal.display_name}</small>
        <button type="button" disabled={busy === "refresh"} onClick={() => {
          setBusy("refresh");
          void refresh().finally(() => setBusy(null));
        }}>{busy === "refresh" ? "更新中…" : "更新使用狀態"}</button>
      </div>
    </div>

    <div className="account-create-grid">
      <form className="account-create-card" onSubmit={(event) => void submitRental(event)}>
        <div><span>新增租單</span><small>設定租單編號、車輛、車牌與登入密碼</small></div>
        <label><span>租單編號</span><input required maxLength={80} value={rentalForm.rental_id} onChange={(event) => setRentalForm({ ...rentalForm, rental_id: event.target.value })} /></label>
        <label><span>使用者姓名</span><input required maxLength={120} value={rentalForm.display_name} onChange={(event) => setRentalForm({ ...rentalForm, display_name: event.target.value })} /></label>
        <label><span>租車日期</span><input required type="date" value={rentalForm.rental_date} onChange={(event) => setRentalForm({ ...rentalForm, rental_date: event.target.value })} /></label>
        <label><span>還車日期</span><input required type="date" min={rentalForm.rental_date || undefined} value={rentalForm.return_date} onChange={(event) => setRentalForm({ ...rentalForm, return_date: event.target.value })} /></label>
        <label><span>使用者編號</span><input required maxLength={80} placeholder="同一使用者的不同租單請填相同編號" value={rentalForm.customer_id} onChange={(event) => setRentalForm({ ...rentalForm, customer_id: event.target.value })} /></label>
        <label><span>車輛名稱</span><input required maxLength={120} value={rentalForm.vehicle_label} onChange={(event) => setRentalForm({ ...rentalForm, vehicle_label: event.target.value })} /></label>
        <label><span>車牌號碼</span><input required minLength={5} maxLength={10} placeholder="請輸入車牌" value={rentalForm.license_plate} onChange={(event) => setRentalForm({ ...rentalForm, license_plate: event.target.value.toUpperCase() })} /></label>
        <label><span>密碼（4 位數字）</span><input required minLength={4} maxLength={4} pattern="[0-9]{4}" inputMode="numeric" type="password" autoComplete="new-password" value={rentalForm.password} onChange={(event) => setRentalForm({ ...rentalForm, password: event.target.value })} /></label>
        <button className="primary" disabled={busy === "create-rental"}>{busy === "create-rental" ? "建立中…" : "新增租單"}</button>
      </form>

      <form className="account-create-card" onSubmit={(event) => void submitOps(event)}>
        <div><span>新增管理者／營運人員</span><small>密碼至少 8 個字元</small></div>
        <label><span>登入帳號</span><input required maxLength={80} autoComplete="off" value={opsForm.username} onChange={(event) => setOpsForm({ ...opsForm, username: event.target.value })} /></label>
        <label><span>顯示名稱</span><input required maxLength={120} value={opsForm.display_name} onChange={(event) => setOpsForm({ ...opsForm, display_name: event.target.value })} /></label>
        <label><span>角色</span><select value={opsForm.kind} onChange={(event) => setOpsForm({ ...opsForm, kind: event.target.value as "staff" | "admin" })}><option value="staff">營運人員</option><option value="admin">管理者</option></select></label>
        <label><span>密碼</span><input required minLength={8} maxLength={120} type="password" autoComplete="new-password" value={opsForm.password} onChange={(event) => setOpsForm({ ...opsForm, password: event.target.value })} /></label>
        <button className="primary" disabled={busy === "create-ops"}>{busy === "create-ops" ? "建立中…" : "新增人員帳號"}</button>
      </form>
    </div>

    {!management && <div className="account-loading">正在載入帳號資料…</div>}
    {management && <div className="account-groups">
      {[{ title: "租單", accounts: rentals, isRental: true }, { title: "管理者與營運人員", accounts: opsAccounts, isRental: false }].map((group) => <div className="account-group" key={group.title}>
        <div className="account-group__title"><h3>{group.title}</h3><span>{group.accounts.length} 筆</span></div>
        <div className="account-list" id={group.isRental ? "managed-rental-list" : undefined}>
          {(group.isRental && !rentalsExpanded ? group.accounts.slice(0, RENTAL_PREVIEW_COUNT) : group.accounts).map((account) => <article className="account-card" key={account.subject}>
            <div className="account-card__identity">
              <span className={`account-kind account-kind--${account.kind}`}>{kindLabel(account.kind)}</span>
              <div><strong>{account.username}</strong><small>{account.display_name}</small></div>
              {account.is_current
                ? <em className="account-status account-status--current">目前登入</em>
                : account.is_active
                  ? <em className="account-status account-status--active">使用中</em>
                  : <em className="account-status">未登入</em>}
            </div>
            {account.kind === "rental" && <dl className="account-card__details"><div><dt>使用者編號</dt><dd>{account.customer_id ?? account.rental_id}</dd></div><div><dt>車牌號碼</dt><dd>{account.license_plate ?? "尚未設定"}</dd></div><div><dt>車輛名稱</dt><dd>{account.vehicle_label}</dd></div></dl>}
            <div className="account-card__meta">
              <span>建立者：<strong>{account.created_by_name}</strong></span>
              <span>建立時間：<strong>{formatTime(account.created_at)}</strong></span>
            </div>
            <div className="account-card__actions">
              {account.kind === "rental" && <small>租車：{account.rental_date ?? "尚未設定"}／還車：{account.return_date ?? "尚未設定"}</small>}
              <button type="button" onClick={() => openEdit(account)}>修改</button>
              <button type="button" className="danger-outline" disabled={account.is_active || account.is_current} title={account.is_active || account.is_current ? "此帳號目前使用中，登出前不可刪除" : "刪除此帳號"} onClick={() => openDelete(account)}>刪除</button>
            </div>
          </article>)}
          {!group.accounts.length && <div className="account-empty">目前沒有{group.title}資料</div>}
        </div>
        {group.isRental && rentals.length > RENTAL_PREVIEW_COUNT && <button
          type="button"
          className="account-group__toggle"
          aria-expanded={rentalsExpanded}
          aria-controls="managed-rental-list"
          onClick={() => setRentalsExpanded((expanded) => !expanded)}
        >{rentalsExpanded ? "收合租單" : `展開更多租單（還有 ${rentals.length - RENTAL_PREVIEW_COUNT} 筆）`}</button>}
      </div>)}
    </div>}

    {editTarget && <div className="model-delete-backdrop"><form className="model-delete-dialog account-dialog" role="dialog" aria-modal="true" onSubmit={(event) => void saveEdit(event)}>
      <p className="eyebrow">EDIT ACCOUNT</p><h2>修改{kindLabel(editTarget.kind)}</h2>
      <p>識別編號 <strong>{editTarget.username}</strong> 為資料關聯鍵，建立後不可修改。</p>
      {editTarget.kind === "rental" ? <>
        <label><span>使用者姓名</span><input required maxLength={120} value={editDisplayName} onChange={(event) => setEditDisplayName(event.target.value)} /></label>
        <label><span>租車日期</span><input type="date" value={editRentalDate} onChange={(event) => setEditRentalDate(event.target.value)} /></label>
        <label><span>還車日期</span><input type="date" min={editRentalDate || undefined} value={editReturnDate} onChange={(event) => setEditReturnDate(event.target.value)} /></label>
        <label><span>使用者編號</span><input required maxLength={80} value={editCustomerId} onChange={(event) => setEditCustomerId(event.target.value)} /></label>
        <label><span>車輛名稱</span><input required maxLength={120} value={editVehicleLabel} onChange={(event) => setEditVehicleLabel(event.target.value)} /></label>
        <label><span>車牌號碼</span><input required minLength={5} maxLength={10} placeholder="請輸入車牌" value={editLicensePlate} onChange={(event) => setEditLicensePlate(event.target.value.toUpperCase())} /></label>
        <label><span>新密碼（4 位數字，留白代表不變）</span><input minLength={4} maxLength={4} pattern="[0-9]{4}" inputMode="numeric" type="password" autoComplete="new-password" value={editPassword} onChange={(event) => setEditPassword(event.target.value)} /></label>
      </> : <>
        <label><span>顯示名稱</span><input required maxLength={120} value={editDisplayName} onChange={(event) => setEditDisplayName(event.target.value)} /></label>
        <label><span>角色</span><select disabled={editTarget.is_current} value={editKind} onChange={(event) => setEditKind(event.target.value as "staff" | "admin")}><option value="staff">營運人員</option><option value="admin">管理者</option></select>{editTarget.is_current && <small>目前登入帳號不可變更自己的角色</small>}</label>
        <label><span>新密碼（留白代表不變）</span><input minLength={8} maxLength={120} type="password" autoComplete="new-password" value={editPassword} onChange={(event) => setEditPassword(event.target.value)} /></label>
      </>}
      {dialogError && <div className="model-delete-error" role="alert">{dialogError}</div>}
      <div className="model-delete-actions"><button type="button" disabled={busy === "edit"} onClick={() => setEditTarget(null)}>取消</button><button type="submit" className="primary" disabled={busy === "edit"}>{busy === "edit" ? "儲存中…" : "儲存修改"}</button></div>
    </form></div>}

    {deleteTarget && <div className="model-delete-backdrop"><section className="model-delete-dialog account-dialog" role="dialog" aria-modal="true">
      {deleteTarget.kind === "rental" ? <>
        <p className="eyebrow">DELETE RENTAL</p><h2>刪除租單？</h2>
        <p>即將刪除租單 <strong>{deleteTarget.username}</strong>（{deleteTarget.display_name}／{deleteTarget.license_plate ?? "尚未設定車牌"}）。</p>
        <p>一併刪除此租單的登入帳號、密碼及所有登入連線，此租單將無法再登入。</p>
        <p>租還車檢測、照片、AI 分析、車損與賠償、營運紀錄及使用者黑名單仍會保留；同一使用者的其他租單不受影響。</p>
        {dialogError && <div className="model-delete-error" role="alert">{dialogError}</div>}
        <div className="model-delete-actions"><button type="button" disabled={busy === "delete"} onClick={closeDelete}>取消</button><button type="button" className="danger" disabled={busy === "delete"} onClick={() => void confirmDelete()}>{busy === "delete" ? "刪除中…" : "確認刪除"}</button></div>
      </> : deleteStep === 1 ? <>
        <p className="eyebrow">DELETE ACCOUNT · 1 / 2</p><h2>第一次確認：刪除{kindLabel(deleteTarget.kind)}？</h2>
        <p>即將刪除 <strong>{deleteTarget.username}</strong>。歷史營運與租還車紀錄仍會保留，但此帳號將無法再登入。</p>
        <div className="delete-safety-note">使用中的帳號由前端與後端雙重鎖定，必須先登出才能刪除。</div>
        <div className="model-delete-actions"><button type="button" onClick={closeDelete}>取消</button><button type="button" className="danger" onClick={() => setDeleteStep(2)}>我了解，繼續確認</button></div>
      </> : <>
        <p className="eyebrow">DELETE ACCOUNT · 2 / 2</p><h2>第二次確認：輸入識別編號</h2>
        <p>請輸入 <strong>{deleteTarget.username}</strong>，並以目前管理員密碼重新驗證。</p>
        <label><span>確認識別編號</span><input autoFocus value={deleteIdentifier} onChange={(event) => setDeleteIdentifier(event.target.value)} /></label>
        <label><span>目前管理員密碼</span><input type="password" autoComplete="current-password" value={adminPassword} onChange={(event) => setAdminPassword(event.target.value)} /></label>
        {dialogError && <div className="model-delete-error" role="alert">{dialogError}</div>}
        <div className="model-delete-actions"><button type="button" disabled={busy === "delete"} onClick={() => { setDeleteStep(1); setDialogError(null); }}>返回</button><button type="button" className="danger" disabled={busy === "delete" || deleteIdentifier !== deleteTarget.username || !adminPassword} onClick={() => void confirmDelete()}>{busy === "delete" ? "驗證並刪除中…" : "驗證並刪除"}</button></div>
      </>}
    </section></div>}
  </section>;
}
