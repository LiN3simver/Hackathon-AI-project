import { useCallback, useEffect, useState } from "react";

import {
  acknowledgeAlert,
  authenticatedAssetUrl,
  cancelAlert,
  clearDamageReviewDuplicates,
  clearPendingDamageReview,
  clearTestData,
  createOperationsWorkOrder,
  damageCaseImageUrl,
  deleteWorkOrder,
  dispatchNotificationOutbox,
  getAlerts,
  getDamageCases,
  getNotificationOutbox,
  getWorkOrders,
  reanalyzeDamageCase,
  reviewDamageCase,
  reviewCleanlinessAlert,
  retryNotification,
  type DamageCase,
  type OperationsAlert,
  type NotificationOutboxItem,
  type WorkOrder,
  updateWorkOrder,
} from "../lib/api";
import { ZoomableImage, type ImageHighlight } from "./ImageViewer";
import { InspectionPhotoRecords } from "./InspectionPhotoRecords";
import { PhotoListControls, usePhotoListPager } from "./PhotoListPager";
import { UserHistoryExplorer } from "./UserHistoryExplorer";
import { confirmDeletion } from "../lib/confirmDeletion";
import { OperationsActivityHistory } from "./OperationsActivityHistory";

const statusLabel = {
  open: "待確認", acknowledged: "已確認", resolved: "已結案", cancelled: "已取消",
  queued: "待指派", assigned: "已指派", in_progress: "處理中", completed: "已完成",
};

/** A dirt alert is only high severity when the model actually marked where the dirt is. */
function isUnlocatedCleanliness(alert: OperationsAlert): boolean {
  return alert.type === "cleanliness" && cleanlinessHighlights(alert).length === 0;
}

function cleanlinessHighlights(alert: OperationsAlert): ImageHighlight[] {
  if (alert.type !== "cleanliness") return [];
  return alert.evidence.flatMap((item) => {
    const box = item.bounding_box;
    if (!item.visible || !box || Array.isArray(box)) return [];
    const { x1, y1, x2, y2 } = box;
    if (![x1, y1, x2, y2].every(Number.isFinite)
      || x1 < 0 || y1 < 0 || x2 > 1 || y2 > 1 || x2 <= x1 || y2 <= y1) return [];
    return [{ x: x1, y: y1, width: x2 - x1, height: y2 - y1,
      kind: "issue" as const, label: item.description }];
  });
}

const notificationStatusLabel = {
  pending: "待派送",
  processing: "派送中",
  power_automate_queued: "已排入",
  power_automate_failed: "寄送失敗",
  sent: "已寄出",
  failed: "寄送失敗",
};

type WorkOrderChanges = Parameters<typeof updateWorkOrder>[1];

const viewLabel: Record<string, string> = {
  left_front: "左前", right_front: "右前", left_rear: "左後", right_rear: "右後",
  damage_detail: "車損特寫",
};

/** Which review pile a pending case belongs to; each pile has its own buttons. */
type DamageReviewKind = "damage" | "clear" | "retake";

const damageReviewTabs: { kind: DamageReviewKind; label: string; hint: string }[] = [
  { kind: "damage", label: "疑似車損", hint: "AI 看到新傷，請確認是不是真的。" },
  { kind: "clear", label: "沒看到車損", hint: "AI 沒看到新傷，請再看一眼兩張照片。" },
  { kind: "retake", label: "照片對不上", hint: "兩張角度不同，無法比較，請決定是否重拍。" },
];

function damageReviewKind(item: DamageCase): DamageReviewKind {
  if (item.comparison.status === "review_required") return "damage";
  // Aligned photos without a detection are a model "no damage" that a person still checks.
  return item.comparison.alignment.success ? "clear" : "retake";
}

// Sent to review only by pixel differences (light, shadow, ground) while the damage model stayed below its threshold.
function pixelDifferenceOnly(item: DamageCase): boolean {
  return damageReviewKind(item) === "damage" && item.comparison.specialist?.detected === false;
}

function damageFinding(item: DamageCase): string {
  const count = item.comparison.candidates.length;
  if (pixelDifferenceOnly(item)) return "像素差異，車損模型未偵測";
  if (damageReviewKind(item) === "damage") return count ? `AI 看到 ${count} 處可能新傷` : "AI 看到可能新傷";
  if (damageReviewKind(item) === "clear") return "AI 沒看到新傷";
  return "兩張照片角度對不上";
}

// Hide the digits of the mailbox name (often a student or staff ID) so screenshots stay shareable.
function maskRecipient(recipient: string): string {
  const at = recipient.indexOf("@");
  const name = at < 0 ? recipient : recipient.slice(0, at);
  return name.replace(/\d/g, "*") + (at < 0 ? "" : recipient.slice(at));
}

function formatTime(value: string | null): string {
  if (!value) return "尚未設定";
  return new Intl.DateTimeFormat("zh-TW", {
    month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(new Date(value));
}

/** Alert piles in the same order a case moves through them. */
type AlertTab = "open" | "acknowledged" | "closed";

const alertTabs: { tab: AlertTab; label: string; tone: string; hint: string }[] = [
  { tab: "open", label: "待確認", tone: "damage", hint: "AI 或人員發現的異常，請看照片決定要不要處理。" },
  { tab: "acknowledged", label: "已確認", tone: "retake", hint: "已確認要處理，到「整備工單」追蹤進度。" },
  { tab: "closed", label: "已結案", tone: "clear", hint: "工單完成或已取消的警示，留作紀錄。" },
];

function alertTab(alert: OperationsAlert): AlertTab {
  if (alert.status === "open") return "open";
  return alert.status === "acknowledged" ? "acknowledged" : "closed";
}

/** Work order piles by progress; cancelled orders are not listed anywhere. */
const workOrderTabs: { status: WorkOrder["status"]; label: string; tone: string; next: string }[] = [
  { status: "queued", label: "待指派", tone: "damage", next: "填寫負責人或整備站，按「儲存並指派」。" },
  { status: "assigned", label: "已指派", tone: "retake", next: "人員到車旁後按「開始處理」。" },
  { status: "in_progress", label: "處理中", tone: "progress", next: "處理完寫下結果，按「完成並自動結案」，警示會一起關閉。" },
  { status: "completed", label: "已完成", tone: "clear", next: "工單已完成，來源警示已自動結案。" },
];

const workOrderSteps: { status: WorkOrder["status"]; label: string }[] = [
  { status: "queued", label: "建立工單" },
  { status: "assigned", label: "指派負責人" },
  { status: "in_progress", label: "開始處理" },
  { status: "completed", label: "完成結案" },
];

function isOverdue(order: WorkOrder): boolean {
  return !["completed", "cancelled"].includes(order.status)
    && Boolean(order.due_at) && new Date(order.due_at!).getTime() < Date.now();
}

function WorkOrderRow({
  order,
  busy,
  expanded,
  canDelete,
  onToggle,
  onUpdate,
  onDelete,
}: {
  order: WorkOrder;
  busy: boolean;
  expanded: boolean;
  canDelete: boolean;
  onToggle: () => void;
  onUpdate: (changes: WorkOrderChanges) => void;
  onDelete: (order: WorkOrder) => void;
}) {
  const [assignee, setAssignee] = useState(order.assignee ?? "");
  const [notes, setNotes] = useState(order.notes);

  useEffect(() => {
    setAssignee(order.assignee ?? "");
    setNotes(order.notes);
  }, [order.assignee, order.notes]);

  const active = order.status !== "completed";
  const overdue = isOverdue(order);
  const dirty = assignee.trim() !== (order.assignee ?? "") || notes.trim() !== order.notes;
  const saveLabel = order.status === "queued" ? "儲存並指派" : "儲存負責人與備註";
  const stepIndex = workOrderSteps.findIndex((step) => step.status === order.status);
  const latest = order.history.at(-1);
  const panelId = `work-order-panel-${order.id}`;

  return <article className={`order-row ${overdue ? "order-row--overdue" : ""} ${expanded ? "order-row--open" : ""}`}>
    <button type="button" className="order-row__summary" aria-expanded={expanded} aria-controls={panelId} onClick={onToggle}>
      <span className="order-row__chevron" aria-hidden="true">{expanded ? "▾" : "▸"}</span>
      <span className="order-row__main">
        <strong>{order.vehicle_id} · {order.kind === "vehicle_inspection" ? "車身驗傷" : "車內清潔"}</strong>
        <small>{order.id} · {order.assignee ? `負責：${order.assignee}` : "尚未指派"}</small>
        {latest && <small className="order-row__latest">最新：{latest.summary}（{formatTime(latest.created_at)}）</small>}
      </span>
      <span className={`order-row__due ${overdue ? "overdue" : ""}`}>
        <small>{overdue ? "已超過處理期限" : order.status === "completed" ? "完成時間" : "處理期限"}</small>
        <b>{formatTime(order.status === "completed" ? order.completed_at : order.due_at)}</b>
      </span>
      <span className={`order-row__status order-row__status--${order.status}`}>{statusLabel[order.status]}</span>
    </button>
    {expanded && <div className="order-row__panel" id={panelId}>
      <ol className="order-steps" aria-label="工單進度">
        {workOrderSteps.map((step, index) => <li
          key={step.status}
          className={index < stepIndex || order.status === "completed" ? "done" : index === stepIndex ? "current" : ""}
        >{step.label}</li>)}
      </ol>
      <p className="order-next">
        <b>{order.status === "completed" ? "目前狀態" : "下一步"}</b>
        {workOrderTabs.find((tab) => tab.status === order.status)?.next}
      </p>
      <label className="order-field">
        <span>負責人／整備站</span>
        <input
          value={assignee}
          disabled={!active || busy}
          maxLength={80}
          placeholder="例如：台北整備 A 組"
          onChange={(event) => setAssignee(event.target.value)}
        />
      </label>
      <label className="order-field">
        <span>處理備註</span>
        <textarea
          value={notes}
          disabled={!active || busy}
          maxLength={500}
          rows={3}
          placeholder="記錄清潔、驗車或聯絡結果"
          onChange={(event) => setNotes(event.target.value)}
        />
      </label>
      {(active || canDelete) && <div className="card-actions order-actions">
        {active && <button
          disabled={busy || !dirty || (order.status === "queued" && !assignee.trim())}
          onClick={() => onUpdate({ assignee: assignee.trim(), notes: notes.trim() })}
        >{busy ? "處理中…" : saveLabel}</button>}
        {order.status === "assigned" && <button
          className="primary"
          disabled={busy || !assignee.trim()}
          onClick={() => onUpdate({
            status: "in_progress", assignee: assignee.trim(), notes: notes.trim(),
          })}
        >開始處理</button>}
        {order.status === "in_progress" && <button
          className="primary"
          disabled={busy}
          onClick={() => onUpdate({
            status: "completed", assignee: assignee.trim(), notes: notes.trim(),
          })}
        >完成並自動結案</button>}
        {canDelete && <button
          className="danger-outline"
          disabled={busy}
          onClick={() => onDelete(order)}
        >刪除工單</button>}
      </div>}
      <section className="order-history" aria-label="處理紀錄">
        <h4>處理紀錄</h4>
        {order.history.length === 0
          ? <p>尚無紀錄</p>
          : <ol>
            {order.history.map((step, index) => <li key={`${step.created_at}-${index}`}>
              <time dateTime={step.created_at}>{formatTime(step.created_at)}</time>
              <span>{step.summary}</span>
              <small>{step.actor.replace(/^ops:/, "")}</small>
            </li>)}
          </ol>}
      </section>
    </div>}
  </article>;
}

export function OpsDashboard({
  sidebarCollapsed,
  onCloseSidebar,
  onEnterSidebar,
  onLeaveSidebar,
  canDeleteInspectionPhotos,
  canManageNotifications,
  canDeleteWorkOrders,
  showMyActivity,
}: {
  sidebarCollapsed: boolean;
  onCloseSidebar: () => void;
  onEnterSidebar: () => void;
  onLeaveSidebar: () => void;
  canDeleteInspectionPhotos: boolean;
  canManageNotifications: boolean;
  canDeleteWorkOrders: boolean;
  showMyActivity: boolean;
}) {
  const [alerts, setAlerts] = useState<OperationsAlert[]>([]);
  const [orders, setOrders] = useState<WorkOrder[]>([]);
  const [damageCases, setDamageCases] = useState<DamageCase[]>([]);
  const [reanalyzingId, setReanalyzingId] = useState<string | null>(null);
  const [damageTab, setDamageTab] = useState<DamageReviewKind>("damage");
  const [alertTabView, setAlertTabView] = useState<AlertTab>("open");
  const [orderTab, setOrderTab] = useState<WorkOrder["status"]>("queued");
  const [expandedOrderId, setExpandedOrderId] = useState<string | null>(null);
  const [notifications, setNotifications] = useState<NotificationOutboxItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [dispatchConfirmOpen, setDispatchConfirmOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<Date | null>(null);
  const [learningRefreshKey, setLearningRefreshKey] = useState(0);
  const [activeSection, setActiveSection] = useState(() => {
    const section = window.location.hash.slice(1);
    const validSections = ["ops-overview", "inspection-photos", "user-history", "damage-review", "operations-alerts", "work-orders", "notification-outbox", ...(showMyActivity ? ["my-activity"] : [])];
    if (validSections.includes(section)) return section;
    const savedSection = window.localStorage.getItem("guardian-ops-section");
    return savedSection && validSections.includes(savedSection) ? savedSection : "ops-overview";
  });

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const [
        nextAlerts,
        nextOrders,
        nextDamageCases,
        nextNotifications,
      ] = await Promise.all([
        getAlerts(),
        getWorkOrders(),
        getDamageCases(),
        getNotificationOutbox(),
      ]);
      setAlerts(nextAlerts); setOrders(nextOrders); setDamageCases(nextDamageCases);
      setNotifications(nextNotifications); setError(null);
      setLastRefreshedAt(new Date());
      setLearningRefreshKey((current) => current + 1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "讀取營運資料失敗");
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  // Open the first pile that has work when the current one becomes empty.
  useEffect(() => {
    const kinds = damageCases
      .filter((item) => item.review_decision === "pending")
      .map(damageReviewKind);
    setDamageTab((current) => kinds.includes(current)
      ? current
      : damageReviewTabs.find((tab) => kinds.includes(tab.kind))?.kind ?? current);
  }, [damageCases]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 1500);
    return () => window.clearTimeout(timer);
  }, [notice]);

  async function act(id: string, action: () => Promise<unknown>, successMessage: string) {
    setBusy(id);
    setError(null);
    try { await action(); setNotice(successMessage); await refresh(); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "操作失敗"); }
    finally { setBusy(null); }
  }

  async function reviewCase(
    id: string,
    decision: "dismissed" | "retry_photos" | "confirmed_damage",
    reviewerNotes: string,
  ) {
    setBusy(id);
    setError(null);
    setNotice(null);
    try {
      await reviewDamageCase(id, decision, reviewerNotes);
      setNotice(decision === "confirmed_damage"
        ? "已確認新傷，警示已送出。"
        : decision === "retry_photos"
          ? "已要求重拍。"
          : "已記錄：沒有車損。");
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "車損覆核失敗");
    } finally {
      setBusy(null);
    }
  }

  async function reanalyzeCase(id: string) {
    setBusy(id);
    setReanalyzingId(id);
    setError(null);
    setNotice(null);
    try {
      const updated = await reanalyzeDamageCase(id);
      setDamageCases((current) => current.map((item) => item.id === id ? updated : item));
      setNotice(`已重新辨識 ${id}：${damageFinding(updated)}。`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "重新辨識失敗");
    } finally {
      setBusy(null);
      setReanalyzingId(null);
    }
  }

  async function reviewCleanliness(id: string, decision: "confirmed" | "dismissed") {
    setBusy(id);
    setError(null);
    setNotice(null);
    try {
      await reviewCleanlinessAlert(id, decision);
      // One tap: a confirmed stain always needs cleaning, so open the work order too.
      if (decision === "confirmed" && !orders.some((order) => order.alert_id === id)) {
        await createOperationsWorkOrder(id);
      }
      setNotice(decision === "confirmed"
        ? "已確認髒污並建立清潔工單。"
        : "已記錄：沒有髒污。");
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "髒污確認失敗");
    } finally {
      setBusy(null);
    }
  }

  async function clearAllTestData() {
    if (!confirmDeletion(
      "將清除整個營運中心的全部警示、車損覆核案件、整備工單、Email 通知與操作紀錄（不只目前頁面，也不限測試租單）。目前畫面有警示 " + alerts.length + " 筆、覆核 " + damageCases.length + " 件、工單 " + orders.length + " 張、通知 " + notifications.length + " 封。租還車原圖、照片紀錄、帳號、拍攝流程及主動學習資料不會刪除，但上述頁面的圖片卡片會消失。此操作無法復原。"
    )) return;
    setBusy("clear-test-data");
    setError(null);
    setNotice(null);
    try {
      const result = await clearTestData();
      setNotice(`已清除營運紀錄：${result.deleted_damage_cases} 件覆核、${result.deleted_alerts} 筆警示、${result.deleted_work_orders} 張工單。`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "清除營運紀錄失敗");
    } finally {
      setBusy(null);
    }
  }

  async function clearDamageReviewCases() {
    if (!window.confirm(
      "確定清除已存在於「人工真值」的重複車損覆核案件嗎？\n\n只清除與已確認真值相同的照片案件及關聯分析工作；其他待覆核案件、原始照片、警示、工單與人工真值會保留。此操作無法復原。",
    )) return;
    setBusy("clear-damage-review");
    setError(null);
    setNotice(null);
    try {
      const result = await clearDamageReviewDuplicates();
      setNotice(`已清除 ${result.deleted_damage_cases} 件與人工真值重複的覆核案件。`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "清除新車損人工覆核資料失敗");
    } finally {
      setBusy(null);
    }
  }

  async function clearPendingDamageCases() {
    if (!window.confirm(
      `確定清除目前列出的 ${pendingDamageCases.length} 件待覆核車損案件嗎？

只移除這個清單中的案件；學習樣本、原始照片、警示、工單與人工真值都會保留。此操作無法復原。`,
    )) return;
    setBusy("clear-pending-damage");
    setError(null);
    setNotice(null);
    try {
      const result = await clearPendingDamageReview();
      setNotice(`已清除 ${result.deleted_damage_cases} 件待覆核車損案件。`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "清除待覆核車損案件失敗");
    } finally {
      setBusy(null);
    }
  }

  async function dispatchEmails() {
    setBusy("notification-dispatch");
    setError(null);
    try {
      const result = await dispatchNotificationOutbox();
      setNotice(`通知處理完成：${result.handed_off} 筆交給 Power Automate、${result.failed} 筆失敗。`);
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Email 通知派送失敗");
    } finally {
      setBusy(null);
      setDispatchConfirmOpen(false);
    }
  }

  async function saveWorkOrder(order: WorkOrder, changes: WorkOrderChanges) {
    setBusy(order.id);
    setError(null);
    try {
      const saved = await updateWorkOrder(order.id, changes);
      if (saved.status !== order.status) {
        // Follow the order into its new pile and keep it open, so the step just taken stays visible.
        setOrderTab(saved.status);
        setExpandedOrderId(saved.id);
      }
      setNotice(saved.status === order.status
        ? "工單已更新。"
        : saved.status === "assigned"
          ? `已指派給 ${saved.assignee}，工單移到「已指派」。`
          : saved.status === "in_progress"
            ? "已開始處理，工單移到「處理中」。"
            : "工單已完成，來源警示已結案，工單移到「已完成」。");
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "工單更新失敗");
    } finally {
      setBusy(null);
    }
  }

  async function removeWorkOrder(order: WorkOrder) {
    if (!confirmDeletion(
      `確定刪除工單 ${order.id}（${order.vehicle_id}）嗎？\n\n工單會從清單移除，來源警示保留，之後可重新建立工單；刪除動作會留在操作紀錄。此操作無法復原。`,
    )) return;
    setExpandedOrderId(null);
    await act(order.id, () => deleteWorkOrder(order.id), `已刪除工單 ${order.id}。`);
  }

  const openAlerts = alerts.filter((item) => item.status === "open").length;
  const activeOrders = orders.filter((item) => !["completed", "cancelled"].includes(item.status)).length;
  const unassignedOrders = orders.filter((item) => item.status === "queued").length;
  const overdueOrders = orders.filter(isOverdue).length;
  const visibleOrders = orders.filter((item) => item.status !== "cancelled");
  const orderTabItems = visibleOrders.filter((item) => item.status === orderTab);
  const pendingDamage = damageCases.filter((item) => item.review_decision === "pending").length;
  const pendingNotifications = notifications.filter(
    (item) => item.status === "pending" || item.status === "processing",
  ).length;
  const failedNotifications = notifications.filter(
    (item) => item.status === "failed" || item.status === "power_automate_failed",
  ).length;
  const handedOffNotifications = notifications.filter(
    (item) => item.status === "power_automate_queued",
  ).length;
  // Mirrors the backend claim rule: pending or failed, attempts left, retry time reached.
  const dispatchableNotifications = notifications.filter(
    (item) => (item.status === "pending" || item.status === "failed")
      && item.attempts < item.max_attempts
      && (!item.next_attempt_at || new Date(item.next_attempt_at).getTime() <= Date.now()),
  );
  const dispatchRecipients = [...new Set(dispatchableNotifications.map((item) => item.recipient))];
  const pendingDamageCases = damageCases.filter((item) => item.review_decision === "pending");
  const damageTabCases = pendingDamageCases.filter((item) => damageReviewKind(item) === damageTab);
  const damageCasePager = usePhotoListPager(damageTabCases);
  // Located (HIGH) alerts first; the stable sort keeps newest-first within each group.
  const alertTabItems = alerts
    .filter((item) => alertTab(item) === alertTabView)
    .sort((a, b) => Number(isUnlocatedCleanliness(a)) - Number(isUnlocatedCleanliness(b)));
  const alertPager = usePhotoListPager(alertTabItems);
  const navigationItems = [
    { id: "ops-overview", label: "營運總覽" },
    { id: "inspection-photos", label: "租還車照片紀錄" },
    { id: "user-history", label: "借還車AI助理" },
    { id: "damage-review", label: "新車損人工覆核", count: pendingDamage },
    { id: "operations-alerts", label: "異常警示", count: openAlerts },
    { id: "work-orders", label: "整備工單", count: activeOrders },
    {
      id: "notification-outbox",
      label: "Email 通知",
      count: pendingNotifications + failedNotifications,
    },
    ...(showMyActivity ? [{ id: "my-activity", label: "我的操作紀錄" }] : []),
  ];
  return (
    <>
      <aside id="ops-sidebar-navigation" className={`ops-sidebar ${sidebarCollapsed ? "ops-sidebar--collapsed" : ""}`} hidden={sidebarCollapsed} onMouseEnter={onEnterSidebar} onMouseLeave={onLeaveSidebar}>
        <>
          <div className="ops-sidebar-heading">
            <span aria-hidden="true">G</span>
            <strong>快速導覽</strong>
          </div>
          <nav aria-label="營運中心快速導覽">
            {navigationItems.map((item) => <button
              type="button"
              key={item.id}
              className={activeSection === item.id ? "active" : ""}
              aria-current={activeSection === item.id ? "page" : undefined}
              onClick={() => { setActiveSection(item.id); window.localStorage.setItem("guardian-ops-section", item.id); onCloseSidebar(); window.history.replaceState(null, "", `#${item.id}`); window.scrollTo(0, 0); }}
            >
              <b>{item.label}</b>
              {"count" in item && <small>{item.count}</small>}
            </button>)}
          </nav>
        </>
      </aside>
      <main className={`ops-page ${activeSection === "user-history" ? "ops-page--assistant" : ""} ${sidebarCollapsed
        ? "ops-page--sidebar-collapsed"
        : "ops-page--sidebar-open"}`}>
      <div className="ops-section" hidden={activeSection !== "ops-overview"}>
      <section className="ops-hero" id="ops-overview">
        <div><p className="eyebrow">OPERATIONS CONTROL TOWER</p><h1>車況營運中心</h1><p>AI 異常集中到單一佇列，由人員確認、派工並留下處理軌跡。</p></div>
        <div className="ops-refresh">
          <button type="button" disabled={refreshing} onClick={() => void refresh()} aria-busy={refreshing}>
            {refreshing ? "重新整理中…" : "重新整理"}
          </button>
          <small aria-live="polite">
            {lastRefreshedAt
              ? `已更新 ${lastRefreshedAt.toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`
              : "尚未完成資料同步"}
          </small>
        </div>
      </section>
      <section className="ops-kpis">
        <div><span>待確認警示</span><strong>{openAlerts}</strong></div>
        <div><span>進行中工單</span><strong>{activeOrders}</strong></div>
        <div><span>尚未指派</span><strong>{unassignedOrders}</strong></div>
        <div><span>超過處理期限</span><strong>{overdueOrders}</strong></div>
        <div><span>車損待覆核</span><strong>{pendingDamage}</strong></div>
        <div><span>Email 待處理</span><strong>{pendingNotifications + failedNotifications}</strong></div>
      </section>
      <section className="ops-title" aria-label="營運測試資料清理">
        <div><h2>清理營運紀錄</h2><p>清除所有警示、覆核、工單、Email 通知與操作紀錄；租還車原圖及訓練資料保留。</p></div>
        <div className="ops-title-actions"><button className="clear-test-data" type="button" disabled={busy === "clear-test-data"} onClick={() => void clearAllTestData()}>
          {busy === "clear-test-data" ? "清除中…" : "清除全部營運紀錄"}
        </button></div>
      </section>
      </div>
      {error && <div className="error-box"><strong>營運資料載入失敗</strong><span>{error}</span></div>}
      {notice && <div className="ops-notice" role="status" aria-live="polite">
        <span className="ops-notice-icon" aria-hidden="true">✓</span>
        <div><strong>營運通知</strong><p>{notice}</p></div>
        <button type="button" aria-label="關閉營運通知" onClick={() => setNotice(null)}>×</button>
      </div>}

      <div className="ops-section" hidden={activeSection !== "inspection-photos"}>
      <InspectionPhotoRecords
        refreshKey={learningRefreshKey}
        canDelete={canDeleteInspectionPhotos}
      />
      </div>

      <div className="ops-section" hidden={activeSection !== "user-history"}>
      <UserHistoryExplorer canManage={canDeleteInspectionPhotos} isActive={activeSection === "user-history"} />
      </div>

      <section className="damage-review-queue ops-section" id="damage-review" hidden={activeSection !== "damage-review"}>
        <div className="ops-title">
          <div><h2>新車損人工覆核</h2><p>看取車、還車兩張照片，按一個按鈕。</p></div>
          <div className="ops-title-actions damage-review-tools">
            <button
              type="button"
              disabled={busy !== null}
              onClick={() => void clearDamageReviewCases()}
            >{busy === "clear-damage-review" ? "清除中…" : "清除重複"}</button>
            <button
              type="button"
              disabled={busy !== null || pendingDamageCases.length === 0}
              onClick={() => void clearPendingDamageCases()}
            >{busy === "clear-pending-damage" ? "清除中…" : "全部清除"}</button>
          </div>
        </div>
        <div className="damage-review-tabs" role="tablist" aria-label="車損覆核分類">
          {damageReviewTabs.map((tab) => <button
            type="button"
            role="tab"
            key={tab.kind}
            className={`damage-review-tab damage-review-tab--${tab.kind}`}
            aria-selected={damageTab === tab.kind}
            onClick={() => { setDamageTab(tab.kind); damageCasePager.reset(); }}
          >
            {tab.label}
            <b>{pendingDamageCases.filter((item) => damageReviewKind(item) === tab.kind).length}</b>
          </button>)}
        </div>
        <p className="damage-review-hint">{damageReviewTabs.find((tab) => tab.kind === damageTab)?.hint}</p>
        {damageTabCases.length === 0 && <div className="empty-state">這一區沒有要看的照片</div>}
        <div className="damage-review-grid">
          {damageCasePager.visibleItems.map((item) => {
            const kind = damageReviewKind(item);
            const specialist = item.comparison.specialist;
            return <article className={`damage-review-card damage-review-card--${kind}`} key={item.id}>
              {/* Untouched photos first; the aligned overlay is only supporting evidence. */}
              <div className="damage-review-media" aria-label={`${item.id} 原始比對照片`}>
                <div className="damage-review-photo">
                  <span>取車</span>
                  <ZoomableImage
                    className="damage-review-source-image"
                    src={damageCaseImageUrl(item.id, "pickup")}
                    alt={`${item.id} 取車照`}
                    title={`取車照 · ${item.pickup_filename}`}
                    fallback={<div className="review-image-empty">取車照無法載入</div>}
                  />
                </div>
                <div className="damage-review-photo">
                  <span>還車</span>
                  <ZoomableImage
                    className="damage-review-source-image"
                    src={damageCaseImageUrl(item.id, "return")}
                    alt={`${item.id} 還車照`}
                    title={`還車照 · ${item.return_filename}`}
                    fallback={<div className="review-image-empty">還車照無法載入</div>}
                  />
                </div>
              </div>
              <div className="damage-review-body">
                <h3>{item.vehicle_id} · {viewLabel[item.expected_view] ?? item.expected_view}<small>{item.id}</small></h3>
                <ul className="damage-review-facts">
                  <li className={kind === "damage" && !pixelDifferenceOnly(item) ? "damage-specialist-flag" : undefined}>{damageFinding(item)}</li>
                  {specialist && <li>車損模型 {Math.round(specialist.score * 100)}%（門檻 {Math.round(specialist.threshold * 100)}%）</li>}
                  <li>租單 {item.rental_id}</li>
                </ul>
                {kind === "damage" && item.comparison.annotated_return_image && <ZoomableImage
                  className="damage-review-aligned-thumb"
                  src={item.comparison.annotated_return_image}
                  alt={`${item.id} AI 標示位置`}
                  title={`${item.vehicle_id} AI 標示的可能新傷`}
                  overlay={<span>⌕ 看 AI 標示</span>}
                />}
                <div className="card-actions">
                  {kind !== "retake" && <button className="danger" disabled={busy === item.id} onClick={() => void reviewCase(item.id, "confirmed_damage", "人工比對雙圖確認新增車損，轉驗車")}>有新傷</button>}
                  <button disabled={busy === item.id} onClick={() => void reviewCase(item.id, "dismissed", "人工判定沒有新增車損")}>沒車損</button>
                  <button disabled={busy === item.id} onClick={() => void reviewCase(item.id, "retry_photos", "照片無法判斷，要求依同角度重拍")}>要重拍</button>
                </div>
                <button
                  className="damage-reanalyze"
                  type="button"
                  disabled={busy === item.id}
                  onClick={() => void reanalyzeCase(item.id)}
                >{reanalyzingId === item.id ? "重新辨識中…" : "↻ 重新辨識"}</button>
              </div>
            </article>;
          })}
        </div>
        <PhotoListControls
          currentPage={damageCasePager.currentPage}
          expanded={damageCasePager.expanded}
          itemLabel="車損覆核"
          onExpandedChange={damageCasePager.setExpanded}
          onPageChange={damageCasePager.setPage}
          totalItems={damageTabCases.length}
          totalPages={damageCasePager.totalPages}
          visibleEnd={damageCasePager.visibleEnd}
          visibleStart={damageCasePager.visibleStart}
        />
      </section>

      <section className="ops-grid ops-grid--single ops-section" hidden={activeSection !== "operations-alerts" && activeSection !== "work-orders"}>
        <div className="ops-column" id="operations-alerts" hidden={activeSection !== "operations-alerts"}><div className="ops-title"><h2>異常警示</h2><div className="ops-title-actions"><span>共 {alerts.length} 件</span></div></div>
          <div className="damage-review-tabs" role="tablist" aria-label="異常警示分類">
            {alertTabs.map((tab) => <button
              type="button"
              role="tab"
              key={tab.tab}
              className={`damage-review-tab damage-review-tab--${tab.tone}`}
              aria-selected={alertTabView === tab.tab}
              onClick={() => { setAlertTabView(tab.tab); alertPager.reset(); }}
            >
              {tab.label}
              <b>{alerts.filter((item) => alertTab(item) === tab.tab).length}</b>
            </button>)}
          </div>
          <p className="damage-review-hint">{alertTabs.find((tab) => tab.tab === alertTabView)?.hint}</p>
          {alerts.length === 0
            ? <div className="empty-state">目前沒有異常警示<br /><small>髒污或人工確認的新車損會出現在這裡</small></div>
            : alertTabItems.length === 0 && <div className="empty-state">這一區沒有警示</div>}
          {alertPager.visibleItems.map((alert) => {
            const active = !["resolved", "cancelled"].includes(alert.status);
            const hasOrder = orders.some((order) => order.alert_id === alert.id);
            const modelHint = alert.evidence.some((item) => item.score_is_probability === false);
            const facts = alert.evidence
              .filter((item) => item.score_is_probability !== false && !alert.summary.includes(item.description))
              .map((item) => item.description);
            return <article className="alert-card alert-card--compact" key={alert.id}>
              {alert.image_url ? <ZoomableImage
                className="alert-image-link"
                src={authenticatedAssetUrl(alert.image_url)}
                alt={`${alert.vehicle_id} 異常警示證據照片`}
                title={`${alert.vehicle_id} 異常警示證據照片`}
                highlights={cleanlinessHighlights(alert)}
              /> : <div className="alert-image-empty">沒有照片</div>}
              <div className="alert-card__content">
                <div className="alert-card__head">
                  {isUnlocatedCleanliness(alert)
                    ? <span className="severity severity--unsure" title="模型沒有框出位置，請看照片確認">待查</span>
                    : <span className="severity">HIGH</span>}
                  <h3>{alert.title}</h3>
                  <span className="alert-card__status">{statusLabel[alert.status]}</span>
                </div>
                <ul className="alert-facts">
                  <li>{alert.summary}</li>
                  {facts.map((fact) => <li key={fact}>{fact}</li>)}
                  <li className="alert-facts__meta">
                    {alert.vehicle_id} · {alert.rental_id} · {modelHint ? "AI 疑似，待人確認" : `信心 ${Math.round(alert.confidence * 100)}%`}
                    {alert.type === "cleanliness" && alert.review_decision && ` · 已確認${alert.review_decision === "confirmed" ? "有髒污" : "沒有髒污"}`}
                  </li>
                </ul>
              </div>
              <div className="card-actions">
                {alert.type === "cleanliness" && !alert.review_decision && active && <>
                  <button className="danger" disabled={busy === alert.id} onClick={() => void reviewCleanliness(alert.id, "confirmed")}>有髒污</button>
                  <button disabled={busy === alert.id} onClick={() => void reviewCleanliness(alert.id, "dismissed")}>沒髒污</button>
                </>}
                {alert.type === "cleanliness" && alert.review_decision === "confirmed" && active && !hasOrder && <button className="primary" disabled={busy === alert.id} onClick={() => void act(alert.id, () => createOperationsWorkOrder(alert.id), "已建立清潔工單。")}>建立清潔工單</button>}
                {alert.type !== "cleanliness" && active && !hasOrder && <button className="primary" disabled={busy === alert.id} onClick={() => void act(alert.id, () => createOperationsWorkOrder(alert.id), "已建立驗車工單。")}>建立驗車工單</button>}
                {alert.type !== "cleanliness" && alert.status === "open" && <button disabled={busy === alert.id} onClick={() => void act(alert.id, () => acknowledgeAlert(alert.id), "已確認警示。")}>確認警示</button>}
                {alert.type !== "cleanliness" && active && <button className="danger-outline" disabled={busy === alert.id} onClick={() => void act(alert.id, () => cancelAlert(alert.id), "已取消警示。")}>取消警示</button>}
              </div>
            </article>;
          })}
          <PhotoListControls
            currentPage={alertPager.currentPage}
            expanded={alertPager.expanded}
            itemLabel="警示"
            onExpandedChange={alertPager.setExpanded}
            onPageChange={alertPager.setPage}
            totalItems={alertTabItems.length}
            totalPages={alertPager.totalPages}
            visibleEnd={alertPager.visibleEnd}
            visibleStart={alertPager.visibleStart}
          />
        </div>
        <div className="ops-column" id="work-orders" hidden={activeSection !== "work-orders"}><div className="ops-title"><h2>整備工單</h2><div className="ops-title-actions"><span>共 {visibleOrders.length} 件{overdueOrders > 0 && ` · ${overdueOrders} 件超過期限`}</span></div></div>
          <ol className="work-order-guide" aria-label="整備工單操作方式">
            <li><strong>指派</strong>填寫負責人或整備站，按「儲存並指派」。</li>
            <li><strong>開工</strong>人員到車旁後按「開始處理」。</li>
            <li><strong>結案</strong>處理完寫下結果，按「完成並自動結案」，警示會一起關閉。</li>
          </ol>
          <div className="damage-review-tabs" role="tablist" aria-label="整備工單處理進度">
            {workOrderTabs.map((tab) => <button
              type="button"
              role="tab"
              key={tab.status}
              className={`damage-review-tab damage-review-tab--${tab.tone}`}
              aria-selected={orderTab === tab.status}
              onClick={() => { setOrderTab(tab.status); setExpandedOrderId(null); }}
            >
              {tab.label}
              <b>{visibleOrders.filter((item) => item.status === tab.status).length}</b>
            </button>)}
          </div>
          <p className="damage-review-hint">處理期限為建立後 5 個工作天（不含週六、日），超過會標成紅色。點一列展開即可編輯。</p>
          {orderTabItems.length === 0 && <div className="empty-state">{visibleOrders.length === 0 ? "尚未建立工單" : "這一區沒有工單"}</div>}
          <div className="order-list">
            {orderTabItems.map((order) => <WorkOrderRow
              key={order.id}
              order={order}
              busy={busy === order.id}
              expanded={expandedOrderId === order.id}
              canDelete={canDeleteWorkOrders}
              onToggle={() => setExpandedOrderId((current) => current === order.id ? null : order.id)}
              onDelete={(target) => void removeWorkOrder(target)}
              onUpdate={(changes) => void saveWorkOrder(order, changes)}
            />)}
          </div>
        </div>
      </section>

      <section className="notification-panel ops-section" id="notification-outbox" hidden={activeSection !== "notification-outbox"}>
        <div className="ops-title">
          <div>
            <p className="eyebrow">POWER AUTOMATE</p>
            <h2>Email 通知</h2>
          </div>
          <div className="ops-title-actions">
            <span>待送 {pendingNotifications} · 已排入 {handedOffNotifications} · 失敗 {failedNotifications}</span>
            {canManageNotifications && <button
              type="button"
              disabled={busy === "notification-dispatch"}
              onClick={() => setDispatchConfirmOpen(true)}
            >{busy === "notification-dispatch" ? "派送中…" : "派送待處理通知"}</button>}
          </div>
        </div>
        {dispatchConfirmOpen && <div className="model-delete-backdrop" onKeyDown={(event) => { if (event.key === "Escape" && busy !== "notification-dispatch") setDispatchConfirmOpen(false); }}>
          <section className="model-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="notification-dispatch-title">
            <p className="eyebrow">POWER AUTOMATE</p>
            <h2 id="notification-dispatch-title">要寄出 Email 嗎？</h2>
            <p>按下確認，系統就會把信交給 Power Automate 寄出去。</p>
            <dl>
              <div><dt>這次要寄</dt><dd>{dispatchableNotifications.length} 封</dd></div>
              {dispatchRecipients.length > 0 && <div><dt>寄給誰</dt><dd>{dispatchRecipients.join("、")}</dd></div>}
              <div><dt>之前寄的，還在等結果</dt><dd>{handedOffNotifications} 封</dd></div>
            </dl>
            {dispatchableNotifications.length === 0 && <p>現在沒有要寄的信。</p>}
            <div className="model-delete-actions">
              <button type="button" disabled={busy === "notification-dispatch"} onClick={() => setDispatchConfirmOpen(false)}>取消</button>
              <button type="button" className="danger" autoFocus disabled={busy === "notification-dispatch"} onClick={() => void dispatchEmails()}>
                {busy === "notification-dispatch" ? "寄送中…" : dispatchableNotifications.length > 0 ? `確認寄出 ${dispatchableNotifications.length} 封` : "看看寄好了沒"}
              </button>
            </div>
          </section>
        </div>}
        <p className="notification-description">
          系統會將通知寫進 OneDrive 的待寄資料夾；Power Automate 將檔案移到同層的「已寄出／已寄」或「失敗／寄送失敗」後，這裡才更新寄送結果。
        </p>
        {notifications.length === 0
          ? <div className="empty-state">目前沒有 Email 通知紀錄</div>
          : <div className="notification-list">
            {notifications.slice(0, 10).map((item) => <article key={item.id}>
              <span className={`notification-status notification-status--${item.status}`}>
                {notificationStatusLabel[item.status]}
              </span>
              <div>
                <strong>{item.subject}</strong>
                <small>
                  {maskRecipient(item.recipient)}
                  {item.attempts > 1 && ` · 第 ${item.attempts} 次嘗試`}
                </small>
                {item.last_error && <em>{item.last_error}</em>}
              </div>
              <time dateTime={item.handed_off_at ?? item.sent_at ?? item.created_at}>
                {formatTime(item.handed_off_at ?? item.sent_at ?? item.created_at)}
              </time>
              {canManageNotifications && item.status === "failed" && <button
                type="button"
                disabled={busy === item.id}
                onClick={() => void act(item.id, () => retryNotification(item.id), "通知已排入重試。")}
              >重試</button>}
            </article>)}
          </div>}
      </section>

      {showMyActivity && activeSection === "my-activity" && <div className="ops-section">
        <OperationsActivityHistory scope="mine" />
      </div>}
      </main>
    </>
  );
}
