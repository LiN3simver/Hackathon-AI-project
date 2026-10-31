import { useEffect, useRef, useState, type FormEvent } from "react";
import { HistoryConversation, type ChatMessage } from "./HistoryConversation";
import {
  getHistoryExamples, getHistoryModels, getStoredOpsSession, sendHistoryFeedback, streamAssistantHistory, type HistoryChatContext,
  type HistoryModelList, type HistoryQueryResponse, type HistoryScopeOption,
} from "../lib/api";

// Shown until the server's live list (with real names and plates) arrives, or if it cannot be loaded.
const fallbackExamples = ["你可以做什麼？", "這個月有哪些使用者造成車損？", "上個月有哪些使用者造成車損？", "去年有哪些使用者造成車損？",
  "有爭議的車損有哪些？", "有哪些車損還沒賠？", "全部使用者總共還欠多少錢？", "哪些使用者曾留下垃圾？", "這個月的垃圾紀錄",
  "誰最需要注意？", "不良紀錄最多的是誰？", "目前有哪些人列入黑名單？", "全部使用者的租借紀錄"];
const MODEL_KEY = "guardian-assistant-model-v1";
type Conversation = { id: string; title: string; titleEdited?: boolean; messages: ChatMessage[]; context: HistoryChatContext | null };
type ChatHistory = { conversations: Conversation[]; activeId: string | null };

function readHistory(key: string): ChatHistory {
  try {
    const saved = JSON.parse(window.localStorage.getItem(key) ?? "null");
    if (saved && Array.isArray(saved.conversations)) {
      const conversations = (saved.conversations.filter((item: Conversation) => item && typeof item.id === "string"
        && typeof item.title === "string" && Array.isArray(item.messages)
        && item.messages.every((message) => message && typeof message.id === "number" && typeof message.text === "string"
          && (message.role === "user" || message.role === "assistant"))) as Conversation[])
        // A reload mid-answer leaves a half-streamed message; show it as stopped rather than loading forever.
        .map((item) => ({ ...item, messages: item.messages.map((message) => message.streaming
          ? { ...message, streaming: false, queued: false, stopped: true } : message) }));
      return { conversations, activeId: conversations.some((item) => item.id === saved.activeId) ? saved.activeId : null };
    }
  } catch { /* A missing or unreadable cache starts a fresh conversation. */ }
  return { conversations: [], activeId: null };
}

/** The answer card already lists its sources, so the raw event dump is not kept in browser storage. */
function compact(result: HistoryQueryResponse): HistoryQueryResponse {
  return result.answer_card ? { ...result, matched_events: [], agent_steps: undefined } : result;
}

function turnText(message: ChatMessage): string {
  return (message.text || message.result?.answer_card?.headline || message.result?.answer || "").slice(0, 600);
}

export function UserHistoryExplorer({ isActive }: { canManage?: boolean; isActive?: boolean }) {
  const [examples, setExamples] = useState(fallbackExamples);
  const [storageKey] = useState(() => `guardian-assistant-history-v1:${getStoredOpsSession()?.principal.subject ?? "guest"}`);
  const [history, setHistory] = useState<ChatHistory>(() => readHistory(storageKey));
  const [question, setQuestion] = useState("");
  const [pendingIds, setPendingIds] = useState<string[]>([]);
  const pending = useRef(new Set<string>());
  const controllers = useRef(new Map<string, AbortController>());
  const [editingId, setEditingId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [storageError, setStorageError] = useState(false);
  const [models, setModels] = useState<HistoryModelList | null>(null);
  const [modelId, setModelId] = useState(() => { try { return window.localStorage.getItem(MODEL_KEY); } catch { return null; } });
  // A saved choice the server no longer offers quietly returns to the default model.
  const chosenModel = models?.options.some((item) => item.id === modelId) ? modelId! : models?.default_id ?? null;
  const active = history.conversations.find((item) => item.id === history.activeId);
  const busy = !!active && pendingIds.includes(active.id);

  useEffect(() => {
    try { window.localStorage.setItem(storageKey, JSON.stringify(history)); setStorageError(false); }
    catch { setStorageError(true); }
  }, [history, storageKey]);
  useEffect(() => () => { controllers.current.forEach((controller) => controller.abort()); }, []);
  // Reload on every visit so the names and plates follow newly added records.
  useEffect(() => {
    if (isActive === false) return;
    let current = true;
    getHistoryExamples().then((items) => { if (current && items.length) setExamples(items); }).catch(() => undefined);
    getHistoryModels().then((list) => { if (current && list.options.length) setModels(list); }).catch(() => undefined);
    return () => { current = false; };
  }, [isActive]);

  function chooseModel(id: string) {
    setModelId(id);
    try { window.localStorage.setItem(MODEL_KEY, id); } catch { /* The choice then lasts until reload. */ }
  }

  function updateConversation(id: string, change: (item: Conversation) => Conversation) {
    setHistory((current) => ({ ...current, conversations: current.conversations.map((item) => item.id === id ? change(item) : item) }));
  }

  function updateAnswer(id: string, messageId: number, change: (message: ChatMessage) => ChatMessage) {
    updateConversation(id, (item) => ({ ...item, messages: item.messages.some((message) => message.id === messageId)
      ? item.messages.map((message) => message.id === messageId ? change(message) : message)
      : [...item.messages, change({ id: messageId, role: "assistant", text: "" })] }));
  }

  async function ask(value: string, selection?: HistoryScopeOption, contextOverride?: HistoryChatContext | null) {
    const text = value.trim();
    if (text.length < 2) return;
    const id = active?.id ?? crypto.randomUUID();
    if (pending.current.has(id)) return;
    pending.current.add(id);
    setPendingIds((current) => [...current, id]);
    setQuestion("");
    const message: ChatMessage = { id: (active?.messages.at(-1)?.id ?? 0) + 1, role: "user", text: selection ? `${text}（${selection.label}）` : text };
    const answerId = message.id + 1;
    setHistory((current) => ({ activeId: id, conversations: active
      ? current.conversations.map((item) => item.id === id ? { ...item, messages: [...item.messages, message] } : item)
      : [{ id, title: "新對話", messages: [message], context: null }, ...current.conversations] }));
    const controller = new AbortController();
    controllers.current.set(id, controller);
    const adopt = (response: HistoryQueryResponse) => updateConversation(id, (item) => ({ ...item,
      title: !item.titleEdited && item.title === "新對話" && response.conversation_title ? response.conversation_title : item.title,
      context: response.context ?? item.context }));
    try {
      const turns = (active?.messages ?? []).slice(-4).map((item) => ({ role: item.role, content: turnText(item) || "…" }));
      const context = contextOverride !== undefined ? contextOverride : active?.context ?? null;
      const response = await streamAssistantHistory(text, context, {
        selection, messages: turns, signal: controller.signal,
        modelId: chosenModel && chosenModel !== models?.default_id ? chosenModel : null,
        onEvent: (event) => {
          if (event.type === "card") {
            updateAnswer(id, answerId, (answer) => ({ ...answer, result: compact(event.data), text: "", streaming: true }));
            adopt(event.data);
          } else if (event.type === "status") {
            updateAnswer(id, answerId, (answer) => ({ ...answer, queued: true }));
          } else if (event.type === "delta") {
            updateAnswer(id, answerId, (answer) => ({ ...answer, text: answer.text + event.data.text, queued: false }));
          }
        },
      });
      const cardOnly = !!response.answer_card && response.answer_card.intent !== "clarify";
      updateAnswer(id, answerId, (answer) => ({ ...answer, result: compact(response), streaming: false, queued: false,
        text: cardOnly ? (response.generated_by === "ollama_summary" ? response.answer : "") : response.answer }));
      adopt(response);
    } catch {
      updateAnswer(id, answerId, (answer) => controller.signal.aborted && answer.result
        ? { ...answer, streaming: false, queued: false, stopped: true }
        : { ...answer, streaming: false, queued: false, text: answer.result ? answer.text : "查詢暫時無法完成，請再試一次。" });
    } finally {
      controllers.current.delete(id);
      pending.current.delete(id);
      setPendingIds((current) => current.filter((item) => item !== id));
    }
  }

  async function feedback(conversationId: string, message: ChatMessage, rating: 1 | -1, expected: string) {
    const conversation = history.conversations.find((item) => item.id === conversationId);
    const before = conversation?.messages.slice(0, conversation.messages.findIndex((item) => item.id === message.id)) ?? [];
    const prior = [...before].reverse().find((item) => item.role === "assistant" && item.result?.context)?.result?.context ?? null;
    const result = message.result;
    await sendHistoryFeedback({
      question: (result?.question || [...before].reverse().find((item) => item.role === "user")?.text || "").slice(0, 500),
      answer: [result?.answer_card?.headline, message.text].filter(Boolean).join("\n").slice(0, 8000),
      rating, expected, context: prior, intent: result?.answer_card?.intent ?? null, answer_id: result?.answer_id ?? null,
      generated_by: result?.generated_by ?? null, warnings: (result?.warnings ?? []).slice(0, 20),
      cited_event_ids: (result?.answer_card?.rows ?? []).filter((row) => row.cited && row.event_id).map((row) => row.event_id!).slice(0, 50),
    });
    updateConversation(conversationId, (item) => ({ ...item, messages: item.messages.map((entry) => entry.id === message.id ? { ...entry, feedback: rating } : entry) }));
  }

  function selectConversation(id: string | null) {
    setHistory((current) => ({ ...current, activeId: id }));
    setQuestion("");
    setEditingId(null);
  }

  function rename(event: FormEvent, id: string) {
    event.preventDefault();
    if (!title.trim()) return;
    setHistory((current) => ({ ...current, conversations: current.conversations.map((item) => item.id === id ? { ...item, title: title.trim(), titleEdited: true } : item) }));
    setEditingId(null);
  }

  function deleteConversation(id: string) {
    controllers.current.get(id)?.abort();
    setHistory((current) => {
      const conversations = current.conversations.filter((item) => item.id !== id);
      return { conversations, activeId: current.activeId === id ? conversations[0]?.id ?? null : current.activeId };
    });
    if (history.activeId === id) setQuestion("");
    if (editingId === id) setEditingId(null);
  }

  return <section className="history-explorer history-chat history-chat--minimal" id="user-history" aria-label="借還車AI助理">
    <h2 className="history-chat__heading">借還車AI助理</h2>
    <div className="history-chat__workspace">
      <aside className="history-chat__sidebar" aria-label="對話紀錄">
        <button type="button" className="history-chat__new" onClick={() => selectConversation(null)}><svg aria-hidden="true" width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg><span>新對話</span></button>
        <h3>對話紀錄</h3>
        <div className="history-chat__session-list">
          {history.conversations.map((item) => <div className={`history-chat__session${item.id === history.activeId ? " is-active" : ""}`} key={item.id}>
            {editingId === item.id ? <form className="history-chat__rename" onSubmit={(event) => rename(event, item.id)}>
              <input autoFocus aria-label="對話名稱" maxLength={80} value={title} onChange={(event) => setTitle(event.target.value)} onKeyDown={(event) => { if (event.key === "Escape") setEditingId(null); }} />
              <button type="submit" disabled={!title.trim()}>儲存</button><button type="button" onClick={() => setEditingId(null)}>取消</button>
            </form> : <>
              <button type="button" className="history-chat__session-title" aria-current={item.id === history.activeId ? "true" : undefined} title={item.title} onClick={() => selectConversation(item.id)}>{item.title}</button>
              <div className="history-chat__session-actions">
                <button type="button" aria-label={`重新命名：${item.title}`} title="重新命名" onClick={() => { setEditingId(item.id); setTitle(item.title); }}>✎</button>
                <button type="button" aria-label={`刪除對話：${item.title}`} title="刪除對話" onClick={() => deleteConversation(item.id)}>
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7" /></svg>
                </button>
              </div>
            </>}
          </div>)}
          {!history.conversations.length && <p className="history-chat__empty-history">送出問題後，對話會自動儲存於此。</p>}
        </div>
        <small className="history-chat__storage-note">對話儲存在此瀏覽器</small>
      </aside>
      <div className={`history-chat__panel${active ? "" : " history-chat__panel--empty"}`}>
        {!active && <p className="history-chat__welcome">我們該從哪裡開始？</p>}
        <HistoryConversation key={active?.id ?? "new"} messages={active?.messages ?? []} question={question} onQuestionChange={setQuestion}
          onSubmit={(event) => { event.preventDefault(); void ask(question); }} busy={busy}
          onSelectScope={(value, choice) => void ask(value, choice)} quickQuestions={examples}
          onAsk={(value, context) => void ask(value, undefined, context)} onStop={active ? () => controllers.current.get(active.id)?.abort() : undefined}
          onFeedback={active ? (message, rating, expected) => feedback(active.id, message, rating, expected) : undefined}
          models={models?.options} modelId={chosenModel} onModelChange={chooseModel}
          placeholder="輸入姓名、車牌或問題…" />
        {storageError && <p role="status" className="history-chat__storage-note">瀏覽器儲存空間無法使用，目前對話將無法在重新整理後保留。</p>}
      </div>
    </div>
  </section>;
}
