import { useEffect, useId, useLayoutEffect, useRef, useState, type FormEvent } from "react";
import { getHistoryUserSuggestions, type HistoryChatContext, type HistoryModelOption, type HistoryUserSuggestion, type HistoryQueryResponse, type HistoryScopeOption } from "../lib/api";
import { HistoryAnswerView } from "./HistoryAnswerView";
import { RagSeedPhotos } from "./RagSeedPhotos";

export type ChatMessage = {
  id: number; role: "assistant" | "user"; text: string; result?: HistoryQueryResponse;
  streaming?: boolean; queued?: boolean; stopped?: boolean; feedback?: 1 | -1;
};

const TIER_LABELS: Record<HistoryModelOption["tier"], string> = { local: "本機", paid: "付費", gpu: "GPU" };

/** 「Model ∨」 beside the send button: which text model phrases the summary. The answer card never depends on it. */
function ModelPicker({ models, value, onChange, disabled, below }: {
  models: HistoryModelOption[]; value: string | null; onChange: (id: string) => void; disabled: boolean; below: boolean;
}) {
  const [open, setOpen] = useState(false);
  const menuId = useId();
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", close); document.removeEventListener("keydown", escape); };
  }, [open]);
  const current = models.find((item) => item.id === value) ?? models[0];
  return <div className="history-chat__model" ref={root}>
    <button type="button" className="history-chat__model-toggle" aria-haspopup="menu" aria-expanded={open} aria-controls={menuId}
      title={`回答模型：${current.label}`} disabled={disabled} onClick={() => setOpen((state) => !state)}>
      <span>Model</span>
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6" /></svg>
    </button>
    {open && <div id={menuId} role="menu" aria-label="選擇回答模型" className={`history-chat__model-menu${below ? " history-chat__model-menu--below" : ""}`}>
      {models.map((item) => <button type="button" role="menuitemradio" aria-checked={item.id === current.id} key={item.id}
        disabled={!item.available && item.id !== current.id} onClick={() => { onChange(item.id); setOpen(false); }}>
        <span className="history-chat__model-name">{item.label}<small>{TIER_LABELS[item.tier]}</small></span>
        {(item.description || !item.available) && <span className="history-chat__model-note">
          {item.available ? item.description : "尚未就緒"}</span>}
      </button>)}
    </div>}
  </div>;
}

export function HistoryConversation({ messages, question, onQuestionChange, onSubmit, busy,
  quickQuestions, onSelectScope, onAsk, onStop, onFeedback, models, modelId = null, onModelChange, placeholder = "輸入問題…" }: {
  messages: ChatMessage[]; question: string; onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void; busy: boolean;
  quickQuestions: string[]; placeholder?: string;
  onSelectScope?: (question: string, option: HistoryScopeOption) => void;
  /** A capsule or follow-up keeps the context of the answer it belongs to, not the latest one. */
  onAsk?: (question: string, context: HistoryChatContext | null) => void;
  onStop?: () => void;
  onFeedback?: (message: ChatMessage, rating: 1 | -1, expected: string) => Promise<void>;
  /** Text models for the summary; without them the composer shows no picker. */
  models?: HistoryModelOption[]; modelId?: string | null; onModelChange?: (id: string) => void;
}) {
  const showModels = !!models?.length && !!onModelChange;
  const waitingForCard = busy && messages.at(-1)?.role !== "assistant";
  const log = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (!log.current) return;
    const panel = log.current;
    const scroll = () => { panel.scrollTop = panel.scrollHeight; };
    scroll();
    const observer = new ResizeObserver(scroll);
    observer.observe(panel);
    return () => observer.disconnect();
  }, [messages, busy]);
  const input = useRef<HTMLTextAreaElement>(null);
  const suggestionId = useId();
  const examplesId = useId();
  const [showExamples, setShowExamples] = useState(false);
  const [caret, setCaret] = useState(0);
  const [focused, setFocused] = useState(false);
  const [composing, setComposing] = useState(false);
  const [suggestions, setSuggestions] = useState<HistoryUserSuggestion[]>([]);
  const [activeSuggestion, setActiveSuggestion] = useState(-1);
  const fragment = question.slice(0, caret).match(/(?:^|[^A-Za-z0-9_-])([A-Za-z0-9][A-Za-z0-9_-]*)$/)?.[1] ?? "";
  const tokenStart = caret - fragment.length;
  const tokenEnd = caret + (question.slice(caret).match(/^[A-Za-z0-9_-]*/)?.[0].length ?? 0);
  const showSuggestions = focused && !composing && !busy && suggestions.length > 0;

  useEffect(() => {
    setSuggestions([]);
    setActiveSuggestion(-1);
    if (!focused || composing || busy || fragment.length < 1 || fragment.length > 80) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void getHistoryUserSuggestions(fragment, controller.signal).then((items) => {
        if (!controller.signal.aborted) setSuggestions(items);
      }).catch(() => { /* Suggestions are optional; typing and sending stay available. */ });
    }, 200);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [fragment, focused, composing, busy, question, caret]);

  useLayoutEffect(() => {
    if (activeSuggestion >= 0) document.getElementById(`${suggestionId}-${activeSuggestion}`)?.scrollIntoView({ block: "nearest" });
  }, [activeSuggestion, suggestionId]);

  function focusAt(position: number) {
    requestAnimationFrame(() => {
      input.current?.focus();
      input.current?.setSelectionRange(position, position);
      setCaret(position);
    });
  }

  function chooseSuggestion(item: HistoryUserSuggestion) {
    const suffix = question.slice(tokenEnd);
    const existingSpace = /^\s/.test(suffix);
    const replacement = item.user_id + (existingSpace ? "" : " ");
    const nextQuestion = question.slice(0, tokenStart) + replacement + suffix;
    if (nextQuestion.length > 500) return;
    setSuggestions([]);
    setActiveSuggestion(-1);
    onQuestionChange(nextQuestion);
    focusAt(tokenStart + replacement.length + (existingSpace ? 1 : 0));
  }

  function editExample(value: string) {
    setShowExamples(false);
    onQuestionChange(value);
    setSuggestions([]);
    focusAt(value.length);
  }
  useLayoutEffect(() => {
    if (!input.current) return;
    const field = input.current;
    function resize() {
      field.style.height = "44px";
      if (question) field.style.height = `${Math.min(144, Math.max(44, field.scrollHeight + 2))}px`;
    }
    resize();
    let width = field.clientWidth;
    let resizeFrame = 0;
    const observer = new ResizeObserver(() => {
      if (field.clientWidth !== width) {
        width = field.clientWidth;
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(resize);
      }
    });
    observer.observe(field);
    return () => { observer.disconnect(); cancelAnimationFrame(resizeFrame); };
  }, [question]);

  return <>
    {(messages.length > 0 || busy) && <div ref={log} className="history-chat__messages" role="log" aria-live="polite">
      {messages.map((message) => <div className={`history-chat__message history-chat__message--${message.role}`} key={message.id}>
        {message.role === "assistant" && <span className="history-chat__bot" aria-hidden="true">AI</span>}
        {message.result?.answer_card && message.result.answer_card.intent !== "clarify"
          ? <div className="history-chat__message-content history-chat__message-content--card">
            <HistoryAnswerView result={message.result} busy={busy} streaming={message.streaming} queued={message.queued}
              stopped={message.stopped} feedback={message.feedback}
              onAsk={onAsk && ((value) => onAsk(value, message.result?.context ?? null))}
              summary={message.streaming || message.result.generated_by === "ollama_summary" ? message.text : ""}
              onFeedback={onFeedback && ((rating, expected) => onFeedback(message, rating, expected))} />
          </div>
          : <div className="history-chat__message-content"><p>{message.text}</p>
          {!!message.result?.clarification_options?.length && <div className="history-chat__choices" aria-label="選擇查詢對象">
            {message.result.clarification_options.map((option) => <button type="button" key={`${option.scope}:${option.value}`}
              disabled={busy} onClick={() => onSelectScope?.(message.result!.question, option)}>{option.label}</button>)}
          </div>}
          {message.result?.generated_by === "template" && message.result.warnings?.some((warning) => warning.startsWith("ollama_") || warning === "llm_generation_disabled") && <small role="status">AI 整理暫時無法使用，目前顯示直接查詢資料；可稍後重試。</small>}
          {message.result && message.result.matched_events.length > 0 && <details className="history-chat__evidence">
            <summary>查看來源紀錄（{message.result.matched_events.length} 筆）</summary>
            {message.result.matched_events.map((event) => <div key={event.id}>
              <strong>{event.occurred_on} · {event.title}{event.disputed ? "（爭議中）" : ""}</strong>
              <div>{event.rental_id}｜{event.summary}</div><RagSeedPhotos event={event} />
            </div>)}
          </details>}
        </div>}
      </div>)}
      {waitingForCard && <div className="history-chat__message history-chat__message--assistant">
        <span className="history-chat__bot" aria-hidden="true">AI</span><p className="history-chat__typing">AI 正在查詢並整理紀錄…</p>
      </div>}
    </div>}
    <div className="history-chat__input-area">
    {showSuggestions && <div id={suggestionId} role="listbox" aria-label="使用者編號建議"
      className={`history-chat__suggestions${messages.length ? " history-chat__suggestions--above" : ""}`}>
      {suggestions.map((item, index) => <button type="button" role="option" tabIndex={-1}
        aria-selected={activeSuggestion === index} id={`${suggestionId}-${index}`} key={item.user_id}
        onPointerDown={(event) => event.preventDefault()} onClick={() => chooseSuggestion(item)}>
        <span className="history-chat__suggestion-field"><small>編號</small>{item.user_id}</span>
        <span className="history-chat__suggestion-field"><small>姓名</small>{item.display_name}</span>
      </button>)}
    </div>}
    <form className={`history-chat__composer${showModels ? " history-chat__composer--models" : ""}`} onSubmit={onSubmit}>
      <button type="button" className="history-chat__examples-toggle" aria-label={showExamples ? "收合範例問題" : "展開範例問題"}
        aria-expanded={showExamples} aria-controls={examplesId} onClick={() => setShowExamples((value) => !value)}>
        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg>
      </button>
      <textarea ref={input} aria-label="輸入問題" rows={1} maxLength={500} value={question}
        aria-autocomplete="list" aria-controls={showSuggestions ? suggestionId : undefined}
        aria-activedescendant={showSuggestions && activeSuggestion >= 0 ? `${suggestionId}-${activeSuggestion}` : undefined}
        onChange={(event) => { onQuestionChange(event.target.value); setCaret(event.target.selectionStart); }}
        onSelect={(event) => setCaret(event.currentTarget.selectionStart)}
        onClick={(event) => setCaret(event.currentTarget.selectionStart)}
        onKeyUp={(event) => setCaret(event.currentTarget.selectionStart)}
        onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
        onCompositionStart={() => setComposing(true)} onCompositionEnd={(event) => { setComposing(false); setCaret(event.currentTarget.selectionStart); }}
        placeholder={placeholder}
        onKeyDown={(event) => {
          if (event.nativeEvent.isComposing || composing || event.nativeEvent.keyCode === 229) return;
          if (showSuggestions && !event.shiftKey) {
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              setActiveSuggestion((index) => event.key === "ArrowDown"
                ? (index + 1) % suggestions.length : (index <= 0 ? suggestions.length - 1 : index - 1));
              return;
            }
            if (event.key === "Escape") { event.preventDefault(); setSuggestions([]); return; }
            if ((event.key === "Enter" || event.key === "Tab") && activeSuggestion >= 0) {
              event.preventDefault(); chooseSuggestion(suggestions[activeSuggestion]); return;
            }
          }
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            if (!busy && question.trim()) event.currentTarget.form?.requestSubmit();
          }
        }} />
      {showModels && <ModelPicker models={models!} value={modelId} onChange={onModelChange!} disabled={busy} below={!messages.length} />}
      {busy && onStop ? <button type="button" className="history-chat__stop" aria-label="停止產生" title="停止產生" onClick={onStop}>
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" /></svg>
      </button> : <button type="submit" aria-label={busy ? "正在查詢" : "送出問題"} title={busy ? "正在查詢" : "送出問題"}
        disabled={busy || question.trim().length < 2}>
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7" /></svg>
      </button>}
    </form>
    </div>
    <div id={examplesId} className="history-chat__quick-questions" aria-label="範例問題" hidden={!showExamples}>
      {quickQuestions.map((item) => <button type="button" onClick={() => editExample(item)} disabled={busy} key={item}>{item}</button>)}
    </div>
  </>;
}
