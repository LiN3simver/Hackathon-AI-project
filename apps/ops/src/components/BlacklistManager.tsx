import { useEffect, useState } from "react";
import { getHistoryUsers, getUserHistory, type UserHistoryListItem, type UserHistoryProfile } from "../lib/api";
import { HistoryRecordManager } from "./HistoryRecordManager";

/** Matches BLACKLIST_OFFENCES in apps/api/app/services/history_records.py. */
export const BLACKLIST_OFFENCES = 2;

export function offences(user: UserHistoryListItem): number {
  return user.caused_damage_count + user.litter_count + (user.dirty_count ?? 0);
}

/** Blacklist entry point: find a renter, see their confirmed offences, list or unlist them. */
export function BlacklistManager() {
  const [search, setSearch] = useState("");
  const [onlyListed, setOnlyListed] = useState(false);
  const [users, setUsers] = useState<UserHistoryListItem[]>([]);
  const [profile, setProfile] = useState<UserHistoryProfile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    let current = true;
    const timer = window.setTimeout(() => {
      getHistoryUsers(search).then((rows) => { if (current) setUsers(rows); })
        .catch((caught: unknown) => { if (current) setError(caught instanceof Error ? caught.message : "使用者讀取失敗"); });
    }, 250);
    return () => { current = false; window.clearTimeout(timer); };
  }, [search, refresh]);

  async function open(userId: string) {
    setError(null);
    try { setProfile(await getUserHistory(userId)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "使用者讀取失敗"); }
  }

  const shown = onlyListed ? users.filter((user) => user.blacklisted) : users;
  return <section className="rag-seed-manager blacklist-manager">
    <div className="settings-group-heading"><p className="eyebrow">BLACKLIST</p><h2>黑名單管理</h2>
      <p>同一位使用者已確認的車損、留下垃圾與髒污合計達 {BLACKLIST_OFFENCES} 次時，系統會自動列入黑名單；待確認或爭議中的紀錄不計入。
        自動規則只會列入、不會移出，人工移出後也不會再被自動列入。借還車AI助理的黑名單問題以這裡的狀態回答。</p>
    </div>
    {error && <p role="alert" className="rag-seed-error">{error}</p>}
    <div className="blacklist-manager__filters">
      <label><span>搜尋使用者</span><input type="search" maxLength={80} value={search} placeholder="姓名或使用者編號"
        onChange={(event) => setSearch(event.target.value)} /></label>
      <label className="rag-seed-check"><input type="checkbox" checked={onlyListed} onChange={(event) => setOnlyListed(event.target.checked)} />只看黑名單</label>
    </div>
    <div className="blacklist-manager__users">
      {shown.map((user) => <button type="button" key={user.user_id} aria-pressed={profile?.user_id === user.user_id}
        onClick={() => void open(user.user_id)}>
        <strong>{user.display_name}（{user.user_id}）</strong>
        <span>車損 {user.caused_damage_count} 次 · 垃圾 {user.litter_count} 次 · 髒污 {user.dirty_count ?? 0} 次 · 合計 {offences(user)}／{BLACKLIST_OFFENCES}</span>
        {user.blacklisted && <em>黑名單</em>}
      </button>)}
      {!shown.length && <p>{onlyListed ? "目前沒有使用者列入黑名單。" : "找不到符合的使用者。"}</p>}
    </div>
    {profile && <HistoryRecordManager key={`${profile.user_id}-${profile.blacklisted}`} profile={profile} defaultOpen
      onChange={async () => { await open(profile.user_id); setRefresh((value) => value + 1); }} />}
  </section>;
}
