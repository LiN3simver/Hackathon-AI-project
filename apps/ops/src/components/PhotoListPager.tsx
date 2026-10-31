import { useEffect, useMemo, useState } from "react";

const COLLAPSED_ITEM_COUNT = 5;
const PAGE_ITEM_COUNT = 10;

export function usePhotoListPager<T>(
  items: readonly T[],
  pageItemCount = PAGE_ITEM_COUNT,
  initiallyExpanded = false,
  collapsedItemCount = COLLAPSED_ITEM_COUNT,
) {
  const [expanded, setExpanded] = useState(initiallyExpanded);
  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(items.length / pageItemCount));
  const currentPage = Math.min(page, totalPages);

  useEffect(() => {
    if (items.length <= collapsedItemCount) {
      setExpanded(false);
      setPage(1);
      return;
    }
    setPage((current) => Math.min(current, totalPages));
  }, [collapsedItemCount, items.length, totalPages]);

  const visibleItems = useMemo(() => {
    if (!expanded) return items.slice(0, collapsedItemCount);
    const start = (currentPage - 1) * pageItemCount;
    return items.slice(start, start + pageItemCount);
  }, [collapsedItemCount, currentPage, expanded, items, pageItemCount]);

  const visibleStart = visibleItems.length === 0
    ? 0
    : expanded
      ? (currentPage - 1) * pageItemCount + 1
      : 1;
  const visibleEnd = visibleStart === 0 ? 0 : visibleStart + visibleItems.length - 1;

  return {
    collapsedItemCount,
    currentPage,
    expanded,
    reset: () => {
      setExpanded(false);
      setPage(1);
    },
    setExpanded: (next: boolean) => {
      setExpanded(next);
      setPage(1);
    },
    setPage: (next: number) => setPage(Math.max(1, Math.min(next, totalPages))),
    totalPages,
    visibleEnd,
    visibleItems,
    visibleStart,
  };
}

type PhotoListControlsProps = {
  collapsedItemCount?: number;
  currentPage: number;
  expanded: boolean;
  itemLabel: string;
  itemUnit?: string;
  onExpandedChange: (expanded: boolean) => void;
  onPageChange: (page: number) => void;
  totalItems: number;
  totalPages: number;
  visibleEnd: number;
  visibleStart: number;
};

export function PhotoListControls({
  collapsedItemCount = COLLAPSED_ITEM_COUNT,
  currentPage,
  expanded,
  itemLabel,
  itemUnit = "筆",
  onExpandedChange,
  onPageChange,
  totalItems,
  totalPages,
  visibleEnd,
  visibleStart,
}: PhotoListControlsProps) {
  if (totalItems <= collapsedItemCount) return null;

  return <div className="photo-list-controls" aria-label={`${itemLabel}顯示控制`}>
    <button
      className="photo-list-toggle"
      type="button"
      aria-expanded={expanded}
      onClick={() => onExpandedChange(!expanded)}
    >{expanded
        ? `收合，只顯示前 ${collapsedItemCount} ${itemUnit}`
        : `展開查看其餘 ${totalItems - collapsedItemCount} ${itemUnit}${itemLabel}`}</button>
    {expanded && <div className="photo-list-pagination">
      <span>顯示 {visibleStart}–{visibleEnd}／共 {totalItems} {itemUnit}</span>
      {totalPages > 1 && <>
        <button
          type="button"
          disabled={currentPage === 1}
          onClick={() => onPageChange(currentPage - 1)}
        >上一頁</button>
        <strong>第 {currentPage}／{totalPages} 頁</strong>
        <button
          type="button"
          disabled={currentPage === totalPages}
          onClick={() => onPageChange(currentPage + 1)}
        >下一頁</button>
      </>}
    </div>}
  </div>;
}
