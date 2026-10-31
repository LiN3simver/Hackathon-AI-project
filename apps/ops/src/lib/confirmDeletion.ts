/** One deliberate confirmation before a destructive action. Account and model deletion use their own two-step dialogs. */
export function confirmDeletion(description: string): boolean {
  return window.confirm(description);
}
