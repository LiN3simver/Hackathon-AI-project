export function isHistoryPhoto(file: File): boolean {
  return /\.(jpe?g|png|webp)$/i.test(file.name);
}

export function mergeHistoryPhotos(current: File[], incoming: File[]): File[] {
  const identity = (file: File) => JSON.stringify([file.webkitRelativePath || file.name, file.size, file.lastModified]);
  const existing = new Set(current.map(identity));
  return [...current, ...incoming.filter((file) => {
    const key = identity(file);
    if (existing.has(key)) return false;
    existing.add(key);
    return true;
  })];
}
