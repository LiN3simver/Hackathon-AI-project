// Optional catalog for vehicle metadata entry; custom makes and models are supported.
export const vehicleModels: Record<string, string[]> = {
  Toyota: ["Yaris", "Vios", "Corolla Altis", "Corolla Cross", "RAV4", "Prius c"],
  Honda: ["Fit"],
};

export const seatMaterials: Record<string, string> = {
  fabric: "布椅", leather: "真皮", synthetic_leather: "合成皮／皮革", mixed: "皮布混搭", suede: "麂皮／類麂皮",
};
export const seatStyles: Record<string, string> = {
  standard: "一般獨立座椅", sport: "運動型／桶型座椅", bench: "連排座椅", captain: "獨立休旅／商務座椅",
};
export const seatPatterns: Record<string, string> = {
  plain: "素面", striped: "條紋", diamond: "菱格紋", perforated: "打孔", two_tone: "雙色拼接", patterned: "其他花紋",
};

export function seatDescription(value: string | undefined, options: Record<string, string>): string {
  return value ? options[value] ?? value : "";
}
