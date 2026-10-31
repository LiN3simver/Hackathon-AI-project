import { useState } from "react";
import type { LearningVehicleIdentity } from "../lib/api";
import { seatMaterials, seatPatterns, seatStyles, vehicleModels } from "../lib/vehicleCatalog";

function DetailChoice({ title, value, options, onChange }: {
  title: string; value: string; options: Record<string, string>; onChange: (value: string) => void;
}) {
  const [custom, setCustom] = useState(Boolean(value && !(value in options)));
  const isCustom = custom || Boolean(value && !(value in options));
  return <label>
    <span>{title}</span>
    <select value={isCustom ? "custom" : value} onChange={(event) => {
      setCustom(event.target.value === "custom");
      onChange(event.target.value === "custom" ? "" : event.target.value);
    }}>
      <option value="">尚未確認（選填）</option>
      {Object.entries(options).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      <option value="custom">自訂款式</option>
    </select>
    {isCustom && <input aria-label={`自訂${title}`} maxLength={80} value={value}
      placeholder={`輸入${title}`} onChange={(event) => onChange(event.target.value)} />}
  </label>;
}

export function VehicleIdentityEditor({ value, interior, onChange }: {
  value?: LearningVehicleIdentity; interior: boolean; onChange: (value: LearningVehicleIdentity) => void;
}) {
  const identity = value ?? { make: "", vehicle_type: null };
  const [customMake, setCustomMake] = useState(Boolean(identity.make && !(identity.make in vehicleModels)));
  const isCustom = customMake || Boolean(identity.make && !(identity.make in vehicleModels));
  const [customModel, setCustomModel] = useState(false);
  const knownModels = vehicleModels[identity.make] ?? [];
  const isCustomModel = customModel || Boolean(identity.model && !knownModels.includes(identity.model));
  function update(patch: Partial<LearningVehicleIdentity>) { onChange({ ...identity, ...patch }); }
  return <fieldset className="vehicle-identity-fields">
    <legend>{interior ? "車款與座椅款式（選填）" : "品牌與實際車款（選填）"}</legend>
    <label>
      <span>車輛品牌</span>
      <select value={isCustom ? "custom" : identity.make} onChange={(event) => {
        const choice = event.target.value;
        setCustomMake(choice === "custom");
        setCustomModel(false);
        update({ make: choice === "custom" ? "" : choice, model: "", generation: "" });
      }}>
        <option value="">請選擇（選填）</option>
        {Object.keys(vehicleModels).map((make) => <option key={make} value={make}>{make}</option>)}
        <option value="custom">其他品牌／自訂</option>
      </select>
      {isCustom && <input maxLength={80} placeholder="輸入其他品牌" aria-label="自訂車輛品牌"
        value={identity.make} onChange={(event) => update({ make: event.target.value })} />}
    </label>
    <label>
      <span>實際車款</span>
      <select disabled={!identity.make.trim()} value={isCustomModel ? "custom" : identity.model ?? ""}
        onChange={(event) => {
          setCustomModel(event.target.value === "custom");
          update({ model: event.target.value === "custom" ? "" : event.target.value });
        }}>
        <option value="">請選擇（選填）</option>
        {knownModels.map((model) => <option key={model} value={model}>{model}</option>)}
        <option value="custom">自訂款式</option>
      </select>
      {isCustomModel && <input aria-label="自訂實際車款" maxLength={80} value={identity.model ?? ""}
        placeholder="輸入資料集以外的車款" onChange={(event) => update({ model: event.target.value })} />}
    </label>
    <label>
      <span>車型類別</span>
      <select value={identity.vehicle_type ?? ""} onChange={(event) => update({
        vehicle_type: (event.target.value || null) as LearningVehicleIdentity["vehicle_type"],
      })}>
        <option value="">請選擇（選填）</option>
        <option value="passenger_car">小客車</option>
        <option value="suv">休旅車</option>
        <option value="van">箱型車／廂型車</option>
      </select>
    </label>
    <label>
      <span>世代／年式／外觀版本</span>
      <input maxLength={80} value={identity.generation ?? ""} placeholder="例如第 3 代、2020 年式；不確定可留空"
        onChange={(event) => update({ generation: event.target.value })} />
    </label>
    {interior && <>
      <DetailChoice title="座椅材質" value={identity.seat_material ?? ""} options={seatMaterials}
        onChange={(seat_material) => update({ seat_material })} />
      <DetailChoice title="座椅款式" value={identity.seat_style ?? ""} options={seatStyles}
        onChange={(seat_style) => update({ seat_style })} />
      <DetailChoice title="座椅紋路" value={identity.seat_pattern ?? ""} options={seatPatterns}
        onChange={(seat_pattern) => update({ seat_pattern })} />
    </>}
    <small className="vehicle-identity-help">車款選項依提供的訓練／測試照片整理；其餘請選自訂。無法確認可留空。車款、版本{interior ? "與座椅款式" : ""}會隨人工真值保存。</small>
  </fieldset>;
}
