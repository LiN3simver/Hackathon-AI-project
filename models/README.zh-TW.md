# 模型產物

[English](README.md) | 繁體中文

Guardian 將模型整合與模型產物分開。作品集展示視覺任務如何進入應用程式流程；訓練完成的模型檢查點、特徵快取、私人評估報告與資料集衍生產物則須另行提供。

## 可檢視的實作

| 原始碼 | 展示內容 |
| --- | --- |
| [推論協調](../apps/api/app/services/inspection_analysis.py) | 任務派送、失敗處理與審核分流 |
| [服務介接器](../apps/api/app/services/) | 將個別模型整合為結構化的應用程式結果 |
| [訓練腳本](../scripts/) | 資料集準備、候選模型訓練與評估工具 |

## 本機配置

`download_model_weights.py` 可選擇下載上游 DINOv2 與 Torchvision 骨幹模型，但不會重建任務專屬分類器。請檢視上游授權條款，並僅在可信任的環境中執行下載。

請透過 `GUARDIAN_VIEW_MODEL_PATH` 提供有權使用的車外視角模型檢查點。選用的專用模型會使用各服務介接器定義的路徑。啟用推論或模型升版前，請參閱[模型流程](../docs/model-pipeline.zh-TW.md)。缺少的資源會透過就緒檢查、審核狀態或拍攝阻擋條件揭露；全新簽出的原始碼無法執行所有辨識任務。

[專案概覽](../README.zh-TW.md)
