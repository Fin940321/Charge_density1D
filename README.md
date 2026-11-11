# 單一電壓電荷密度分析

目前主程式是 `code/charge_density1D_V11.py`。它讀取 MDAnalysis 軌跡，以 OpenMM 力場取得粒子電荷，輸出陽離子、陰離子、溶劑及總溶液電荷密度的 z 向分布。

此次整理只搬移及分類檔案，不修改原有程式的計算、取樣或輸出設定。

## 目錄

| 位置 | 用途 |
| --- | --- |
| `code/charge_density1D_V11.py` | 目前維護的主程式 |
| `code/others/` | 原有輔助／其他流程，保留現況；不代表本次均已驗證 |
| `expired/` | 原有 V1–V10 等舊版 |
| `expired/from_Code/charge_density1D_V10.py` | 原本散放於 `Research/Code/` 的另一份 V10，保留來源區別 |
| `pic/` | 後續選定的成果圖片；本次未新增研究圖片 |
| `report/` | 版本差異、搬移對照及驗證紀錄 |
| `tab/` | 後續整理後的數據及資料來源說明；本次未還原已移除的原始資料 |

## 執行 V11

需要 Python、MDAnalysis、OpenMM、NumPy、Matplotlib 與 tqdm。工作區已有共用 Pixi 環境，此專案尚未附獨立鎖定環境。

主程式目前設定：

- `voltage = 6`。
- `framestart = 1000`，分析索引 1000 至最後一幀，**不是自動取後半段**；輸入必須超過 1000 幀。
- 目標 bin 寬度約為 0.01 nm，實際值依電極間距及整數 bin 數計算。
- 從執行時的工作目錄讀取輸入，輸出到該目錄的 `charge_density_6V/`。

為避免覆寫原有成果，請使用另備的分析執行目錄：

```text
analysis_run/
├── 6V_FV_NVT.dcd
├── 6V_start_drudes.pdb
├── 6V_start_nodrudes.pdb
├── ffdir/
└── electrode_ffdir/
```

```bash
cd /path/to/analysis_run
python /path/to/Charge_density1D/code/charge_density1D_V11.py
```

切換電壓時須檢查程式頂部的 `voltage`、對應輸入檔及取樣區間。正常輸出為四個 `hist_q_*_6V.dat` 與 `charge_density_6V.png`；重跑會覆寫同名輸出。

## 資料與力場來源

原始 run 與分析紀錄保留在工作區的 `Elec_sys/`，原始資料副本保留在 `Result/`。本專案不以新複本取代這兩處。

共用力場位於本專案同層的 `../SAPT_FF/ffdir/` 與 `../SAPT_FF/electrode_ffdir/`。使用者需要在分析執行目錄另備這兩組資料夾，且須與所選結構一致。此次核對的 34 個共用檔案與本專案整理前 HEAD 中的版本完全相同，沒有搬移、修改或重新產生力場。

只下載此 repository 尚不包含大型原始軌跡或上述共用力場，不能直接視為完整可執行的獨立套件。

## 版本與驗證範圍

兩份 V10 都保留：差異僅在終端顯示的版本文字。V11 則新增電壓變數、輸出目錄，並將起始影格改為 1000；本次不將它改回 V10 的取樣方式。

本次執行驗證使用 6 V 原始資料的暫存短軌跡，比較搬移前後的同一份 V11。這只用於確認整理沒有改變輸出，不構成完整軌跡或物理方法的重新驗證。既有單位標籤、原子索引與 Poisson 串接亦須依各自方法另行審核。

Git 歷史、遠端與既有未提交變更保留；沒有還原已移除檔案，也沒有 commit 或 push。詳見 [整理報告](report/organization.md) 與 [檔案對照](report/file_map.tsv)。
