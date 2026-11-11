# 第六階段整理紀錄

## 檔案安排

- 專案由 `New_Analysis/Others/Charge_density1D/` 搬至 `New_Analysis/Code/Charge_density1D/`。
- 保留使用者已整理的 `code/`、`code/others/` 與 `expired/`，不修改原有程式內容。
- 保留完整 `.git`、`.gitignore`、既有未提交刪除及未追蹤檔案，不執行還原、暫存、提交或推送。
- 原 `New_Analysis/Code/charge_density1D_V10.py` 移入 `expired/from_Code/`，不覆蓋原有 `expired/charge_density1D_V10.py`。
- 新增 `pic/`、`report/`、`tab/` 的用途說明，逐檔對照見 `file_map.tsv`。

## 版本核對

原 `expired/charge_density1D_V10.py` 與 HEAD 的根目錄 V10 相同；另一份散放於 `Code/` 的 V10 僅改了終端顯示文字中的版本標籤。兩份都保留來源。

V11 使用 `voltage = 6` 組合輸入及輸出名稱，輸出到 `charge_density_6V/`，從影格索引 1000 開始。原 V10 使用不帶電壓前綴的輸入檔名，並取軌跡後半段。本次不統一或改變這些設定。

## 力場核對

`Code/SAPT_FF/` 的 34 個檔案與整理前此專案 HEAD 中對應力場完全相同。來源檔保留現址；測試只將副本放入暫存執行目錄。

## 驗證範圍

使用 `Result/BMIM_TFSI_2CNTG_100ns_CV_Done/6V/` 的 PDB 與 DCD，在暫存目錄建立含原始索引 0–1001 的 1002 幀軌跡。V11 保持原樣，自行分析最後兩幀（原始索引 1000、1001）；因此不必為了小型測試修改正式程式的起始影格設定。

整理前後使用相同輸入與力場，比較四個數據表與 PNG，核對數值是否有限、bin 座標是否一致。原始軌跡與既有研究結果不被覆寫。此測試不等同於完整軌跡或 Poisson 串接的驗證。

驗證結果：四個 1074×2 數據表與一張 PNG 在搬移前後逐位元組一致，數據皆為有限值；四組數據使用同一組遞增的 z 座標，陽離子、陰離子、溶劑數值之和在輸出捨入精度內等於總值，沒有新增警告。

初次以固定容差檢查 bin 間距時未通過。進一步從原 PDB 重建原程式的計算，確認 z 中心點使用 float32，輸出座標逐值符合該計算與八位小數格式。此測試的相鄰間距約為 0.10003662–0.10006714 Å，搬移前後相同；本次未修改精度或算法。

## 既有提示

MDAnalysis 讀取此 PDB 時提示部分原子的 `EP` 元素欄位無法辨識；DCDReader 亦提示未來版本的 timestep 行為將改變。本次保留原始資料及算法，不為消除這些既有提示而修改檔案。

## 2026-10-29 外層整理更新

目前路徑為 `/home/atuo/Academic_Workspace/Research/Code/Charge_density1D/`。先前紀錄中的來源路徑保留作歷史對照。README 的目前執行範例已更新；共通手動事項見工作區根目錄 MANUAL_ADJUSTMENTS.md。
