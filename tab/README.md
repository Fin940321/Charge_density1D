# 數據與輸入來源

此目錄供後續整理後的數據使用。本次未還原 Git 狀態中已顯示移除的舊數據、PDB、log 或力場。

- 原始資料與既有分析現場：工作區 `Elec_sys/`。
- 額外保存的原始資料副本：工作區 `Result/`。
- 共用力場：本專案同層的 `../SAPT_FF/`，需在執行目錄準備 `ffdir/`、`electrode_ffdir/`。
- V11 預設從執行目錄讀取 `6V_FV_NVT.dcd`、`6V_start_drudes.pdb`、`6V_start_nodrudes.pdb`。

數據收錄時應記錄來源 run、電壓、使用的程式版本、取樣影格與單位，不能只依檔名推定條件。
