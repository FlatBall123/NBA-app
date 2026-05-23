# 計算機程式設計 NBA APP 🏀

一站式 NBA 球員戰力分析工具：把原本分散在多個檔案的功能整合成單一 Streamlit 應用程式。

## 功能

| 頁面 | 說明 |
|---|---|
| **球員能力查詢** | 依隊伍與能力指標查詢球員、看分群、看散布圖、看 Top 40 |
| **比賽勝負預測** | 依 SPV 主力輪替 + 主場優勢預測單場或批次賽程 |
| **傷兵模擬器** | 扣除指定球員後重新計算戰力，模擬「今天某人不打」的勝率 |
| **最佳陣容** | 在薪資上限下用線性規劃挑出 SPV 最大的陣容 |
| **效率前緣** | 掃過多個預算上限，畫出風險 vs 報酬的效率前緣 |

## 專案結構

```
NBA_APP/
├── app.py                    # Streamlit 主程式（入口）
├── requirements.txt          # 套件清單
├── core/
│   ├── __init__.py
│   ├── data_loader.py        # 讀取 Excel + 清理 + 多年度掃描
│   ├── spv_calculator.py     # SPV 計算 + KMeans 分群
│   ├── predictor.py          # 比賽預測 + 傷兵模擬
│   └── portfolio.py          # Sharpe + PuLP 最佳化 + 效率前緣
└── data/
    ├── 2024/NBA_資料.xlsx
    ├── 2025/NBA_資料.xlsx
    └── ...
```

## 安裝

```bash
pip install -r requirements.txt
```

## 準備資料

把每一年的 NBA 資料 Excel 檔放到 `data/<年度>/` 底下。
檔名隨意（程式會自動掃 `*.xlsx`），但 **Sheet 名稱必須是**：
- `Player Salaries`
- `Player Stats Per Game`
- `Player Advanced Stats`

例如：

```
data/
  2023/NBA_資料.xlsx
  2024/NBA_資料.xlsx
  2025/NBA_資料.xlsx
```

App 啟動時會自動掃描有哪些年度，讓你在側邊欄切換。

## 啟動

```bash
streamlit run app.py
```

打開瀏覽器 `http://localhost:8501` 即可使用。

## 在 Python 裡單獨使用各模組

如果想寫 notebook 或別的腳本：

```python
from core import (
    load_and_clean_nba_data, compute_spv, run_clustering,
    build_team_strength, predict_single_game,
    compute_sharpe, optimize_lineup, efficient_frontier,
)

# 讀資料 + 算 SPV
df = load_and_clean_nba_data(season="2024")
df = compute_spv(df)
df = run_clustering(df, column="SPV", n_clusters=3)

# 比賽預測
team_stats = build_team_strength(df, rotation_size=8)
print(predict_single_game("LAL", "GSW", team_stats))

# 在 1.4 億預算下挑最佳陣容
df = compute_sharpe(df)
result = optimize_lineup(df, budget=140_000_000, team_size=15)
print(result.selected)
```

## 從舊版遷移

| 舊檔案 | 新位置 |
|---|---|
| `DataCleanAndSPV.py`（清理 + SPV）| `core/data_loader.py` + `core/spv_calculator.py` |
| `portfolio_..._frontier.ipynb` | `core/portfolio.py` |
| `game_prediction.ipynb` | `core/predictor.py` |
| `NBAApp.py`（Streamlit） | 併入 `app.py` 的「球員能力查詢」頁 |
| `clean_nba_data.csv`（中介檔）| **不再需要**，每次啟動都即時計算（有 cache） |

## 自訂 SPV 權重

```python
from core.spv_calculator import compute_spv, SPVWeights

custom_weights = SPVWeights(PSPP=1.20, PPB=2.50)
df = compute_spv(df, weights=custom_weights)
```
