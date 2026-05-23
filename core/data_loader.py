"""
資料讀取與清理模組（v2 — 自動相容多版本格式）
==============================================
支援兩種 Excel 格式：
  - 舊版：Sheet 名為 'Player Salaries' / 'Player Stats Per Game' / 'Player Advanced Stats'
          有 '-additional' 與 'Player-additional' 欄位作為 PlayerID
  - 新版（如 2026）：Sheet 名為 '2025-2026 player salaries' 等小寫變形
          沒有 PlayerID 欄位 → 自動用 'Player + Team' 組合當合併鍵

資料夾結構：
    data/
      2024/NBA_資料.xlsx
      2026/2026_nba_regular.xlsx
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def list_available_seasons(data_root: str | Path = DEFAULT_DATA_ROOT) -> List[str]:
    """掃描 data/ 底下有哪些年度子資料夾可用。"""
    root = Path(data_root)
    if not root.exists():
        return []
    seasons = [d.name for d in root.iterdir() if d.is_dir() and any(d.glob("*.xlsx"))]
    return sorted(seasons)


def _find_xlsx(season_dir: Path) -> Path:
    candidates = list(season_dir.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"在 {season_dir} 找不到任何 .xlsx 檔案")
    return candidates[0]


def _find_sheet(sheet_names: list, keywords: list) -> str:
    """模糊比對 sheet 名稱：找出包含所有 keyword 的 sheet。"""
    for name in sheet_names:
        lower = name.lower()
        if all(kw.lower() in lower for kw in keywords):
            return name
    raise ValueError(
        f"找不到包含關鍵字 {keywords} 的 Sheet。可用的 Sheet：{sheet_names}"
    )


def _clean_salary(df: pd.DataFrame) -> pd.DataFrame:
    """清理薪資資料，相容有/沒有 -additional 欄位的版本。"""
    has_player_id = "-additional" in df.columns
    salary_col = next((c for c in df.columns if str(c).startswith("Salary")), df.columns[3])

    if has_player_id:
        df = df.rename(columns={
            "-additional": "PlayerID",
            df.columns[1]: "Player",
            df.columns[2]: "Team",
            salary_col: "Salary",
        })
        df = df[["PlayerID", "Player", "Team", "Salary"]].copy()
    else:
        df = df.rename(columns={
            df.columns[1]: "Player",
            df.columns[2]: "Team",
            salary_col: "Salary",
        })
        df = df[["Player", "Team", "Salary"]].copy()

    df["Salary"] = df["Salary"].astype(str).str.replace(r"[$,]", "", regex=True)
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df = df.dropna(subset=["Salary", "Player"])
    df = df.drop_duplicates(subset=["Player"])
    df = df[df["Player"].astype(str).str.lower() != "player"].reset_index(drop=True)
    return df


def _dedupe_multi_team(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """處理 2TM/3TM 合計列：保留實際隊伍那筆。"""
    df = df.copy()
    df["is_multi"] = df["Team"].isin(["2TM", "3TM"])
    df = df.sort_values(by=[id_col, "is_multi"])
    df = df.drop_duplicates(subset=[id_col], keep="first")
    return df


def _clean_stats(df: pd.DataFrame) -> pd.DataFrame:
    has_player_id = "Player-additional" in df.columns

    if has_player_id:
        df = df.rename(columns={"Player-additional": "PlayerID"})
        df = _dedupe_multi_team(df, id_col="PlayerID")
        df = df.drop(columns=["Player", "Team", "is_multi"])
    else:
        df = _dedupe_multi_team(df, id_col="Player")
        df = df.drop(columns=["is_multi"])

    if "PTS" in df.columns:
        df = df[pd.to_numeric(df["PTS"], errors="coerce").notna()].reset_index(drop=True)
    return df


def _clean_advanced(df: pd.DataFrame) -> pd.DataFrame:
    has_player_id = "Player-additional" in df.columns

    if has_player_id:
        df = df.rename(columns={"Player-additional": "PlayerID"})
        df = _dedupe_multi_team(df, id_col="PlayerID")
        drop_cols = ["Rk", "Player", "Age", "Team", "Pos", "G", "GS", "MP", "is_multi"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    else:
        df = _dedupe_multi_team(df, id_col="Player")
        drop_cols = ["Rk", "Age", "Pos", "G", "GS", "MP", "is_multi"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def load_and_clean_nba_data(
    season: str | None = None,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    file_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    主要入口：載入並清理 NBA 球員資料。自動偵測新舊格式。
    """
    if file_path is not None:
        xlsx_path = Path(file_path)
    elif season is not None:
        xlsx_path = _find_xlsx(Path(data_root) / season)
    else:
        raise ValueError("請提供 season 或 file_path 其中一個")

    if not xlsx_path.exists():
        raise FileNotFoundError(f"找不到資料檔: {xlsx_path}")

    xl = pd.ExcelFile(xlsx_path)
    sheet_salary = _find_sheet(xl.sheet_names, ["salar"])
    sheet_stats = _find_sheet(xl.sheet_names, ["stats", "per game"])
    sheet_adv = _find_sheet(xl.sheet_names, ["advanced"])

    salary = pd.read_excel(xlsx_path, sheet_name=sheet_salary)
    stats = pd.read_excel(xlsx_path, sheet_name=sheet_stats)
    adv = pd.read_excel(xlsx_path, sheet_name=sheet_adv)

    salary = _clean_salary(salary)
    stats = _clean_stats(stats)
    adv = _clean_advanced(adv)

    # 自動選擇合併鍵
    if "PlayerID" in salary.columns and "PlayerID" in stats.columns:
        merge_keys = ["PlayerID"]
    else:
        merge_keys = ["Player", "Team"]

    merged = pd.merge(salary, stats, on=merge_keys, how="left")
    merged = pd.merge(merged, adv, on=merge_keys, how="left")

    if "PlayerID" not in merged.columns:
        merged["PlayerID"] = merged["Player"].astype(str) + "_" + merged["Team"].astype(str)

    cleaned = merged.dropna(subset=["Salary", "MP"]).copy().reset_index(drop=True)
    return cleaned
