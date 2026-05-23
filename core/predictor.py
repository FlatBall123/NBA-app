"""
比賽預測模組
============
基於球員 SPV 推算球隊戰力，並預測單場/批次對戰結果。
原始邏輯來自 game_prediction.ipynb，已重構為純函式（無 input()）
以便 Streamlit 介面驅動。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


HOME_ADVANTAGE = 1.05  # 主場戰力加成 5%
ROTATION_SIZE = 8  # 取每隊 SPV 前 N 名作為主力輪替


def build_team_strength(
    player_df: pd.DataFrame,
    rotation_size: int = ROTATION_SIZE,
    recent_form: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    計算每支球隊的綜合戰力。

    Parameters
    ----------
    player_df : 已經跑過 compute_spv 的球員資料表。
    rotation_size : 取每隊 SPV 前幾名作主力，預設 8 人。
    recent_form : 近況資料 DataFrame，需有 'Team' 和 'EFF' 兩欄；若 None
                  則所有球隊的 Form_Modifier=1.0。

    Returns
    -------
    DataFrame 欄位：Team, Base_SPV, Avg_SPV_per_Min, Player_Count,
                    Form_Modifier, EFF, Final_Strength
    """
    if "SPV" not in player_df.columns:
        raise ValueError("player_df 必須先跑過 compute_spv()")

    # 基礎戰力：每隊前 N 名球員 SPV 加總
    base = (
        player_df.groupby("Team")["SPV"]
        .apply(lambda x: x.nlargest(rotation_size).sum())
        .reset_index()
        .rename(columns={"SPV": "Base_SPV"})
    )

    extras = (
        player_df.groupby("Team")
        .agg(
            Avg_SPV_per_Min=("SPV_per_min", "mean"),
            Player_Count=("PlayerID", "count"),
        )
        .reset_index()
    )

    team_stats = pd.merge(base, extras, on="Team")

    # 近況修正
    if recent_form is None or recent_form.empty:
        team_stats["Form_Modifier"] = 1.0
        team_stats["EFF"] = np.nan
    else:
        avg_eff = recent_form["EFF"].mean()
        recent_form = recent_form.copy()
        recent_form["Form_Modifier"] = recent_form["EFF"] / avg_eff
        team_stats = pd.merge(
            team_stats,
            recent_form[["Team", "Form_Modifier", "EFF"]],
            on="Team",
            how="left",
        )
        team_stats["Form_Modifier"] = team_stats["Form_Modifier"].fillna(1.0)
        team_stats["EFF"] = team_stats["EFF"].fillna(avg_eff)

    team_stats["Final_Strength"] = team_stats["Base_SPV"] * team_stats["Form_Modifier"]
    return team_stats


def _strength_of(team_stats: pd.DataFrame, team: str) -> float:
    """從 team_stats 抓出某隊的 Final_Strength；找不到回 nan。"""
    row = team_stats.loc[team_stats["Team"] == team, "Final_Strength"]
    return float(row.values[0]) if len(row) else float("nan")


def predict_single_game(
    home_team: str,
    away_team: str,
    team_stats: pd.DataFrame,
    home_advantage: float = HOME_ADVANTAGE,
) -> dict:
    """
    預測單場比賽。回傳 dict 含勝者、雙方戰力、勝率。
    """
    h = _strength_of(team_stats, home_team)
    a = _strength_of(team_stats, away_team)

    if np.isnan(h) or np.isnan(a):
        return {
            "home": home_team, "away": away_team,
            "winner": None, "error": "資料庫中找不到隊伍",
            "home_strength": h, "away_strength": a,
        }

    h_adj = h * home_advantage
    a_adj = a
    total = h_adj + a_adj

    if h_adj >= a_adj:
        winner, win_prob = home_team, h_adj / total * 100
    else:
        winner, win_prob = away_team, a_adj / total * 100

    return {
        "home": home_team, "away": away_team,
        "home_strength": round(h_adj, 2),
        "away_strength": round(a_adj, 2),
        "winner": winner,
        "win_probability": round(win_prob, 2),
        "margin": round(abs(h_adj - a_adj), 2),
    }


def predict_schedule(schedule_df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    批次預測一整張賽程表。

    schedule_df 必須含欄位：Date, Home_Team, Away_Team
    """
    required = {"Date", "Home_Team", "Away_Team"}
    missing = required - set(schedule_df.columns)
    if missing:
        raise ValueError(f"schedule_df 缺少欄位: {missing}")

    records = []
    for _, row in schedule_df.iterrows():
        result = predict_single_game(row["Home_Team"], row["Away_Team"], team_stats)
        result["Date"] = row["Date"]
        records.append(result)

    return pd.DataFrame(records)


def simulate_with_injuries(
    home_team: str,
    away_team: str,
    player_df: pd.DataFrame,
    team_stats: pd.DataFrame,
    injured_players: Iterable[str] = (),
    rotation_size: int = ROTATION_SIZE,
    home_advantage: float = HOME_ADVANTAGE,
) -> dict:
    """
    扣除傷兵後重新計算戰力並預測。
    取代原本 game_prediction.ipynb 中需要 input() 的模擬器，
    改為純函式，方便 Streamlit/API 呼叫。
    """
    injured = set(injured_players or [])

    def custom_base(team: str) -> float:
        avail = player_df[(player_df["Team"] == team) & (~player_df["Player"].isin(injured))]
        if avail.empty:
            return 0.0
        return float(avail.nlargest(rotation_size, "SPV")["SPV"].sum())

    home_base = custom_base(home_team)
    away_base = custom_base(away_team)

    # 從 team_stats 撈近況修正係數
    def modifier(team: str) -> float:
        row = team_stats.loc[team_stats["Team"] == team, "Form_Modifier"]
        return float(row.values[0]) if len(row) else 1.0

    h_strength = home_base * modifier(home_team) * home_advantage
    a_strength = away_base * modifier(away_team)

    total = h_strength + a_strength
    if total == 0:
        return {
            "home": home_team, "away": away_team,
            "winner": None, "error": "兩隊戰力皆為 0",
        }

    if h_strength >= a_strength:
        winner, win_prob = home_team, h_strength / total * 100
    else:
        winner, win_prob = away_team, a_strength / total * 100

    return {
        "home": home_team, "away": away_team,
        "injured": sorted(injured),
        "home_strength": round(h_strength, 2),
        "away_strength": round(a_strength, 2),
        "winner": winner,
        "win_probability": round(win_prob, 2),
    }
