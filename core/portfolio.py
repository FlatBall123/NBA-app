"""
投資組合最佳化模組
==================
把球員當作金融資產：
  - SPV = 預期報酬
  - Player_Sigma = 風險（波動度）
  - Sharpe Ratio = (報酬 - 基準) / 風險
  - 在薪資上限下，用線性規劃挑出最佳陣容
  - 掃過不同預算 → 畫出效率前緣

來源：portfolio_and_efficiency_frontier_under_fixed_budget.ipynb
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    """單次最佳化結果的容器。"""
    budget: float
    team_size: int
    selected: pd.DataFrame  # 被選中的球員
    total_salary: float
    total_spv: float
    avg_sigma: float
    status: str  # 'Optimal' / 'Infeasible' / ...


def compute_sharpe(
    df: pd.DataFrame,
    seed: int = 42,
    sigma_scale: float = 5.0,
    sigma_noise: tuple = (1.5, 3.5),
) -> pd.DataFrame:
    """
    為每位球員計算「夏普比率」。
    Player_Sigma 由 SPV_per_min 推導 + 隨機雜訊（模擬市場波動）。
    若你日後有真實波動度資料，把這段換掉即可。
    """
    if "SPV_per_min" not in df.columns:
        raise ValueError("df 必須先跑過 compute_spv() 才有 SPV_per_min")

    out = df.copy()
    rng = np.random.default_rng(seed)

    out["Player_Sigma"] = (
        np.abs(out["SPV_per_min"] * sigma_scale)
        + rng.uniform(sigma_noise[0], sigma_noise[1], size=len(out))
    )

    league_mean = out["SPV"].mean()
    out["Player_Sharpe"] = (out["SPV"] - league_mean) / out["Player_Sigma"]
    return out


def optimize_lineup(
    df: pd.DataFrame,
    budget: float,
    team_size: int = 15,
    objective: str = "SPV",
) -> PortfolioResult:
    """
    在薪資上限 budget 下，挑出 team_size 名球員使 objective 總和最大。

    用 PuLP 解 0/1 整數規劃。
    """
    try:
        import pulp
    except ImportError as e:
        raise ImportError("請先安裝 pulp： pip install pulp") from e

    if objective not in df.columns:
        raise ValueError(f"目標欄位 {objective} 不存在")

    prob = pulp.LpProblem("NBA_Portfolio_Optimization", pulp.LpMaximize)
    idx = df.index.tolist()
    x = pulp.LpVariable.dicts("Player", idx, cat="Binary")

    # 最大化目標
    prob += pulp.lpSum([df.loc[i, objective] * x[i] for i in idx]), "Objective"
    # 薪資上限
    prob += (
        pulp.lpSum([df.loc[i, "Salary"] * x[i] for i in idx]) <= budget,
        "Salary_Cap",
    )
    # 人數限制
    prob += pulp.lpSum([x[i] for i in idx]) == team_size, "Team_Size"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status_str = pulp.LpStatus[status]

    if status_str != "Optimal":
        return PortfolioResult(
            budget=budget, team_size=team_size,
            selected=df.iloc[0:0], total_salary=0,
            total_spv=0, avg_sigma=0, status=status_str,
        )

    selected_idx = [i for i in idx if x[i].varValue == 1]
    selected = df.loc[selected_idx].copy()

    avg_sigma = (
        float(selected["Player_Sigma"].mean())
        if "Player_Sigma" in selected.columns else 0.0
    )

    return PortfolioResult(
        budget=budget,
        team_size=team_size,
        selected=selected,
        total_salary=float(selected["Salary"].sum()),
        total_spv=float(selected[objective].sum()),
        avg_sigma=avg_sigma,
        status=status_str,
    )


def efficient_frontier(
    df: pd.DataFrame,
    min_budget_m: float = 80,
    max_budget_m: float = 200,
    step_m: float = 5,
    team_size: int = 15,
) -> pd.DataFrame:
    """
    掃過不同預算 → 每個預算下找最大 SPV → 蒐集 (風險, 報酬) 構成效率前緣。

    Returns
    -------
    DataFrame  欄位：Budget_M, Total_SPV, Avg_Sigma
    """
    if "Player_Sigma" not in df.columns:
        df = compute_sharpe(df)

    rows: List[dict] = []
    budget_m = min_budget_m
    while budget_m <= max_budget_m:
        result = optimize_lineup(
            df, budget=budget_m * 1_000_000, team_size=team_size, objective="SPV"
        )
        if result.status == "Optimal":
            rows.append({
                "Budget_M": budget_m,
                "Total_SPV": result.total_spv,
                "Avg_Sigma": result.avg_sigma,
                "Total_Salary": result.total_salary,
            })
        budget_m += step_m

    return pd.DataFrame(rows)
