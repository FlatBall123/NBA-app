"""
SPV (Statistical Performance Value) 計算與分群模組
=================================================
SPV 是依據球員每場各項數據加權後得到的綜合戰力指標。

常數定義（沿用原專案）：
    PSPP = 1.12   每場得分加權
    PAPP = 1.08   每場失分扣分
    PPB  = 2.24   助攻轉換為得分權重
    DB   = 0.05   防守籃板額外加權
    LPPB = 2.30   阻攻權重
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class SPVWeights:
    """SPV 計算用的權重；可在程式入口處覆寫做敏感度分析。"""
    PSPP: float = 1.12
    PAPP: float = 1.08
    PPB: float = 2.24
    DB: float = 0.05
    LPPB: float = 2.30


def compute_spv(df: pd.DataFrame, weights: SPVWeights | None = None) -> pd.DataFrame:
    """
    依據球員每場數據計算 SPV、進攻 SPV、防守 SPV、每分鐘 SPV。

    需要的欄位：MP, PTS, TRB, AST, STL, BLK, TOV, Age
    """
    w = weights or SPVWeights()
    out = df.copy()

    out["Age_Sq"] = out["Age"] ** 2

    out["MIN_VAL"] = out["MP"]
    out["PTS_VAL"] = out["PTS"]
    out["REB_VAL"] = out["TRB"] * (w.PSPP + w.DB)
    out["AST_VAL"] = out["AST"] * w.PPB
    out["STL_VAL"] = out["STL"] * (w.PSPP + w.PAPP)
    out["BLK_VAL"] = out["BLK"] * w.LPPB
    out["TOV_VAL"] = -out["TOV"] * (w.PSPP + w.PAPP)

    out["SPV"] = (
        out["MIN_VAL"] + out["PTS_VAL"] + out["REB_VAL"]
        + out["AST_VAL"] + out["STL_VAL"] + out["BLK_VAL"] + out["TOV_VAL"]
    )

    out["SPV_Offense"] = (
        out["MIN_VAL"] + out["PTS_VAL"] + out["AST_VAL"] - out["TOV_VAL"]
    )
    out["SPV_Defense"] = (
        out["MIN_VAL"] + out["REB_VAL"] + out["STL_VAL"] + out["BLK_VAL"]
    )

    # 每分鐘效率（不含 MIN_VAL，否則 MP 會抵銷）
    out["SPV_per_min"] = (
        out["PTS_VAL"] + out["REB_VAL"] + out["AST_VAL"]
        + out["STL_VAL"] + out["BLK_VAL"] + out["TOV_VAL"]
    ) / out["MP"]

    return out


def run_clustering(
    df: pd.DataFrame,
    column: str = "SPV",
    n_clusters: int = 3,
    random_state: int = 123,
) -> pd.DataFrame:
    """
    對指定欄位做 KMeans 分群，並把群編號依平均值由大到小重排
    （讓 cluster=0 永遠是「最強」群，方便後續解讀）。
    """
    out = df.copy()
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(out[[column]])

    # 重新編號：用群心由大到小排
    centers = km.cluster_centers_.flatten()
    order = sorted(range(n_clusters), key=lambda i: -centers[i])
    remap = {old: new for new, old in enumerate(order)}
    out[f"cluster_{column}"] = [remap[c] for c in raw_labels]

    # 為了相容舊欄位命名
    if column == "SPV":
        out["cluster"] = out[f"cluster_{column}"]
    return out
