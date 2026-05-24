"""
計算機程式設計 NBA APP
=====================
一站式介面，支援：
  1. 切換不同年度的 NBA 資料
  2. 球員 SPV 查詢與分群檢視
  3. 比賽勝負預測（含主場優勢）
  4. 傷兵情境模擬
  5. 薪資預算下的最佳陣容
  6. 投資組合效率前緣

執行方式：
    streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core import (
    build_team_strength,
    compute_sharpe,
    compute_spv,
    efficient_frontier,
    list_available_seasons,
    load_and_clean_nba_data,
    optimize_lineup,
    predict_schedule,
    predict_single_game,
    run_clustering,
    simulate_with_injuries,
)

# ---------- 頁面設定 ----------
st.set_page_config(
    page_title="計算機程式設計 NBA APP",
    page_icon="🏀",
    layout="wide",
)
st.title("🏀 計算機程式設計 NBA APP")
st.caption("球員戰力分析 × 比賽預測 × 投資組合最佳化")


# ---------- 資料載入（用 cache 避免每次互動都重算）----------
@st.cache_data(show_spinner="正在載入與清理資料…")
def _load_full_pipeline(season: str) -> pd.DataFrame:
    """讀檔 → 清理 → 算 SPV → 分群 → 算 Sharpe。一條龍。"""
    raw = load_and_clean_nba_data(season=season)
    with_spv = compute_spv(raw)
    clustered = run_clustering(with_spv, column="SPV", n_clusters=3)
    clustered = run_clustering(clustered, column="SPV_per_min", n_clusters=3)
    return compute_sharpe(clustered)


# ---------- 側邊欄：年度選擇 + 全域篩選 ----------
st.sidebar.header("📅 資料年度")
seasons = list_available_seasons()
if not seasons:
    st.error(
        "在 `data/` 目錄下找不到任何年度資料夾。\n\n"
        "請建立如下結構：\n```\n"
        "data/\n  2024/NBA_資料.xlsx\n  2025/NBA_資料.xlsx\n"
        "```"
    )
    st.stop()

selected_season = st.sidebar.selectbox("選擇賽季年度：", seasons, index=len(seasons) - 1)
data = _load_full_pipeline(selected_season)
team_options = ["全部球隊"] + sorted(data["Team"].dropna().unique().tolist())

st.sidebar.markdown(f"**已載入 {len(data)} 名球員**")
st.sidebar.markdown("---")
st.sidebar.header("🔍 功能選單")
page = st.sidebar.radio(
    "選擇頁面：",
    [
        "球員能力查詢",
        "比賽勝負預測",
        "傷兵模擬器",
        "最佳陣容（薪資上限）",
        "效率前緣",
    ],
)


# =========================================================
# 頁面 1：球員能力查詢（取代原 NBAApp.py）
# =========================================================
if page == "球員能力查詢":
    st.subheader("🔎 球員能力查詢")

    col1, col2 = st.columns(2)
    with col1:
        team_choice = st.selectbox("選擇隊伍：", team_options)
    with col2:
        ability = st.selectbox(
            "選擇能力指標：",
            ["SPV", "SPV_Offense", "SPV_Defense", "SPV_per_min",
             "PTS_VAL", "REB_VAL", "AST_VAL", "STL_VAL", "BLK_VAL",
             "Player_Sharpe"],
        )

    if team_choice == "全部球隊":
        df_show = data
    else:
        df_show = data[data["Team"] == team_choice]

    st.markdown(f"**目前顯示**: {team_choice} 的 `{ability}` 排序")

    display_cols = ["Player", "Team", "cluster", ability, "Salary"]
    st.dataframe(
        df_show[display_cols].sort_values(ability, ascending=False),
        use_container_width=True,
        height=400,
    )

    # 散布圖
    fig = px.scatter(
        df_show, x=ability, y="Salary",
        color=df_show["cluster"].astype(str),
        hover_name="Player",
        labels={"color": "分群", "Salary": "薪水（美元）"},
        title=f"{ability} 與薪資的關係（顏色＝SPV 分群）",
    )
    fig.update_traces(marker=dict(size=11, opacity=0.85,
                                  line=dict(width=1, color="DarkSlateGrey")))
    st.plotly_chart(fig, use_container_width=True)

    # Top 40 長條圖
    top = df_show.nlargest(40, ability)
    bar = px.bar(top, x=ability, y="Player", orientation="h",
                 title=f"{ability} Top 40", height=800)
    bar.update_yaxes(autorange="reversed")
    st.plotly_chart(bar, use_container_width=True)


# =========================================================
# 頁面 2：比賽勝負預測（取代 game_prediction.ipynb 批次預測）
# =========================================================
elif page == "比賽勝負預測":
    st.subheader("🏟️ 比賽勝負預測")
    team_stats = build_team_strength(data, rotation_size=8)

    st.markdown("### 單場預測")
    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("主隊：", sorted(team_stats["Team"].unique()), key="single_home")
    with c2:
        away = st.selectbox("客隊：", sorted(team_stats["Team"].unique()),
                            index=1, key="single_away")

    if st.button("預測這場", type="primary"):
        result = predict_single_game(home, away, team_stats)
        if result.get("error"):
            st.error(result["error"])
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{home}（主）戰力", result["home_strength"])
            m2.metric(f"{away}（客）戰力", result["away_strength"])
            m3.metric("預測勝者", result["winner"],
                      delta=f"{result['win_probability']:.1f}% 勝率")

    st.markdown("---")
    st.markdown("### 批次預測賽程")
    st.caption("上傳 CSV，需要欄位：`Date`, `Home_Team`, `Away_Team`")
    uploaded = st.file_uploader("選擇賽程 CSV", type=["csv"])

    if uploaded is not None:
        schedule = pd.read_csv(uploaded)
        try:
            results = predict_schedule(schedule, team_stats)
            st.dataframe(results, use_container_width=True)
            csv = results.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ 下載預測結果", csv, "predictions.csv", "text/csv")
        except ValueError as e:
            st.error(str(e))

    with st.expander("球隊戰力總表"):
        st.dataframe(team_stats.sort_values("Final_Strength", ascending=False),
                     use_container_width=True)


# =========================================================
# 頁面 3：傷兵模擬器（取代 game_prediction.ipynb 中的 input() 模擬器）
# =========================================================
elif page == "傷兵模擬器":
    st.subheader("🤕 傷兵模擬器")
    team_stats = build_team_strength(data, rotation_size=8)
    teams = sorted(team_stats["Team"].unique())

    c1, c2 = st.columns(2)
    with c1:
        home = st.selectbox("主隊：", teams, key="inj_home")
    with c2:
        away = st.selectbox("客隊：", teams, index=1, key="inj_away")

    # 從兩隊可選出傷兵
    candidates = data[data["Team"].isin([home, away])]
    candidates = candidates.sort_values("SPV", ascending=False)
    injured = st.multiselect(
        "選擇今天無法上場的球員：",
        options=candidates["Player"].tolist(),
        help="可以多選；列出的球員會從前 8 名主力中剔除後重新計算",
    )

    if st.button("模擬", type="primary"):
        result = simulate_with_injuries(home, away, data, team_stats,
                                        injured_players=injured)
        if result.get("error"):
            st.error(result["error"])
        else:
            st.success(f"預測勝者：**{result['winner']}**（{result['win_probability']:.1f}%）")
            st.json(result)


# =========================================================
# 頁面 4：最佳陣容（取代 portfolio notebook）
# =========================================================
elif page == "最佳陣容（薪資上限）":
    st.subheader("💰 在薪資上限下挑出最強陣容")
    st.caption("用線性規劃（PuLP）找出 SPV 總和最大、且符合薪資上限的陣容組合")

    c1, c2 = st.columns(2)
    with c1:
        budget_m = st.slider("薪資上限（百萬美金）", 50, 300, 140, step=5)
    with c2:
        team_size = st.slider("陣容人數", 5, 20, 15)

    if st.button("計算最佳陣容", type="primary"):
        with st.spinner("線性規劃求解中…"):
            result = optimize_lineup(data, budget=budget_m * 1_000_000,
                                     team_size=team_size)

        if result.status != "Optimal":
            st.error(f"找不到可行解（狀態：{result.status}）。請調高預算或減少人數。")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("總薪資", f"${result.total_salary:,.0f}")
            m2.metric("總 SPV", f"{result.total_spv:.2f}")
            m3.metric("平均風險 σ", f"{result.avg_sigma:.2f}")

            st.dataframe(
                result.selected[["Player", "Team", "Salary", "SPV",
                                 "Player_Sharpe"]].reset_index(drop=True),
                use_container_width=True,
            )

            # 標出被選中的球員 vs 全體
            data_plot = data.copy()
            data_plot["Selected"] = data_plot.index.isin(result.selected.index)

            # 兩張圖並排：左＝薪資 vs SPV，右＝風險(σ) vs SPV
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_salary = px.scatter(
                    data_plot, x="Salary", y="SPV",
                    color="Selected", hover_name="Player",
                    title=f"薪資 vs 戰力（預算 ${budget_m}M）",
                    labels={"Salary": "薪資（成本）", "SPV": "綜合戰力 SPV"},
                    color_discrete_map={True: "red", False: "lightgray"},
                )
                st.plotly_chart(fig_salary, use_container_width=True)

            with chart_col2:
                fig_risk = px.scatter(
                    data_plot, x="Player_Sigma", y="SPV",
                    color="Selected", hover_name="Player",
                    title=f"風險 vs 報酬（紅點＝最佳陣容）",
                    labels={"Player_Sigma": "表現波動度 σ（風險）",
                            "SPV": "綜合戰力 SPV（報酬）"},
                    color_discrete_map={True: "red", False: "lightgray"},
                )
                st.plotly_chart(fig_risk, use_container_width=True)


# =========================================================
# 頁面 5：效率前緣
# =========================================================
elif page == "效率前緣":
    st.subheader("📈 投資組合效率前緣")
    st.caption("掃過不同預算上限，記錄每個預算下能達到的最大 SPV 與對應風險。")

    c1, c2, c3 = st.columns(3)
    with c1:
        min_b = st.number_input("最小預算（百萬）", 50, 250, 80, step=5)
    with c2:
        max_b = st.number_input("最大預算（百萬）", 60, 400, 200, step=5)
    with c3:
        step = st.number_input("間隔（百萬）", 5, 50, 10, step=5)

    if st.button("繪製效率前緣", type="primary"):
        with st.spinner("正在掃描多個預算組合… 這可能需要 30~60 秒"):
            frontier = efficient_frontier(
                data, min_budget_m=min_b, max_budget_m=max_b, step_m=step,
            )

        if frontier.empty:
            st.error("沒有任何預算找到可行解。")
        else:
            st.dataframe(frontier, use_container_width=True)
            fig = px.line(
                frontier, x="Avg_Sigma", y="Total_SPV",
                markers=True, text="Budget_M",
                labels={"Avg_Sigma": "平均風險 σ", "Total_SPV": "投資組合報酬 (總 SPV)"},
                title="NBA 陣容效率前緣",
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
