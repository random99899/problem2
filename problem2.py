"""
Problem 2: 双轨动态反事实演化模型 (Dual-Track Dynamic Counterfactual Evolution Model)
对比「排名制」与「百分比制」两种赛制在所有赛季的影响，分析四位争议选手，并使用AHP推荐未来赛制。
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. 配置与路径 (使用相对路径)
# ==========================================
# 输入文件
FILE_Q1_SOL = 'problem1/problem1_detailed_solution.csv'
FILE_WEEK_DATA = 'df_week_level.csv'

# 输出目录
OUTPUT_DIR = 'problem2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 可视化配色
COLORS = ['#56a6c7', '#98afba', '#bc8eb7', '#b7a6b5']

# 蒙特卡洛次数
N_SIMULATIONS = 100

# 四位争议选手定义
CONTROVERSY_STARS = [
    {'season': 2, 'name': 'Jerry Rice', 'note': 'runner up despite lowest judges scores in 5 weeks'},
    {'season': 4, 'name': 'Billy Ray Cyrus', 'note': '5th despite last place judge scores in 6 weeks'},
    {'season': 11, 'name': 'Bristol Palin', 'note': '3rd with lowest judge scores 12 times'},
    {'season': 27, 'name': 'Bobby Bones', 'note': 'won despite consistently low judges scores'},
]

# 绘图风格
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

def clean_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)


# ==========================================
# 1. 裁判分生成模块 (Judge Score Generator)
# ==========================================
class JudgeScoreGenerator:
    """解决右截尾缺失：为反事实中'复活'的选手生成合理的评委分"""

    def __init__(self, df_week_level):
        self.data = df_week_level.copy()
        # 计算评分列 (优先使用avg_score，否则用total_score)
        if 'avg_score' in self.data.columns:
            self.score_col = 'avg_score'
        elif 'total_score' in self.data.columns:
            self.score_col = 'total_score'
        else:
            raise ValueError("找不到评分列")
        
        # 计算每周的赛季平均分趋势 (Global Trend)
        self.season_week_means = self.data.groupby(['season', 'week'])[self.score_col].mean()

        # 计算选手的个人能力偏差 (Contestant Bias)
        self.data = self.data.merge(
            self.season_week_means.rename('week_mean').reset_index(), 
            on=['season', 'week'], how='left'
        )
        self.data['bias'] = self.data[self.score_col] - self.data['week_mean']

        self.contestant_bias = self.data.groupby(['season', 'celebrity_name'])['bias'].agg(['mean', 'std']).fillna(0)
        self.global_noise = max(self.data['bias'].std(), 0.5)
        
        # 评分范围
        self.score_min = self.data[self.score_col].min()
        self.score_max = self.data[self.score_col].max()

    def get_score(self, season, week, celebrity_name, observed_df=None):
        # 1. 优先使用真实历史数据
        if observed_df is not None:
            match = observed_df[(observed_df['season'] == season) &
                                (observed_df['week'] == week) &
                                (observed_df['celebrity_name'] == celebrity_name)]
            if not match.empty:
                val = match[self.score_col].values[0]
                if not np.isnan(val) and val > 0: 
                    return val

        # 2. 缺失值生成 (Imputation)
        if (season, week) in self.season_week_means.index:
            base = self.season_week_means.loc[(season, week)]
        else:
            base = self.data[self.score_col].mean()

        # 获取个人偏差
        if (season, celebrity_name) in self.contestant_bias.index:
            bias_mean = self.contestant_bias.loc[(season, celebrity_name), 'mean']
            bias_std = self.contestant_bias.loc[(season, celebrity_name), 'std']
            if bias_std == 0: bias_std = self.global_noise
        else:
            bias_mean = 0
            bias_std = self.global_noise

        # 随机生成并截断
        return np.clip(base + np.random.normal(bias_mean, bias_std), self.score_min, self.score_max)


# ==========================================
# 2. 双轨动态反事实仿真器
# ==========================================
class CounterfactualSimulator:
    def __init__(self, q1_df, week_df):
        self.q1 = q1_df
        self.week_data = week_df
        self.jsg = JudgeScoreGenerator(week_df)
        self.score_col = self.jsg.score_col

    def get_fan_votes(self, season, week, active_contestants):
        """从问题一的后验分布中采样粉丝投票份额"""
        subset = self.q1[(self.q1['season'] == season) &
                         (self.q1['week'] == week) &
                         (self.q1['celebrity_name'].isin(active_contestants))]
        votes = {}
        for _, row in subset.iterrows():
            # 采样并保证非负
            std = row.get('uncertainty_std', 0.02)
            votes[row['celebrity_name']] = max(0.001, np.random.normal(row['est_fan_vote_pct'], std))

        # 补全缺失选手
        for c in active_contestants:
            if c not in votes: votes[c] = 0.01

        # 归一化
        total = sum(votes.values())
        return {k: v / total for k, v in votes.items()}

    def run_season(self, season, system, judges_save=False, track_celebrity=None):
        """
        运行单赛季仿真
        Args:
            season: 赛季号
            system: 'Rank' 或 'Percentage'
            judges_save: 是否启用评委救人机制
            track_celebrity: 特定追踪的选手名称
        Returns:
            elimination_log: {选手: 淘汰周次}
            fii: 粉丝影响力指数
            tracked_survival: 追踪选手的存活周数 (若指定)
        """
        # 初始化赛季
        season_data = self.week_data[self.week_data['season'] == season]
        season_weeks = sorted(season_data['week'].unique())
        if len(season_weeks) == 0:
            return {}, np.nan, 0
            
        active = season_data[season_data['week'] == season_weeks[0]]['celebrity_name'].unique().tolist()
        
        elimination_log = {}
        fii_history = []
        tracked_survival = 0

        for w in season_weeks:
            if len(active) <= 1: 
                break

            # A. 获取分数
            j_scores = {c: self.jsg.get_score(season, w, c, self.week_data) for c in active}
            f_votes = self.get_fan_votes(season, w, active)

            # B. 计算系统得分与排名
            if system == 'Rank':
                j_rank = {c: r for c, r in zip(active, rankdata([-j_scores[c] for c in active], method='min'))}
                f_rank = {c: r for c, r in zip(active, rankdata([-f_votes[c] for c in active], method='min'))}
                total_scores = {c: j_rank[c] + f_rank[c] for c in active}
                ranked_list = sorted(active, key=lambda x: total_scores[x])  # 升序，低分更好

                # FII (Kendall Tau)
                sys_ranks = [total_scores[c] for c in active]
                fan_ranks = [f_rank[c] for c in active]
                tau, _ = kendalltau(sys_ranks, fan_ranks)
                fii_history.append(tau)

            elif system == 'Percentage':
                total_j = sum(j_scores.values())
                j_pct = {c: s / total_j for c, s in j_scores.items()}
                total_scores = {c: 0.5 * j_pct[c] + 0.5 * f_votes[c] for c in active}
                ranked_list = sorted(active, key=lambda x: total_scores[x], reverse=True)  # 降序，高分更好

                # FII
                sys_ranks = rankdata([-total_scores[c] for c in active])
                fan_ranks = rankdata([-f_votes[c] for c in active])
                tau, _ = kendalltau(sys_ranks, fan_ranks)
                fii_history.append(tau)

            # C. 淘汰逻辑
            bottom_2 = ranked_list[-2:] if len(ranked_list) >= 2 else ranked_list

            if judges_save and len(bottom_2) == 2:
                c1, c2 = bottom_2[0], bottom_2[1]
                delta = 0.5  # 犹豫阈值
                if abs(j_scores[c1] - j_scores[c2]) > delta:
                    eliminated = c2 if j_scores[c1] > j_scores[c2] else c1
                else:
                    eliminated = c2  # 差异不大，维持原判
            else:
                eliminated = ranked_list[-1]

            # D. 更新追踪
            if track_celebrity and track_celebrity in active:
                tracked_survival = w

            if eliminated in active:
                active.remove(eliminated)
                elimination_log[eliminated] = w

        # 记录冠军
        if active: 
            elimination_log[active[0]] = season_weeks[-1] + 1
            if track_celebrity and track_celebrity in active:
                tracked_survival = season_weeks[-1] + 1

        return elimination_log, np.nanmean(fii_history), tracked_survival


# ==========================================
# 3. 全赛季分析
# ==========================================
def run_all_seasons_analysis(sim, seasons):
    """对所有赛季运行双轨仿真并计算FII、公平性、稳健性"""
    results = []
    
    print("运行全赛季双轨反事实仿真...")
    for s in tqdm(seasons, desc="Seasons"):
        fii_rank_list = []
        fii_pct_list = []
        champion_rank = []
        champion_pct = []
        flip_count = 0
        
        for _ in range(N_SIMULATIONS):
            # 运行两种赛制
            log_r, fii_r, _ = sim.run_season(s, 'Rank', False)
            log_p, fii_p, _ = sim.run_season(s, 'Percentage', False)
            
            fii_rank_list.append(fii_r)
            fii_pct_list.append(fii_p)
            
            # 记录冠军
            if log_r:
                champ_r = max(log_r, key=log_r.get)
                champion_rank.append(champ_r)
            if log_p:
                champ_p = max(log_p, key=log_p.get)
                champion_pct.append(champ_p)
                
            # 统计翻转
            if log_r and log_p:
                if max(log_r, key=log_r.get) != max(log_p, key=log_p.get):
                    flip_count += 1
        
        results.append({
            'season': s,
            'FII_Rank_mean': np.nanmean(fii_rank_list),
            'FII_Rank_std': np.nanstd(fii_rank_list),
            'FII_Pct_mean': np.nanmean(fii_pct_list),
            'FII_Pct_std': np.nanstd(fii_pct_list),
            'Champion_Flip_Rate': flip_count / N_SIMULATIONS,
            'FII_Diff': np.nanmean(fii_pct_list) - np.nanmean(fii_rank_list)
        })
    
    return pd.DataFrame(results)


# ==========================================
# 4. 争议选手分析
# ==========================================
def analyze_controversy_stars(sim, stars_list):
    """对四位争议选手进行命运分叉分析"""
    results = []
    
    print("\n分析四位争议选手...")
    for star in tqdm(stars_list, desc="Stars"):
        season = star['season']
        name = star['name']
        
        survival_rank = []
        survival_pct = []
        survival_rank_save = []
        
        for _ in range(N_SIMULATIONS):
            _, _, surv_r = sim.run_season(season, 'Rank', False, track_celebrity=name)
            _, _, surv_p = sim.run_season(season, 'Percentage', False, track_celebrity=name)
            _, _, surv_rs = sim.run_season(season, 'Rank', True, track_celebrity=name)
            
            survival_rank.append(surv_r)
            survival_pct.append(surv_p)
            survival_rank_save.append(surv_rs)
        
        # 计算命运分叉度
        divergence = np.abs(np.array(survival_rank) - np.array(survival_pct))
        
        results.append({
            'season': season,
            'celebrity_name': name,
            'note': star['note'],
            'survival_Rank_mean': np.mean(survival_rank),
            'survival_Rank_std': np.std(survival_rank),
            'survival_Pct_mean': np.mean(survival_pct),
            'survival_Pct_std': np.std(survival_pct),
            'survival_RankSave_mean': np.mean(survival_rank_save),
            'fate_divergence_mean': np.mean(divergence),
            'fate_divergence_max': np.max(divergence),
            'judges_save_impact': np.mean(survival_rank_save) - np.mean(survival_rank)
        })
    
    return pd.DataFrame(results)


# ==========================================
# 5. AHP决策层
# ==========================================
def run_ahp_analysis(season_results, sim, seasons):
    """使用AHP方法推荐未来赛制"""
    
    print("\n运行AHP决策分析...")
    
    # 计算公平性 (与评委最终排名的相关性)
    fair_rank_list = []
    fair_pct_list = []
    robust_rank = []
    robust_pct = []
    
    for s in tqdm(seasons[:10], desc="Fairness & Robustness"):  # 采样前10个赛季
        for _ in range(20):
            log_r, _, _ = sim.run_season(s, 'Rank', False)
            log_p, _, _ = sim.run_season(s, 'Percentage', False)
            
            # 模拟扰动后的结果翻转率
            log_r2, _, _ = sim.run_season(s, 'Rank', False)
            log_p2, _, _ = sim.run_season(s, 'Percentage', False)
            
            if log_r and log_r2:
                if max(log_r, key=log_r.get) != max(log_r2, key=log_r2.get):
                    robust_rank.append(0)
                else:
                    robust_rank.append(1)
            if log_p and log_p2:
                if max(log_p, key=log_p.get) != max(log_p2, key=log_p2.get):
                    robust_pct.append(0)
                else:
                    robust_pct.append(1)
    
    # 公平性得分 (基于FII均值的反向：FII越低表示越偏向评委)
    avg_fii_rank = season_results['FII_Rank_mean'].mean()
    avg_fii_pct = season_results['FII_Pct_mean'].mean()
    
    # 公平性 = 1 - FII偏向粉丝程度 (越倾向评委越公平)
    fair_rank = (1 - avg_fii_rank + 1) / 2  # 归一化到 [0,1]
    fair_pct = (1 - avg_fii_pct + 1) / 2
    
    # 稳健性
    robust_rank_score = np.mean(robust_rank) if robust_rank else 0.5
    robust_pct_score = np.mean(robust_pct) if robust_pct else 0.5
    
    # AHP权重 (假设公平性:稳健性 = 2:1)
    a = 2  # 重要性比
    w_fair = a / (1 + a)  # 0.667
    w_robust = 1 / (1 + a)  # 0.333
    
    # 综合得分
    score_rank = w_fair * fair_rank + w_robust * robust_rank_score
    score_pct = w_fair * fair_pct + w_robust * robust_pct_score
    
    # Judges' Save 增量评估
    # 运行带Save的仿真
    flip_with_save = 0
    for s in seasons[:10]:
        for _ in range(10):
            log_r, _, _ = sim.run_season(s, 'Rank', False)
            log_rs, _, _ = sim.run_season(s, 'Rank', True)
            if log_r and log_rs:
                if max(log_r, key=log_r.get) != max(log_rs, key=log_rs.get):
                    flip_with_save += 1
    save_intervention_rate = flip_with_save / (len(seasons[:10]) * 10)
    
    ahp_results = {
        'Fair_Rank': fair_rank,
        'Fair_Pct': fair_pct,
        'Robust_Rank': robust_rank_score,
        'Robust_Pct': robust_pct_score,
        'w_Fair': w_fair,
        'w_Robust': w_robust,
        'Score_Rank': score_rank,
        'Score_Pct': score_pct,
        'Recommended': 'Rank' if score_rank > score_pct else 'Percentage',
        'JudgesSave_Intervention_Rate': save_intervention_rate,
        'Recommend_JudgesSave': save_intervention_rate > 0.05  # 若干预率>5%则推荐
    }
    
    return ahp_results


# ==========================================
# 6. 可视化生成
# ==========================================
def generate_visualizations(season_results, star_results, ahp_results, output_dir):
    """生成所有问题二可视化图表"""
    
    print("\n生成可视化图表...")
    
    # 1. 全赛季FII对比
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(season_results))
    width = 0.35
    ax.bar(x - width/2, season_results['FII_Rank_mean'], width, 
           yerr=season_results['FII_Rank_std'], label='Rank', color=COLORS[0], capsize=2)
    ax.bar(x + width/2, season_results['FII_Pct_mean'], width, 
           yerr=season_results['FII_Pct_std'], label='Percentage', color=COLORS[2], capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(season_results['season'].astype(int), fontsize=9, rotation=45)
    ax.set_xlabel('Season', fontweight='bold')
    ax.set_ylabel('Fan Influence Index (FII)', fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(frameon=False)
    clean_plot(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_FII_All_Seasons.png'))
    plt.close()
    
    # 2. FII差异分布
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(season_results['season'], season_results['FII_Diff'], 
           color=[COLORS[2] if v > 0 else COLORS[0] for v in season_results['FII_Diff']])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Season', fontweight='bold')
    ax.set_ylabel('FII Difference (Pct - Rank)', fontweight='bold')
    clean_plot(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_FII_Difference.png'))
    plt.close()
    
    # 3. 争议选手命运分叉雷达图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(star_results))
    width = 0.25
    ax.bar(x - width, star_results['survival_Rank_mean'], width, 
           yerr=star_results['survival_Rank_std'], label='Rank', color=COLORS[0], capsize=3)
    ax.bar(x, star_results['survival_Pct_mean'], width, 
           yerr=star_results['survival_Pct_std'], label='Percentage', color=COLORS[2], capsize=3)
    ax.bar(x + width, star_results['survival_RankSave_mean'], width, 
           label='Rank+Save', color=COLORS[3], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['celebrity_name']}\n(S{r['season']})" for _, r in star_results.iterrows()], fontsize=10)
    ax.set_ylabel('Average Survival Week', fontweight='bold')
    ax.legend(frameon=False)
    clean_plot(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_Controversy_Stars_Survival.png'))
    plt.close()
    
    # 4. 命运分叉度
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(star_results['celebrity_name'], star_results['fate_divergence_mean'], 
                   color=COLORS[2], xerr=star_results['fate_divergence_max']/2, capsize=3)
    ax.set_xlabel('Fate Divergence (Weeks)', fontweight='bold')
    clean_plot(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_Fate_Divergence.png'))
    plt.close()
    
    # 5. AHP决策雷达图
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ['Rank', 'Percentage']
    fair_scores = [ahp_results['Fair_Rank'], ahp_results['Fair_Pct']]
    robust_scores = [ahp_results['Robust_Rank'], ahp_results['Robust_Pct']]
    total_scores = [ahp_results['Score_Rank'], ahp_results['Score_Pct']]
    
    x = np.arange(2)
    width = 0.25
    ax.bar(x - width, fair_scores, width, label='Fairness', color=COLORS[0])
    ax.bar(x, robust_scores, width, label='Robustness', color=COLORS[1])
    ax.bar(x + width, total_scores, width, label='AHP Score', color=COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    clean_plot(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_AHP_Decision.png'))
    plt.close()
    
    # 6. 冠军翻转率
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(season_results['season'], season_results['Champion_Flip_Rate'] * 100, 
           color=COLORS[2])
    ax.set_xlabel('Season', fontweight='bold')
    ax.set_ylabel('Champion Flip Rate (%)', fontweight='bold')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% Threshold')
    ax.legend(frameon=False)
    clean_plot(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_Champion_Flip_Rate.png'))
    plt.close()
    
    print(f"   所有图表已保存到 {output_dir}/")


# ==========================================
# 7. 主程序
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print("MCM Problem C - 问题二：双轨动态反事实演化模型")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(FILE_Q1_SOL):
        print(f"错误: 找不到问题一输出文件 {FILE_Q1_SOL}")
        exit(1)
    if not os.path.exists(FILE_WEEK_DATA):
        print(f"错误: 找不到周数据文件 {FILE_WEEK_DATA}")
        exit(1)
    
    # 加载数据
    print("\n加载数据...")
    q1_df = pd.read_csv(FILE_Q1_SOL)
    week_df = pd.read_csv(FILE_WEEK_DATA)
    print(f"   问题一数据: {len(q1_df)} 行")
    print(f"   周数据: {len(week_df)} 行")
    
    # 创建仿真器
    sim = CounterfactualSimulator(q1_df, week_df)
    
    # 获取所有赛季
    seasons = sorted(week_df['season'].unique())
    print(f"   共 {len(seasons)} 个赛季")
    
    # 1. 全赛季分析
    season_results = run_all_seasons_analysis(sim, seasons)
    season_results.to_csv(os.path.join(OUTPUT_DIR, 'season_fii_all.csv'), index=False)
    print(f"\n全赛季FII结果已保存")
    
    # 2. 争议选手分析
    star_results = analyze_controversy_stars(sim, CONTROVERSY_STARS)
    star_results.to_csv(os.path.join(OUTPUT_DIR, 'controversy_stars.csv'), index=False)
    print(f"争议选手分析已保存")
    
    # 3. AHP决策
    ahp_results = run_ahp_analysis(season_results, sim, seasons)
    ahp_df = pd.DataFrame([ahp_results])
    ahp_df.to_csv(os.path.join(OUTPUT_DIR, 'ahp_decision.csv'), index=False)
    print(f"AHP决策结果已保存")
    
    # 4. 可视化
    generate_visualizations(season_results, star_results, ahp_results, OUTPUT_DIR)
    
    # 5. 打印摘要
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)
    print(f"\n【全赛季FII统计】")
    print(f"   Rank制平均FII: {season_results['FII_Rank_mean'].mean():.4f}")
    print(f"   Percentage制平均FII: {season_results['FII_Pct_mean'].mean():.4f}")
    print(f"   结论: {'Percentage更偏向粉丝' if season_results['FII_Pct_mean'].mean() > season_results['FII_Rank_mean'].mean() else 'Rank更偏向粉丝'}")
    
    print(f"\n【争议选手命运分叉】")
    for _, row in star_results.iterrows():
        print(f"   {row['celebrity_name']} (S{row['season']}): 分叉度 {row['fate_divergence_mean']:.2f} 周")
    
    print(f"\n【AHP推荐】")
    print(f"   推荐赛制: {ahp_results['Recommended']}")
    print(f"   Rank综合得分: {ahp_results['Score_Rank']:.4f}")
    print(f"   Percentage综合得分: {ahp_results['Score_Pct']:.4f}")
    print(f"   是否推荐Judges' Save: {'是' if ahp_results['Recommend_JudgesSave'] else '否'}")
    print(f"   Judges' Save干预率: {ahp_results['JudgesSave_Intervention_Rate']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("问题二分析完成！")
    print("=" * 60)