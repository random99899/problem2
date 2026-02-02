"""
Problem 2 Deep Analysis - 深度统计分析与可视化
为论文提供充分的数据支撑和图表佐证
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 配置
# ==========================================
INPUT_DIR = 'problem2'
OUTPUT_DIR = 'problem2'
COLORS = ['#56a6c7', '#98afba', '#bc8eb7', '#b7a6b5']

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
# 1. 加载数据
# ==========================================
print("加载问题二结果数据...")
season_df = pd.read_csv(os.path.join(INPUT_DIR, 'season_fii_all.csv'))
stars_df = pd.read_csv(os.path.join(INPUT_DIR, 'controversy_stars.csv'))
ahp_df = pd.read_csv(os.path.join(INPUT_DIR, 'ahp_decision.csv'))

# 加载问题一数据用于交叉分析
q1_df = pd.read_csv('problem1/problem1_detailed_solution.csv')
week_df = pd.read_csv('df_week_level.csv')

print(f"   全赛季数据: {len(season_df)} 行")
print(f"   争议选手数据: {len(stars_df)} 行")

# ==========================================
# 2. 详细统计量计算
# ==========================================
print("\n计算详细统计量...")

stats_results = {}

# 2.1 FII 统计分析
fii_rank = season_df['FII_Rank_mean'].values
fii_pct = season_df['FII_Pct_mean'].values

stats_results['FII_Rank'] = {
    'mean': np.mean(fii_rank),
    'std': np.std(fii_rank),
    'var': np.var(fii_rank),
    'median': np.median(fii_rank),
    'min': np.min(fii_rank),
    'max': np.max(fii_rank),
    'range': np.max(fii_rank) - np.min(fii_rank),
    'iqr': np.percentile(fii_rank, 75) - np.percentile(fii_rank, 25),
    'cv': np.std(fii_rank) / np.mean(fii_rank),  # 变异系数
    'skewness': stats.skew(fii_rank),
    'kurtosis': stats.kurtosis(fii_rank),
    'ci_95_lower': np.mean(fii_rank) - 1.96 * np.std(fii_rank) / np.sqrt(len(fii_rank)),
    'ci_95_upper': np.mean(fii_rank) + 1.96 * np.std(fii_rank) / np.sqrt(len(fii_rank))
}

stats_results['FII_Pct'] = {
    'mean': np.mean(fii_pct),
    'std': np.std(fii_pct),
    'var': np.var(fii_pct),
    'median': np.median(fii_pct),
    'min': np.min(fii_pct),
    'max': np.max(fii_pct),
    'range': np.max(fii_pct) - np.min(fii_pct),
    'iqr': np.percentile(fii_pct, 75) - np.percentile(fii_pct, 25),
    'cv': np.std(fii_pct) / np.mean(fii_pct),
    'skewness': stats.skew(fii_pct),
    'kurtosis': stats.kurtosis(fii_pct),
    'ci_95_lower': np.mean(fii_pct) - 1.96 * np.std(fii_pct) / np.sqrt(len(fii_pct)),
    'ci_95_upper': np.mean(fii_pct) + 1.96 * np.std(fii_pct) / np.sqrt(len(fii_pct))
}

# 2.2 配对 t 检验
t_stat, p_value = ttest_ind(fii_pct, fii_rank)
stats_results['Paired_ttest'] = {'t_statistic': t_stat, 'p_value': p_value}

# 2.3 Mann-Whitney U 检验 (非参数)
u_stat, u_pvalue = mannwhitneyu(fii_pct, fii_rank, alternative='greater')
stats_results['MannWhitney'] = {'U_statistic': u_stat, 'p_value': u_pvalue}

# 2.4 效应量 Cohen's d
cohens_d = (np.mean(fii_pct) - np.mean(fii_rank)) / np.sqrt((np.var(fii_pct) + np.var(fii_rank)) / 2)
stats_results['Effect_Size'] = {'Cohens_d': cohens_d}

# 2.5 FII差异的分布统计
fii_diff = season_df['FII_Diff'].values
stats_results['FII_Diff'] = {
    'mean': np.mean(fii_diff),
    'std': np.std(fii_diff),
    'positive_count': np.sum(fii_diff > 0),  # Pct更偏向粉丝的赛季数
    'negative_count': np.sum(fii_diff < 0),
    'positive_ratio': np.mean(fii_diff > 0)
}

# 2.6 冠军翻转率统计
flip_rate = season_df['Champion_Flip_Rate'].values
stats_results['Champion_Flip'] = {
    'mean': np.mean(flip_rate),
    'std': np.std(flip_rate),
    'max': np.max(flip_rate),
    'seasons_with_flip': np.sum(flip_rate > 0.1)  # 翻转率>10%的赛季数
}

# 2.7 争议选手命运分叉统计
divergence = stars_df['fate_divergence_mean'].values
stats_results['Fate_Divergence'] = {
    'mean': np.mean(divergence),
    'std': np.std(divergence),
    'max': np.max(divergence),
    'total': np.sum(divergence)
}

# 输出统计表
stats_df = pd.DataFrame({
    'Metric': ['Mean', 'Std', 'Variance', 'Median', 'Min', 'Max', 'Range', 'IQR', 
               'CV', 'Skewness', 'Kurtosis', '95% CI Lower', '95% CI Upper'],
    'FII_Rank': [stats_results['FII_Rank'][k] for k in 
                 ['mean', 'std', 'var', 'median', 'min', 'max', 'range', 'iqr', 
                  'cv', 'skewness', 'kurtosis', 'ci_95_lower', 'ci_95_upper']],
    'FII_Pct': [stats_results['FII_Pct'][k] for k in 
                ['mean', 'std', 'var', 'median', 'min', 'max', 'range', 'iqr', 
                 'cv', 'skewness', 'kurtosis', 'ci_95_lower', 'ci_95_upper']]
})
stats_df.to_csv(os.path.join(OUTPUT_DIR, 'detailed_statistics.csv'), index=False)
print("   统计量已保存: detailed_statistics.csv")

# 假设检验汇总
hypothesis_df = pd.DataFrame([
    {'Test': 'Independent t-test', 't/U': stats_results['Paired_ttest']['t_statistic'], 
     'p-value': stats_results['Paired_ttest']['p_value'], 
     'Significant (α=0.05)': stats_results['Paired_ttest']['p_value'] < 0.05},
    {'Test': 'Mann-Whitney U', 't/U': stats_results['MannWhitney']['U_statistic'], 
     'p-value': stats_results['MannWhitney']['p_value'], 
     'Significant (α=0.05)': stats_results['MannWhitney']['p_value'] < 0.05},
    {'Test': "Cohen's d (Effect Size)", 't/U': stats_results['Effect_Size']['Cohens_d'], 
     'p-value': np.nan, 'Significant (α=0.05)': abs(stats_results['Effect_Size']['Cohens_d']) > 0.8}
])
hypothesis_df.to_csv(os.path.join(OUTPUT_DIR, 'hypothesis_tests.csv'), index=False)
print("   假设检验结果已保存: hypothesis_tests.csv")

# ==========================================
# 3. 深度可视化
# ==========================================
print("\n生成深度分析可视化...")

# 3.1 FII 散点图 (Rank vs Pct)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(season_df['FII_Rank_mean'], season_df['FII_Pct_mean'], 
           c=COLORS[0], s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
for i, row in season_df.iterrows():
    ax.annotate(f"S{int(row['season'])}", (row['FII_Rank_mean'], row['FII_Pct_mean']),
                fontsize=8, alpha=0.7, ha='center', va='bottom')
# 对角线
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Equal FII')
ax.set_xlabel('FII (Rank Method)', fontweight='bold')
ax.set_ylabel('FII (Percentage Method)', fontweight='bold')
ax.set_xlim(0.3, 0.9)
ax.set_ylim(0.6, 1.0)
ax.legend(frameon=False)
clean_plot(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '7_FII_Scatter.png'))
plt.close()

# 3.2 FII分布箱线图
fig, ax = plt.subplots(figsize=(8, 6))
data = [fii_rank, fii_pct]
bp = ax.boxplot(data, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], [COLORS[0], COLORS[2]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(['Rank', 'Percentage'], fontweight='bold')
ax.set_ylabel('Fan Influence Index (FII)', fontweight='bold')
# 添加均值点
means = [np.mean(fii_rank), np.mean(fii_pct)]
ax.scatter([1, 2], means, marker='D', color='black', s=100, zorder=5, label='Mean')
ax.legend(frameon=False)
clean_plot(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '8_FII_Boxplot.png'))
plt.close()

# 3.3 赛制阶段 × FII 热力图
season_df['Phase'] = season_df['season'].apply(
    lambda s: 'Ranking (S1-2)' if s <= 2 else ('Percentage (S3-27)' if s <= 27 else 'Judges Save (S28+)')
)
phase_fii = season_df.groupby('Phase').agg({
    'FII_Rank_mean': ['mean', 'std'],
    'FII_Pct_mean': ['mean', 'std'],
    'FII_Diff': 'mean',
    'Champion_Flip_Rate': 'mean'
}).reset_index()
phase_fii.columns = ['Phase', 'FII_Rank_mean', 'FII_Rank_std', 'FII_Pct_mean', 'FII_Pct_std', 'FII_Diff', 'Flip_Rate']
phase_fii.to_csv(os.path.join(OUTPUT_DIR, 'phase_fii_summary.csv'), index=False)

# 热力图数据准备
heatmap_data = phase_fii[['Phase', 'FII_Rank_mean', 'FII_Pct_mean', 'FII_Diff', 'Flip_Rate']].set_index('Phase')
fig, ax = plt.subplots(figsize=(10, 5))
from matplotlib.colors import LinearSegmentedColormap
custom_cmap = LinearSegmentedColormap.from_list('custom', ['#56a6c7', '#b7a6b5', '#bc8eb7'])
sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap=custom_cmap, ax=ax, linewidths=1)
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '9_Phase_Heatmap.png'))
plt.close()

# 3.4 争议选手对比雷达图 (使用极坐标)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories = ['Survival\n(Rank)', 'Survival\n(Pct)', 'Survival\n(R+Save)', 'Divergence']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, (_, star) in enumerate(stars_df.iterrows()):
    values = [
        star['survival_Rank_mean'] / 12,  # 归一化
        star['survival_Pct_mean'] / 12,
        star['survival_RankSave_mean'] / 12,
        star['fate_divergence_mean'] / 5
    ]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=star['celebrity_name'], color=COLORS[i % len(COLORS)])
    ax.fill(angles, values, alpha=0.15, color=COLORS[i % len(COLORS)])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_Stars_Radar.png'))
plt.close()

# 3.5 95%置信区间对比图
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(season_df))
ax.errorbar(x, season_df['FII_Rank_mean'], 
            yerr=1.96*season_df['FII_Rank_std']/np.sqrt(100),  # 95% CI
            fmt='o', color=COLORS[0], label='Rank (95% CI)', capsize=3, markersize=6)
ax.errorbar(x + 0.3, season_df['FII_Pct_mean'], 
            yerr=1.96*season_df['FII_Pct_std']/np.sqrt(100),
            fmt='s', color=COLORS[2], label='Percentage (95% CI)', capsize=3, markersize=6)
ax.set_xticks(x + 0.15)
ax.set_xticklabels(season_df['season'].astype(int), rotation=45, fontsize=9)
ax.set_xlabel('Season', fontweight='bold')
ax.set_ylabel('FII with 95% Confidence Interval', fontweight='bold')
ax.legend(frameon=False)
clean_plot(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '11_FII_Confidence_Interval.png'))
plt.close()

# 3.6 Judges' Save 影响分析
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(stars_df))
width = 0.4
bars1 = ax.bar(x - width/2, stars_df['survival_Rank_mean'], width, 
               label='Without Save', color=COLORS[0])
bars2 = ax.bar(x + width/2, stars_df['survival_RankSave_mean'], width, 
               label='With Save', color=COLORS[3])
# 添加影响值
for i, (_, row) in enumerate(stars_df.iterrows()):
    impact = row['judges_save_impact']
    color = 'green' if impact > 0 else 'red'
    ax.annotate(f"{impact:+.1f}", (i + width/2, row['survival_RankSave_mean'] + 0.3),
                ha='center', fontsize=10, fontweight='bold', color=color)
ax.set_xticks(x)
ax.set_xticklabels([f"{r['celebrity_name']}\n(S{r['season']})" for _, r in stars_df.iterrows()], fontsize=10)
ax.set_ylabel('Average Survival Week', fontweight='bold')
ax.legend(frameon=False)
clean_plot(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '12_Judges_Save_Impact.png'))
plt.close()

# 3.7 FII与评委分差异的相关性分析
# 计算每个赛季的评委-粉丝分差异
season_judge_fan_gap = q1_df.groupby('season').apply(
    lambda g: (g['judge_pct'] - g['est_fan_vote_pct']).abs().mean()
).reset_index(name='avg_judge_fan_gap')
merged_df = season_df.merge(season_judge_fan_gap, on='season')

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(merged_df['avg_judge_fan_gap'], merged_df['FII_Diff'], 
           c=COLORS[2], s=80, alpha=0.7, edgecolors='white')
# 添加回归线
z = np.polyfit(merged_df['avg_judge_fan_gap'], merged_df['FII_Diff'], 1)
p = np.poly1d(z)
x_line = np.linspace(merged_df['avg_judge_fan_gap'].min(), merged_df['avg_judge_fan_gap'].max(), 100)
ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7)
# 计算相关系数
corr, p_val = spearmanr(merged_df['avg_judge_fan_gap'], merged_df['FII_Diff'])
ax.text(0.05, 0.95, f'Spearman r = {corr:.3f}\np = {p_val:.4f}', 
        transform=ax.transAxes, fontsize=11, va='top')
ax.set_xlabel('Avg |Judge - Fan| Gap', fontweight='bold')
ax.set_ylabel('FII Difference (Pct - Rank)', fontweight='bold')
clean_plot(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '13_Gap_FII_Correlation.png'))
plt.close()

# 3.8 三维投影图: Season × FII_Rank × FII_Pct
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
xs = season_df['season'].values
ys = season_df['FII_Rank_mean'].values
zs = season_df['FII_Pct_mean'].values
colors_3d = [COLORS[0] if diff < 0 else COLORS[2] for diff in season_df['FII_Diff']]
ax.scatter(xs, ys, zs, c=colors_3d, s=100, alpha=0.8, edgecolors='white')
ax.set_xlabel('Season', fontweight='bold')
ax.set_ylabel('FII (Rank)', fontweight='bold')
ax.set_zlabel('FII (Percentage)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '14_3D_FII_Projection.png'))
plt.close()

# 3.9 争议选手命运分叉的瀑布图
fig, ax = plt.subplots(figsize=(10, 6))
stars_sorted = stars_df.sort_values('fate_divergence_mean', ascending=False)
y_pos = np.arange(len(stars_sorted))
bars = ax.barh(y_pos, stars_sorted['fate_divergence_mean'], color=COLORS[2], height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['celebrity_name']} (S{r['season']})" for _, r in stars_sorted.iterrows()], fontsize=11)
ax.set_xlabel('Fate Divergence (Weeks)', fontweight='bold')
# 添加数值标签
for i, (_, row) in enumerate(stars_sorted.iterrows()):
    ax.text(row['fate_divergence_mean'] + 0.1, i, f"{row['fate_divergence_mean']:.2f}", 
            va='center', fontsize=10, fontweight='bold')
ax.invert_yaxis()
clean_plot(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '15_Divergence_Waterfall.png'))
plt.close()

# 3.10 FII分布直方图对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(fii_rank, bins=15, color=COLORS[0], alpha=0.7, edgecolor='white')
axes[0].axvline(np.mean(fii_rank), color='black', linestyle='--', linewidth=2, label=f'Mean={np.mean(fii_rank):.3f}')
axes[0].set_xlabel('FII (Rank)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].legend(frameon=False)
clean_plot(axes[0])

axes[1].hist(fii_pct, bins=15, color=COLORS[2], alpha=0.7, edgecolor='white')
axes[1].axvline(np.mean(fii_pct), color='black', linestyle='--', linewidth=2, label=f'Mean={np.mean(fii_pct):.3f}')
axes[1].set_xlabel('FII (Percentage)', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].legend(frameon=False)
clean_plot(axes[1])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '16_FII_Histograms.png'))
plt.close()

# ==========================================
# 4. 论文摘要统计表
# ==========================================
print("\n生成论文摘要统计表...")

summary_table = pd.DataFrame([
    {'Category': 'FII Comparison', 'Metric': 'Mean FII (Rank)', 'Value': f"{stats_results['FII_Rank']['mean']:.4f}"},
    {'Category': 'FII Comparison', 'Metric': 'Mean FII (Percentage)', 'Value': f"{stats_results['FII_Pct']['mean']:.4f}"},
    {'Category': 'FII Comparison', 'Metric': 'FII Difference (Pct-Rank)', 'Value': f"{stats_results['FII_Diff']['mean']:.4f}"},
    {'Category': 'FII Comparison', 'Metric': 'Seasons where Pct > Rank', 'Value': f"{stats_results['FII_Diff']['positive_count']}/34 ({stats_results['FII_Diff']['positive_ratio']*100:.1f}%)"},
    {'Category': 'Statistical Test', 'Metric': 't-test p-value', 'Value': f"{stats_results['Paired_ttest']['p_value']:.2e}"},
    {'Category': 'Statistical Test', 'Metric': "Cohen's d", 'Value': f"{stats_results['Effect_Size']['Cohens_d']:.3f}"},
    {'Category': 'Champion Stability', 'Metric': 'Avg Flip Rate', 'Value': f"{stats_results['Champion_Flip']['mean']*100:.1f}%"},
    {'Category': 'Champion Stability', 'Metric': 'Max Flip Rate', 'Value': f"{stats_results['Champion_Flip']['max']*100:.1f}%"},
    {'Category': 'Controversy Stars', 'Metric': 'Avg Fate Divergence', 'Value': f"{stats_results['Fate_Divergence']['mean']:.2f} weeks"},
    {'Category': 'Controversy Stars', 'Metric': 'Max Fate Divergence', 'Value': f"{stats_results['Fate_Divergence']['max']:.2f} weeks"},
    {'Category': 'AHP Decision', 'Metric': 'Recommended System', 'Value': ahp_df['Recommended'].iloc[0]},
    {'Category': 'AHP Decision', 'Metric': 'Recommend Judges Save', 'Value': 'Yes' if ahp_df['Recommend_JudgesSave'].iloc[0] else 'No'},
])
summary_table.to_csv(os.path.join(OUTPUT_DIR, 'paper_summary_table.csv'), index=False)
print("   论文摘要表已保存: paper_summary_table.csv")

# ==========================================
# 5. 打印最终统计摘要
# ==========================================
print("\n" + "="*70)
print("问题二深度统计分析摘要")
print("="*70)

print("\n【FII对比统计】")
print(f"   Rank制FII: {stats_results['FII_Rank']['mean']:.4f} ± {stats_results['FII_Rank']['std']:.4f}")
print(f"   Percentage制FII: {stats_results['FII_Pct']['mean']:.4f} ± {stats_results['FII_Pct']['std']:.4f}")
print(f"   差异: {stats_results['FII_Diff']['mean']:.4f}")
print(f"   Percentage > Rank 的赛季: {stats_results['FII_Diff']['positive_count']}/34 ({stats_results['FII_Diff']['positive_ratio']*100:.1f}%)")

print("\n【假设检验】")
print(f"   t检验: t={stats_results['Paired_ttest']['t_statistic']:.3f}, p={stats_results['Paired_ttest']['p_value']:.2e}")
print(f"   结论: {'显著差异' if stats_results['Paired_ttest']['p_value'] < 0.05 else '无显著差异'} (α=0.05)")
print(f"   效应量 Cohen's d: {stats_results['Effect_Size']['Cohens_d']:.3f} ({'大效应' if abs(stats_results['Effect_Size']['Cohens_d']) > 0.8 else '中等效应' if abs(stats_results['Effect_Size']['Cohens_d']) > 0.5 else '小效应'})")

print("\n【冠军稳定性】")
print(f"   平均翻转率: {stats_results['Champion_Flip']['mean']*100:.1f}%")
print(f"   最大翻转率: {stats_results['Champion_Flip']['max']*100:.1f}%")

print("\n【争议选手命运分叉】")
for _, row in stars_df.iterrows():
    print(f"   {row['celebrity_name']} (S{row['season']}): 分叉 {row['fate_divergence_mean']:.2f} 周, Save影响 {row['judges_save_impact']:+.2f} 周")

print("\n" + "="*70)
print(f"共生成 {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])} 张图表")
print(f"共生成 {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')])} 个数据文件")
print("="*70)
