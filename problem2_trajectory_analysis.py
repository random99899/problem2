"""
Problem 2 争议选手深度轨迹分析
分析Bobby Bones (S27) 和 Jerry Rice (S2) 在不同赛制下的各项指标随时间变化
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 配置
# ==========================================
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
print("加载数据...")
q1_df = pd.read_csv('problem1/problem1_detailed_solution.csv')
week_df = pd.read_csv('df_week_level.csv')
season_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'season_fii_all.csv'))
ahp_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'ahp_decision.csv'))
stars_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'controversy_stars.csv'))

# ==========================================
# 2. 修改柱状图添加数据标签
# ==========================================
print("\n修改柱状图，添加数据标签...")

# 2.1 重新生成 5_AHP_Decision.png
methods = ['Rank', 'Percentage']
fair_scores = [ahp_df['Fair_Rank'].values[0], ahp_df['Fair_Pct'].values[0]]
robust_scores = [ahp_df['Robust_Rank'].values[0], ahp_df['Robust_Pct'].values[0]]
total_scores = [ahp_df['Score_Rank'].values[0], ahp_df['Score_Pct'].values[0]]

fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(2)
width = 0.25
bars1 = ax.bar(x - width, fair_scores, width, label='Fairness', color=COLORS[0])
bars2 = ax.bar(x, robust_scores, width, label='Robustness', color=COLORS[1])
bars3 = ax.bar(x + width, total_scores, width, label='AHP Score', color=COLORS[2])

# 添加数据标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(methods, fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_ylim(0, 0.85)
ax.legend(frameon=False)
clean_plot(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5_AHP_Decision.png'))
plt.close()
print("   已更新: 5_AHP_Decision.png")

# 2.2 重新生成 12_Judges_Save_Impact.png
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(stars_df))
width = 0.4
bars1 = ax.bar(x - width/2, stars_df['survival_Rank_mean'], width, 
               label='Without Save', color=COLORS[0])
bars2 = ax.bar(x + width/2, stars_df['survival_RankSave_mean'], width, 
               label='With Save', color=COLORS[3])

# 添加数据标签
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

# 添加影响值标注
for i, (_, row) in enumerate(stars_df.iterrows()):
    impact = row['judges_save_impact']
    color = 'green' if impact > 0 else 'red'
    ax.annotate(f"Δ{impact:+.1f}", (i, max(row['survival_Rank_mean'], row['survival_RankSave_mean']) + 1),
                ha='center', fontsize=9, fontweight='bold', color=color)

ax.set_xticks(x)
ax.set_xticklabels([f"{r['celebrity_name']}\n(S{r['season']})" for _, r in stars_df.iterrows()], fontsize=10)
ax.set_ylabel('Average Survival Week', fontweight='bold')
ax.set_ylim(0, 12)
ax.legend(frameon=False)
clean_plot(ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '12_Judges_Save_Impact.png'))
plt.close()
print("   已更新: 12_Judges_Save_Impact.png")

# 2.3 重新生成 16_FII_Histograms.png
fii_rank = season_df['FII_Rank_mean'].values
fii_pct = season_df['FII_Pct_mean'].values

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Rank直方图
n1, bins1, patches1 = axes[0].hist(fii_rank, bins=12, color=COLORS[0], alpha=0.7, edgecolor='white')
for i, (count, patch) in enumerate(zip(n1, patches1)):
    if count > 0:
        axes[0].annotate(f'{int(count)}', xy=(patch.get_x() + patch.get_width()/2, count),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
axes[0].axvline(np.mean(fii_rank), color='black', linestyle='--', linewidth=2, label=f'Mean={np.mean(fii_rank):.3f}')
axes[0].set_xlabel('FII (Rank)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].legend(frameon=False)
clean_plot(axes[0])

# Percentage直方图
n2, bins2, patches2 = axes[1].hist(fii_pct, bins=12, color=COLORS[2], alpha=0.7, edgecolor='white')
for i, (count, patch) in enumerate(zip(n2, patches2)):
    if count > 0:
        axes[1].annotate(f'{int(count)}', xy=(patch.get_x() + patch.get_width()/2, count),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
axes[1].axvline(np.mean(fii_pct), color='black', linestyle='--', linewidth=2, label=f'Mean={np.mean(fii_pct):.3f}')
axes[1].set_xlabel('FII (Percentage)', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].legend(frameon=False)
clean_plot(axes[1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '16_FII_Histograms.png'))
plt.close()
print("   已更新: 16_FII_Histograms.png")

# ==========================================
# 3. 争议选手深度轨迹分析
# ==========================================
print("\n进行争议选手深度轨迹分析...")

# 定义要分析的选手
TARGET_STARS = [
    {'season': 27, 'name': 'Bobby Bones', 'note': 'won despite consistently low judges scores'},
    {'season': 2, 'name': 'Jerry Rice', 'note': 'runner up despite lowest judges scores in 5 weeks'},
]

def simulate_trajectory(season_data, q1_data, celebrity_name, system, judges_save=False):
    """
    模拟单个选手在指定赛制下的完整轨迹
    返回每周的各项指标
    """
    weeks = sorted(season_data['week'].unique())
    active = season_data[season_data['week'] == weeks[0]]['celebrity_name'].unique().tolist()
    
    trajectory = []
    
    for w in weeks:
        if celebrity_name not in active:
            break
            
        week_data = season_data[season_data['week'] == w]
        q1_week = q1_data[(q1_data['week'] == w)]
        
        # 获取当前选手数据
        celeb_week = week_data[week_data['celebrity_name'] == celebrity_name]
        celeb_q1 = q1_week[q1_week['celebrity_name'] == celebrity_name]
        
        if celeb_week.empty or celeb_q1.empty:
            continue
        
        # 评委分
        if 'avg_score' in week_data.columns:
            score_col = 'avg_score'
        else:
            score_col = 'total_score'
        
        j_score = celeb_week[score_col].values[0]
        f_vote = celeb_q1['est_fan_vote_pct'].values[0]
        
        # 计算当周所有选手得分
        active_data = week_data[week_data['celebrity_name'].isin(active)]
        active_q1 = q1_week[q1_week['celebrity_name'].isin(active)]
        
        j_scores = dict(zip(active_data['celebrity_name'], active_data[score_col]))
        f_votes = {}
        for _, row in active_q1.iterrows():
            f_votes[row['celebrity_name']] = row['est_fan_vote_pct']
        
        # 补全缺失
        for c in active:
            if c not in j_scores: j_scores[c] = np.mean(list(j_scores.values()))
            if c not in f_votes: f_votes[c] = 1.0 / len(active)
        
        # 归一化
        total_f = sum(f_votes.values())
        f_votes = {k: v/total_f for k, v in f_votes.items()}
        
        # 计算排名
        j_rank_list = rankdata([-j_scores[c] for c in active], method='min')
        f_rank_list = rankdata([-f_votes[c] for c in active], method='min')
        j_rank = {c: r for c, r in zip(active, j_rank_list)}
        f_rank = {c: r for c, r in zip(active, f_rank_list)}
        
        # 计算综合得分
        if system == 'Rank':
            total_scores = {c: j_rank[c] + f_rank[c] for c in active}
            ranked_list = sorted(active, key=lambda x: total_scores[x])
            celeb_total = total_scores[celebrity_name]
            sys_rank = list(ranked_list).index(celebrity_name) + 1
        else:  # Percentage
            total_j = sum(j_scores.values())
            j_pct = {c: s/total_j for c, s in j_scores.items()}
            total_scores = {c: 0.5 * j_pct[c] + 0.5 * f_votes[c] for c in active}
            ranked_list = sorted(active, key=lambda x: total_scores[x], reverse=True)
            celeb_total = total_scores[celebrity_name]
            sys_rank = list(ranked_list).index(celebrity_name) + 1
        
        # 评委排名
        celeb_j_rank = j_rank[celebrity_name]
        # 粉丝排名
        celeb_f_rank = f_rank[celebrity_name]
        
        # 距离淘汰的安全余量（排名差）
        safety_margin = len(active) - sys_rank
        
        # 记录轨迹
        trajectory.append({
            'week': w,
            'n_contestants': len(active),
            'judge_score': j_score,
            'fan_vote_pct': f_vote,
            'judge_rank': celeb_j_rank,
            'fan_rank': celeb_f_rank,
            'system_rank': sys_rank,
            'total_score': celeb_total,
            'safety_margin': safety_margin,
            'is_bottom_2': sys_rank >= len(active) - 1
        })
        
        # 淘汰逻辑
        bottom_2 = ranked_list[-2:] if len(ranked_list) >= 2 else ranked_list
        
        if judges_save and len(bottom_2) == 2:
            c1, c2 = bottom_2[0], bottom_2[1]
            if abs(j_scores.get(c1, 0) - j_scores.get(c2, 0)) > 0.5:
                eliminated = c2 if j_scores.get(c1, 0) > j_scores.get(c2, 0) else c1
            else:
                eliminated = c2
        else:
            eliminated = ranked_list[-1]
        
        if eliminated in active:
            active.remove(eliminated)
    
    return pd.DataFrame(trajectory)


# 分析每位争议选手
all_trajectories = {}

for star in TARGET_STARS:
    season = star['season']
    name = star['name']
    print(f"\n分析 {name} (Season {season})...")
    
    season_data = week_df[week_df['season'] == season]
    q1_season = q1_df[q1_df['season'] == season]
    
    # 三种赛制的轨迹
    traj_rank = simulate_trajectory(season_data, q1_season, name, 'Rank', False)
    traj_pct = simulate_trajectory(season_data, q1_season, name, 'Percentage', False)
    traj_rank_save = simulate_trajectory(season_data, q1_season, name, 'Rank', True)
    
    traj_rank['system'] = 'Rank'
    traj_pct['system'] = 'Percentage'
    traj_rank_save['system'] = 'Rank+Save'
    
    combined = pd.concat([traj_rank, traj_pct, traj_rank_save], ignore_index=True)
    all_trajectories[name] = combined
    
    # 保存轨迹数据
    combined.to_csv(os.path.join(OUTPUT_DIR, f'trajectory_{name.replace(" ", "_")}_S{season}.csv'), index=False)

# ==========================================
# 4. 生成争议选手轨迹可视化
# ==========================================
print("\n生成争议选手轨迹可视化...")

for star in TARGET_STARS:
    name = star['name']
    season = star['season']
    traj = all_trajectories[name]
    
    # 4.1 综合排名随时间变化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) 系统排名随时间变化
    ax1 = axes[0, 0]
    for sys, color in zip(['Rank', 'Percentage', 'Rank+Save'], [COLORS[0], COLORS[2], COLORS[3]]):
        data = traj[traj['system'] == sys]
        ax1.plot(data['week'], data['system_rank'], 'o-', color=color, label=sys, linewidth=2, markersize=8)
    ax1.invert_yaxis()  # 排名越低越好
    ax1.set_xlabel('Week', fontweight='bold')
    ax1.set_ylabel('System Rank (lower is better)', fontweight='bold')
    ax1.legend(frameon=False)
    ax1.axhline(y=traj['n_contestants'].max() - 1, color='red', linestyle='--', alpha=0.5, label='Bottom 2 Line')
    clean_plot(ax1)
    
    # (2) 评委排名 vs 粉丝排名
    ax2 = axes[0, 1]
    data_rank = traj[traj['system'] == 'Rank']
    ax2.plot(data_rank['week'], data_rank['judge_rank'], 's--', color=COLORS[1], label='Judge Rank', linewidth=2, markersize=8)
    ax2.plot(data_rank['week'], data_rank['fan_rank'], 'o-', color=COLORS[0], label='Fan Rank', linewidth=2, markersize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Week', fontweight='bold')
    ax2.set_ylabel('Rank', fontweight='bold')
    ax2.legend(frameon=False)
    # 标注排名差距
    for _, row in data_rank.iterrows():
        gap = row['judge_rank'] - row['fan_rank']
        if abs(gap) >= 2:
            ax2.annotate(f"Δ{int(gap)}", (row['week'], (row['judge_rank'] + row['fan_rank'])/2),
                        fontsize=9, color='red' if gap > 0 else 'green', fontweight='bold')
    clean_plot(ax2)
    
    # (3) 安全余量随时间变化
    ax3 = axes[1, 0]
    for sys, color in zip(['Rank', 'Percentage', 'Rank+Save'], [COLORS[0], COLORS[2], COLORS[3]]):
        data = traj[traj['system'] == sys]
        ax3.fill_between(data['week'], 0, data['safety_margin'], alpha=0.3, color=color)
        ax3.plot(data['week'], data['safety_margin'], 'o-', color=color, label=sys, linewidth=2, markersize=6)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Elimination Zone')
    ax3.set_xlabel('Week', fontweight='bold')
    ax3.set_ylabel('Safety Margin (ranks from elimination)', fontweight='bold')
    ax3.legend(frameon=False, loc='upper right')
    clean_plot(ax3)
    
    # (4) 评委分 vs 粉丝份额散点图
    ax4 = axes[1, 1]
    data_rank = traj[traj['system'] == 'Rank']
    scatter = ax4.scatter(data_rank['judge_score'], data_rank['fan_vote_pct'] * 100, 
                          c=data_rank['week'], cmap='viridis', s=150, edgecolors='white', linewidth=2)
    plt.colorbar(scatter, ax=ax4, label='Week')
    ax4.set_xlabel('Judge Score', fontweight='bold')
    ax4.set_ylabel('Fan Vote Share (%)', fontweight='bold')
    # 添加周次标签
    for _, row in data_rank.iterrows():
        ax4.annotate(f"W{int(row['week'])}", (row['judge_score'], row['fan_vote_pct']*100),
                    fontsize=8, ha='left', va='bottom')
    clean_plot(ax4)
    
    plt.suptitle(f"{name} (Season {season}) - Trajectory Analysis", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'17_Trajectory_{name.replace(" ", "_")}_S{season}.png'))
    plt.close()
    print(f"   已保存: 17_Trajectory_{name.replace(' ', '_')}_S{season}.png")

# 4.2 两位选手对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, star in enumerate(TARGET_STARS):
    name = star['name']
    traj = all_trajectories[name]
    ax = axes[i]
    
    # 绘制三种赛制下的排名轨迹
    for sys, color, marker in zip(['Rank', 'Percentage', 'Rank+Save'], 
                                   [COLORS[0], COLORS[2], COLORS[3]], ['o', 's', '^']):
        data = traj[traj['system'] == sys]
        ax.plot(data['week'], data['system_rank'], marker=marker, linestyle='-', 
               color=color, label=sys, linewidth=2.5, markersize=10)
    
    ax.invert_yaxis()
    ax.set_xlabel('Week', fontweight='bold')
    ax.set_ylabel('System Rank', fontweight='bold')
    ax.set_title(f"{name} (S{star['season']})", fontweight='bold', fontsize=14)
    ax.legend(frameon=False)
    
    # 标注淘汰周/最终名次
    for sys in ['Rank', 'Percentage', 'Rank+Save']:
        data = traj[traj['system'] == sys]
        if not data.empty:
            last_week = data['week'].max()
            last_rank = data[data['week'] == last_week]['system_rank'].values[0]
            ax.annotate(f"W{int(last_week)}", (last_week, last_rank),
                       fontsize=10, fontweight='bold', ha='left')
    clean_plot(ax)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '18_Stars_Comparison.png'))
plt.close()
print("   已保存: 18_Stars_Comparison.png")

# ==========================================
# 5. 统计分析汇总
# ==========================================
print("\n生成争议选手统计汇总...")

summary_stats = []
for star in TARGET_STARS:
    name = star['name']
    season = star['season']
    traj = all_trajectories[name]
    
    for sys in ['Rank', 'Percentage', 'Rank+Save']:
        data = traj[traj['system'] == sys]
        if not data.empty:
            summary_stats.append({
                'Celebrity': name,
                'Season': season,
                'System': sys,
                'Weeks_Survived': len(data),
                'Avg_System_Rank': data['system_rank'].mean(),
                'Avg_Judge_Rank': data['judge_rank'].mean(),
                'Avg_Fan_Rank': data['fan_rank'].mean(),
                'Judge_Fan_Rank_Gap': (data['judge_rank'] - data['fan_rank']).mean(),
                'Weeks_in_Bottom2': data['is_bottom_2'].sum(),
                'Min_Safety_Margin': data['safety_margin'].min(),
                'Avg_Safety_Margin': data['safety_margin'].mean()
            })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'controversy_stars_trajectory_stats.csv'), index=False)
print("   已保存: controversy_stars_trajectory_stats.csv")

# ==========================================
# 6. 打印分析摘要
# ==========================================
print("\n" + "="*70)
print("争议选手深度轨迹分析摘要")
print("="*70)

for star in TARGET_STARS:
    name = star['name']
    season = star['season']
    print(f"\n【{name} (Season {season})】")
    print(f"   争议描述: {star['note']}")
    
    for sys in ['Rank', 'Percentage', 'Rank+Save']:
        stats = summary_df[(summary_df['Celebrity'] == name) & (summary_df['System'] == sys)]
        if not stats.empty:
            s = stats.iloc[0]
            print(f"\n   {sys}制:")
            print(f"      存活周数: {s['Weeks_Survived']}")
            print(f"      平均系统排名: {s['Avg_System_Rank']:.2f}")
            print(f"      评委-粉丝排名差: {s['Judge_Fan_Rank_Gap']:+.2f} (正值=评委排名更低)")
            print(f"      进入Bottom2次数: {int(s['Weeks_in_Bottom2'])}")
            print(f"      最小安全余量: {int(s['Min_Safety_Margin'])}")

print("\n" + "="*70)
print("分析完成！")
print("="*70)

# ==========================================
# 分析目的说明
# ==========================================
purpose_text = """
========================================
争议选手深度轨迹分析的目的与意义
========================================

1. **揭示赛制对"高人气低专业"选手的差异化影响**
   - 通过追踪Bobby Bones和Jerry Rice在不同赛制下的排名轨迹，
     可以定量展示Percentage制如何放大粉丝优势，而Rank制如何压缩这种优势。

2. **可视化"评委-粉丝分歧"的动态演化**
   - Judge Rank vs Fan Rank图表直观展示每周的评委-粉丝意见分歧，
     为"争议性淘汰"提供具体周次的证据。

3. **量化安全余量(Safety Margin)的概念**
   - 安全余量图展示选手距离淘汰有多"危险"，
     证明Judges' Save机制在关键周次的干预效果。

4. **为AHP推荐提供微观证据**
   - 宏观统计显示Rank制更公平，
     微观轨迹分析则通过具体案例解释"为什么Rank制更公平"——
     它能防止评委分持续垫底的选手仅凭人气走到决赛。

5. **论文论证的因果链条**
   问题陈述 → 宏观统计差异 → 微观轨迹验证 → 机制解释 → 政策推荐
   本分析填补了"微观轨迹验证"这一关键环节。
"""
print(purpose_text)
