"""
Figure 4.5.1 - Risk Score Comparison (PRIMARY GRAPH)
Bar chart comparing risk scores across three approaches:
Conventional (0.82) → IoT (0.48) → AI (0.39)
Visually proves risk reduction effectiveness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_risk_scores():
    """
    Risk scores for the three approaches
    Lower risk is better
    """
    scores = {
        'Conventional': 0.82,
        'IoT': 0.48,
        'AI': 0.39
    }
    return scores


def create_risk_comparison_bar_chart(output_path: str = "Figure_4_5_1_Risk_Comparison.png"):
    """
    Create primary bar chart comparing risk scores.
    """
    scores = get_risk_scores()
    
    approaches = list(scores.keys())
    risk_values = list(scores.values())
    
    # Calculate improvements
    conventional_score = scores['Conventional']
    improvements = {
        'Conventional': 0,
        'IoT': ((scores['Conventional'] - scores['IoT']) / scores['Conventional']) * 100,
        'AI': ((scores['Conventional'] - scores['AI']) / scores['Conventional']) * 100
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color scheme - from red (high risk) to green (low risk)
    colors = ['#FF6B6B', '#FFD93D', '#51CF66']
    
    bars = ax.bar(approaches, risk_values, color=colors, edgecolor='black', linewidth=2.5, alpha=0.85)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, risk_values)):
        height = bar.get_height()
        approach = approaches[i]
        improvement = improvements[approach]
        
        # Main value label
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Improvement label (if not conventional)
        if improvement > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height / 2.,
                    f'↓ {improvement:.1f}%',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.8))
    
    # Highlight AI (best approach)
    bars[2].set_edgecolor('gold')
    bars[2].set_linewidth(4)
    
    # Customize plot
    ax.set_ylabel('Risk Score', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title(
        'Figure 4.5.1 – Risk Score Comparison (PRIMARY GRAPH)\nConventional vs IoT vs AI Integration',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Moderate Risk Threshold')
    ax.axhline(y=0.82, color='red', linestyle='--', linewidth=1.5, alpha=0.3, label='Conventional Baseline')
    
    # Add annotation for best approach
    ax.annotate('BEST APPROACH', xy=(2, 0.39), xytext=(2.3, 0.65),
                arrowprops=dict(arrowstyle='->', color='gold', lw=2.5),
                fontsize=12, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Risk comparison bar chart saved: {output_path}")
    
    return fig, improvements


def create_risk_reduction_waterfall(output_path: str = "Figure_4_5_1_Risk_Reduction_Waterfall.png"):
    """
    Create waterfall chart showing risk reduction progression.
    """
    scores = get_risk_scores()
    
    conventional = scores['Conventional']
    iot_reduction = conventional - scores['IoT']
    ai_reduction = scores['IoT'] - scores['AI']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create waterfall data
    categories = ['Conventional\nBaseline', 'IoT\nReduction', 'AI\nImprovement', 'AI Final\nScore']
    values = [conventional, -iot_reduction, -ai_reduction, scores['AI']]
    
    # Starting positions
    x_pos = np.arange(len(categories))
    cumulative = [conventional, scores['IoT'], scores['AI'], scores['AI']]
    
    colors_waterfall = ['#FF6B6B', '#FFD93D', '#51CF66', '#2ECC71']
    
    # Plot bars
    for i, (cat, val, cum) in enumerate(zip(categories, values, cumulative)):
        if i == 0:
            # First bar
            ax.bar(i, cum, color=colors_waterfall[i], edgecolor='black', linewidth=2, alpha=0.85, width=0.6)
            ax.text(i, cum / 2, f'{cum:.2f}', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='white')
        elif i == len(categories) - 1:
            # Last bar (final score)
            ax.bar(i, cum, color=colors_waterfall[i], edgecolor='gold', linewidth=3, alpha=0.85, width=0.6)
            ax.text(i, cum / 2, f'{cum:.2f}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
        else:
            # Reduction bars
            if val < 0:
                ax.bar(i, -val, bottom=cum, color=colors_waterfall[i], edgecolor='black', linewidth=2, alpha=0.85, width=0.6)
                ax.text(i, cum + (-val) / 2, f'{-val:.2f}', ha='center', va='center',
                       fontsize=11, fontweight='bold')
            
            # Connection lines
            if i > 0:
                ax.plot([i-0.4, i-0.4], [cumulative[i-1], cumulative[i]], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylabel('Risk Score', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title(
        'Figure 4.5.1 – Risk Reduction Waterfall\nProgressive Improvement from Conventional to AI',
        fontsize=15,
        fontweight='bold',
        pad=20
    )
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Risk reduction waterfall saved: {output_path}")
    
    return fig


def create_risk_comparison_table(output_path: str = "Figure_4_5_1_Risk_Comparison_Table.png"):
    """
    Create detailed comparison table.
    """
    scores = get_risk_scores()
    
    conventional_score = scores['Conventional']
    
    table_data = {
        'Approach': ['Conventional', 'IoT', 'AI'],
        'Risk Score': [f"{scores['Conventional']:.2f}", f"{scores['IoT']:.2f}", f"{scores['AI']:.2f}"],
        'Improvement': ['Baseline', f"{((scores['Conventional'] - scores['IoT']) / scores['Conventional']) * 100:.1f}%", 
                       f"{((scores['Conventional'] - scores['AI']) / scores['Conventional']) * 100:.1f}%"],
        'vs Conventional': ['—', f"-{(scores['Conventional'] - scores['IoT']):.2f}", 
                           f"-{(scores['Conventional'] - scores['AI']):.2f}"],
        'Risk Level': ['High', 'Medium', 'Low']
    }
    
    df_table = pd.DataFrame(table_data)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                     cellLoc='center', loc='center', colWidths=[0.15, 0.15, 0.2, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)
    
    # Color header
    for i in range(len(df_table.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Color rows
    row_colors = ['#FF6B6B', '#FFD93D', '#51CF66']
    for i in range(1, len(df_table) + 1):
        for j in range(len(df_table.columns)):
            table[(i, j)].set_facecolor(row_colors[i-1])
            table[(i, j)].set_alpha(0.3)
            if i == 3:  # Highlight AI row
                table[(i, j)].set_alpha(0.5)
                table[(i, j)].set_text_props(weight='bold')
    
    plt.title(
        'Figure 4.5.1 – Risk Score Comparison Table\nDetailed Performance Metrics',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Risk comparison table saved: {output_path}")
    
    return fig


def create_comprehensive_comparison(output_path: str = "Figure_4_5_1_Comprehensive.png"):
    """
    Create a comprehensive multi-panel comparison figure.
    """
    scores = get_risk_scores()
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(
        'Figure 4.5.1 – Comprehensive Risk Score Analysis\nMultiple Perspectives on Approach Comparison',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    # Panel 1: Main bar chart
    ax1 = fig.add_subplot(gs[0, :])
    approaches = list(scores.keys())
    risk_values = list(scores.values())
    colors = ['#FF6B6B', '#FFD93D', '#51CF66']
    
    bars = ax1.bar(approaches, risk_values, color=colors, edgecolor='black', linewidth=2, alpha=0.85, width=0.5)
    for i, (bar, value) in enumerate(zip(bars, risk_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{value:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    bars[2].set_edgecolor('gold')
    bars[2].set_linewidth(3.5)
    
    ax1.set_ylabel('Risk Score', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.set_title('Risk Score Comparison', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_axisbelow(True)
    
    # Panel 2: Improvement percentages
    ax2 = fig.add_subplot(gs[1, 0])
    improvements = [0, 
                   ((scores['Conventional'] - scores['IoT']) / scores['Conventional']) * 100,
                   ((scores['Conventional'] - scores['AI']) / scores['Conventional']) * 100]
    
    bars2 = ax2.bar(approaches, improvements, color=colors, edgecolor='black', linewidth=2, alpha=0.85, width=0.5)
    for bar, imp in zip(bars2, improvements):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{imp:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    bars2[2].set_edgecolor('gold')
    bars2[2].set_linewidth(3.5)
    
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Risk Reduction vs Conventional', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_axisbelow(True)
    
    # Panel 3: Absolute reduction
    ax3 = fig.add_subplot(gs[1, 1])
    reductions = [0,
                 scores['Conventional'] - scores['IoT'],
                 scores['Conventional'] - scores['AI']]
    
    bars3 = ax3.bar(approaches, reductions, color=colors, edgecolor='black', linewidth=2, alpha=0.85, width=0.5)
    for bar, red in zip(bars3, reductions):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{red:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    bars3[2].set_edgecolor('gold')
    bars3[2].set_linewidth(3.5)
    
    ax3.set_ylabel('Risk Score Reduction', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Risk Reduction', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_axisbelow(True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive comparison saved: {output_path}")
    
    return fig


def print_risk_analysis_summary():
    """
    Print detailed risk score analysis.
    """
    scores = get_risk_scores()
    
    print("\n" + "="*80)
    print("RISK SCORE COMPARISON ANALYSIS (Figure 4.5.1)")
    print("="*80)
    
    print("\n" + "-"*80)
    print("RISK SCORES")
    print("-"*80)
    
    for approach, score in scores.items():
        if score <= 0.3:
            risk_level = "VERY LOW (Excellent)"
        elif score <= 0.5:
            risk_level = "LOW (Good)"
        elif score <= 0.7:
            risk_level = "MEDIUM (Acceptable)"
        else:
            risk_level = "HIGH (Concerning)"
        
        print(f"\n✔ {approach:15s}: {score:.2f} | {risk_level}")
    
    conventional = scores['Conventional']
    
    print("\n" + "-"*80)
    print("IMPROVEMENT ANALYSIS")
    print("-"*80)
    
    print(f"\n✔ IoT vs Conventional:")
    iot_improvement = ((conventional - scores['IoT']) / conventional) * 100
    print(f"   Risk Reduction: {conventional - scores['IoT']:.2f} ({iot_improvement:.1f}%)")
    print(f"   Score: {conventional:.2f} → {scores['IoT']:.2f}")
    
    print(f"\n✔ AI vs Conventional:")
    ai_improvement = ((conventional - scores['AI']) / conventional) * 100
    print(f"   Risk Reduction: {conventional - scores['AI']:.2f} ({ai_improvement:.1f}%)")
    print(f"   Score: {conventional:.2f} → {scores['AI']:.2f}")
    
    print(f"\n✔ AI vs IoT:")
    ai_vs_iot = ((scores['IoT'] - scores['AI']) / scores['IoT']) * 100
    print(f"   Additional Reduction: {scores['IoT'] - scores['AI']:.2f} ({ai_vs_iot:.1f}%)")
    print(f"   Score: {scores['IoT']:.2f} → {scores['AI']:.2f}")
    
    print("\n" + "-"*80)
    print("KEY INSIGHTS")
    print("-"*80)
    
    print(f"\n✔ Risk Reduction Achievement:")
    print(f"   • IoT reduces risk by {iot_improvement:.1f}% - solid improvement")
    print(f"   • AI reduces risk by {ai_improvement:.1f}% - significant progress")
    print(f"   • Absolute risk reduction: {conventional - scores['AI']:.2f} points")
    
    print(f"\n✔ Performance Classification:")
    print(f"   • Conventional: HIGH RISK (requires intensive monitoring)")
    print(f"   • IoT: LOW RISK (automated monitoring effective)")
    print(f"   • AI: VERY LOW RISK (AI-enhanced prediction superior)")
    
    print(f"\n✔ Practical Implications:")
    print(f"   • Conventional approach manages {conventional*100:.0f}% residual risk")
    print(f"   • IoT-enabled approach manages {scores['IoT']*100:.0f}% residual risk")
    print(f"   • AI approach manages only {scores['AI']*100:.0f}% residual risk")
    
    print(f"\n✔ Business Value:")
    total_reduction = conventional - scores['AI']
    print(f"   • Total risk mitigation: {total_reduction:.2f} ({ai_improvement:.1f}%)")
    print(f"   • This translates to significantly lower operational risk")
    print(f"   • Direct impact on farm profitability and sustainability")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("Generating Risk Score Comparison visualizations...")
    print("="*80)
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    create_risk_comparison_bar_chart()
    create_risk_reduction_waterfall()
    create_risk_comparison_table()
    create_comprehensive_comparison()
    
    # Print summary
    print_risk_analysis_summary()
    
    print("\n✓ All risk score comparison visualizations complete!")
    print("="*80)
