"""
Figure 4.2.1 - Data Distribution Analysis
Visualizes DO, Ammonia, and Temperature distributions
Shows dataset understanding with histogram and KDE plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_dataset(csv_path: str = "../shrimp_disease_detection_dataset_professional.csv"):
    """Load the shrimp disease detection dataset."""
    return pd.read_csv(csv_path)


def create_distribution_plots(df, output_path: str = "Figure_4_2_1_Distributions.png"):
    """
    Create comprehensive distribution plots for key variables.
    Combines histogram with KDE overlay for visual clarity.
    """
    # Select key variables
    variables = {
        'do_min_night_mg_l': 'DO (Night) - mg/L',
        'do_max_day_mg_l': 'DO (Day) - mg/L',
        'ammonia_mg_l': 'Ammonia - mg/L',
        'water_temperature_c': 'Temperature - °C'
    }
    
    # Filter available columns
    available_vars = {k: v for k, v in variables.items() if k in df.columns}
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Figure 4.2.1 – Data Distribution (MUST)\nDataset Understanding with Histograms & KDE',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    axes = axes.flatten()
    
    for idx, (col, label) in enumerate(available_vars.items()):
        ax = axes[idx]
        
        # Filter out NaN values
        data = df[col].dropna()
        
        # Create histogram with KDE overlay
        ax.hist(
            data,
            bins=50,
            density=True,
            alpha=0.6,
            color='steelblue',
            edgecolor='black',
            linewidth=0.5,
            label='Histogram'
        )
        
        # Overlay KDE
        data.plot(
            kind='kde',
            ax=ax,
            color='darkred',
            linewidth=2.5,
            label='KDE'
        )
        
        # Customize subplot
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        
        # Add statistics annotation
        mean = data.mean()
        median = data.median()
        std = data.std()
        
        stats_text = f'μ: {mean:.2f}\nM: {median:.2f}\nσ: {std:.2f}'
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution plots saved: {output_path}")
    
    return fig


def create_focused_distribution_plots(df, output_path: str = "Figure_4_2_1_Focus_Distributions.png"):
    """
    Create focused distribution plots for the three main variables:
    DO, Ammonia, Temperature
    """
    # Main variables to focus on
    focus_vars = {
        'do_min_night_mg_l': 'DO (Dissolved Oxygen)',
        'ammonia_mg_l': 'Ammonia',
        'water_temperature_c': 'Temperature'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Figure 4.2.1 – Key Variable Distributions\nDO | Ammonia | Temperature',
        fontsize=15,
        fontweight='bold',
        y=1.00
    )
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (col, label) in enumerate(focus_vars.items()):
        ax = axes[idx]
        
        # Filter out NaN values
        data = df[col].dropna()
        
        # Create histogram with KDE overlay
        ax.hist(
            data,
            bins=60,
            density=True,
            alpha=0.65,
            color=colors[idx],
            edgecolor='black',
            linewidth=0.4,
            label='Histogram'
        )
        
        # Overlay KDE
        data.plot(
            kind='kde',
            ax=ax,
            color='darkred',
            linewidth=3,
            label='KDE'
        )
        
        # Customize subplot
        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold', pad=12)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
        
        # Add comprehensive statistics
        mean = data.mean()
        median = data.median()
        std = data.std()
        min_val = data.min()
        max_val = data.max()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        
        stats_text = f'Mean: {mean:.3f}\nMedian: {median:.3f}\nStd: {std:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}\nQ1: {q1:.3f}\nQ3: {q3:.3f}'
        ax.text(
            0.98, 0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, pad=0.8)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Focused distribution plots saved: {output_path}")
    
    return fig


def print_distribution_statistics(df):
    """
    Print detailed statistics for the key variables.
    """
    print("\n" + "="*70)
    print("DATA DISTRIBUTION STATISTICS (Figure 4.2.1)")
    print("="*70)
    
    variables = {
        'do_min_night_mg_l': 'DO (Night)',
        'do_max_day_mg_l': 'DO (Day)',
        'ammonia_mg_l': 'Ammonia',
        'water_temperature_c': 'Temperature'
    }
    
    for col, label in variables.items():
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        
        print(f"\n✔ {label}")
        print(f"   Count: {len(data)}")
        print(f"   Mean: {data.mean():.4f}")
        print(f"   Median: {data.median():.4f}")
        print(f"   Std Dev: {data.std():.4f}")
        print(f"   Min: {data.min():.4f}")
        print(f"   Max: {data.max():.4f}")
        print(f"   Q1 (25%): {data.quantile(0.25):.4f}")
        print(f"   Q3 (75%): {data.quantile(0.75):.4f}")
        print(f"   IQR: {data.quantile(0.75) - data.quantile(0.25):.4f}")
        print(f"   Skewness: {data.skew():.4f}")
        print(f"   Kurtosis: {data.kurtosis():.4f}")


def create_violin_box_plots(df, output_path: str = "Figure_4_2_1_Violin_BoxPlots.png"):
    """
    Create additional visualization with violin plots and box plots.
    Provides complementary view of distributions.
    """
    # Prepare data for visualization
    variables = [
        ('do_min_night_mg_l', 'DO (Night)'),
        ('ammonia_mg_l', 'Ammonia'),
        ('water_temperature_c', 'Temperature')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Figure 4.2.1 – Supplementary: Violin & Box Plots',
        fontsize=15,
        fontweight='bold',
        y=1.00
    )
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (col, label) in enumerate(variables):
        ax = axes[idx]
        
        data = df[col].dropna()
        
        # Create violin plot
        parts = ax.violinplot(
            [data],
            positions=[0],
            widths=0.6,
            showmeans=True,
            showmedians=True
        )
        
        # Color the violin
        for pc in parts['bodies']:
            pc.set_facecolor(colors[idx])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)
        
        # Overlay box plot
        bp = ax.boxplot(
            [data],
            positions=[0.15],
            widths=0.15,
            patch_artist=True,
            boxprops=dict(facecolor='white', alpha=0.7),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5)
        )
        
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks([0])
        ax.set_xticklabels([label])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Violin & box plots saved: {output_path}")
    
    return fig


if __name__ == "__main__":
    # Load dataset
    df = load_dataset()
    print(f"Dataset loaded: {len(df)} records, {len(df.columns)} columns")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_distribution_plots(df)
    create_focused_distribution_plots(df)
    create_violin_box_plots(df)
    
    # Print statistics
    print_distribution_statistics(df)
    
    print("\n" + "="*70)
    print("✓ All distribution analyses complete!")
    print("="*70)
