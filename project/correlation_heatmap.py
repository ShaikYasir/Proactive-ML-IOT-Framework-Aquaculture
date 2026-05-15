"""
Figure 4.2.2 - Correlation Heatmap Analysis
Visualizes key relationships in the shrimp disease detection dataset
Shows: DO vs Mortality, Ammonia vs Risk, Temperature vs Stress
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_dataset(csv_path: str = "../shrimp_disease_detection_dataset_professional.csv"):
    """Load the shrimp disease detection dataset."""
    return pd.read_csv(csv_path)


def prepare_correlation_data(df):
    """
    Prepare data for correlation analysis.
    Maps categorical risk levels to numeric values.
    """
    # Create a working copy
    df_corr = df.copy()
    
    # Map disease_risk_level to numeric values for correlation
    risk_mapping = {
        'Low': 1,
        'Moderate': 2,
        'High': 3,
        'Critical': 4
    }
    df_corr['risk_numeric'] = df_corr['disease_risk_level'].map(risk_mapping)
    
    # Select key columns for correlation heatmap
    correlation_cols = [
        'do_min_night_mg_l',           # DO minimum
        'do_max_day_mg_l',             # DO maximum
        'daily_mortality_count',        # Mortality
        'ammonia_mg_l',                # Ammonia
        'risk_numeric',                # Risk (mapped)
        'water_temperature_c',         # Temperature
        'behavioral_stress_index',     # Stress
    ]
    
    # Filter to only available columns
    available_cols = [col for col in correlation_cols if col in df_corr.columns]
    
    return df_corr[available_cols].dropna()


def create_correlation_heatmap(df_corr, output_path: str = "Figure_4_2_2_Correlation_Heatmap.png"):
    """
    Create and save the correlation heatmap visualization.
    Highlights the key relationships: DO-Mortality, Ammonia-Risk, Temperature-Stress
    """
    # Calculate correlation matrix
    corr_matrix = df_corr.corr()
    
    # Create figure with larger size for clarity
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap with annotations
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 9}
    )
    
    # Customize the plot
    ax.set_title(
        'Figure 4.2.2 – Correlation Heatmap (HIGH IMPACT)\nKey Relationships in Shrimp Farm Data',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Improve label readability
    labels = [
        'DO Min (Night)',
        'DO Max (Day)',
        'Mortality Count',
        'Ammonia',
        'Risk Level',
        'Temperature',
        'Stress Index'
    ]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved: {output_path}")
    
    return fig, corr_matrix


def print_key_relationships(corr_matrix):
    """
    Print the key correlations highlighted in the analysis.
    Shows: DO vs Mortality, Ammonia vs Risk, Temperature vs Stress
    """
    print("\n" + "="*60)
    print("KEY RELATIONSHIPS (Figure 4.2.2)")
    print("="*60)
    
    # Define key relationships to highlight
    relationships = [
        ('DO Min (Night)', 'Mortality Count', 'DO vs Mortality'),
        ('DO Max (Day)', 'Mortality Count', 'DO vs Mortality (Day)'),
        ('Ammonia', 'Risk Level', 'Ammonia vs Risk'),
        ('Temperature', 'Stress Index', 'Temperature vs Stress'),
    ]
    
    # Map display names to column names
    col_map = {
        'DO Min (Night)': 'do_min_night_mg_l',
        'DO Max (Day)': 'do_max_day_mg_l',
        'Mortality Count': 'daily_mortality_count',
        'Ammonia': 'ammonia_mg_l',
        'Risk Level': 'risk_numeric',
        'Temperature': 'water_temperature_c',
        'Stress Index': 'behavioral_stress_index',
    }
    
    # Extract and print key correlations
    for display_col1, display_col2, label in relationships:
        col1 = col_map.get(display_col1)
        col2 = col_map.get(display_col2)
        
        if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
            corr_value = corr_matrix.loc[col1, col2]
            strength = "Strong" if abs(corr_value) > 0.5 else "Moderate" if abs(corr_value) > 0.3 else "Weak"
            direction = "positive" if corr_value > 0 else "negative"
            
            print(f"\n✔ {label}:")
            print(f"   Correlation: {corr_value:.4f}")
            print(f"   Strength: {strength} {direction}")


def create_focused_heatmap(df_corr, output_path: str = "Figure_4_2_2_Focused_Heatmap.png"):
    """
    Create a focused heatmap showing only the key variables of interest.
    """
    # Select only the key columns mentioned in the requirements
    key_cols = {
        'do_min_night_mg_l': 'DO (Night)',
        'do_max_day_mg_l': 'DO (Day)',
        'daily_mortality_count': 'Mortality',
        'ammonia_mg_l': 'Ammonia',
        'risk_numeric': 'Risk',
        'water_temperature_c': 'Temperature',
        'behavioral_stress_index': 'Stress'
    }
    
    # Filter data
    df_focused = df_corr[[col for col in key_cols.keys() if col in df_corr.columns]]
    
    # Rename columns for display
    df_focused.columns = [key_cols[col] for col in df_focused.columns]
    
    # Calculate correlation
    corr_focused = df_focused.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr_focused,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    ax.set_title(
        'Figure 4.2.2 – Key Variable Correlations\n(HIGH IMPACT)',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Focused heatmap saved: {output_path}")
    
    return fig, corr_focused


if __name__ == "__main__":
    # Load and prepare data
    df = load_dataset()
    df_corr = prepare_correlation_data(df)
    
    print(f"Dataset loaded: {len(df)} records, {len(df.columns)} columns")
    print(f"Correlation data: {len(df_corr)} records for analysis")
    
    # Create full correlation heatmap
    fig1, corr_full = create_correlation_heatmap(
        df_corr,
        output_path="Figure_4_2_2_Correlation_Heatmap.png"
    )
    
    # Create focused heatmap
    fig2, corr_focused = create_focused_heatmap(
        df_corr,
        output_path="Figure_4_2_2_Focused_Heatmap.png"
    )
    
    # Print key relationships
    print_key_relationships(corr_full)
    
    print("\n" + "="*60)
    print("✓ Analysis complete!")
    print("="*60)
