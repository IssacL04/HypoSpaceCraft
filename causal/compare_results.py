#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare standard mode and CoT mode benchmark results across multiple datasets

Features:
- Support for multiple node datasets (n3, n4, n5)
- Comprehensive metric comparison
- Visualization with matplotlib
- Statistical analysis
- Export results to CSV and plots

Usage:
    python compare_results.py --results-dir results/cot_comparison
    python compare_results.py --results-dir results/cot_comparison --output comparison_report.pdf
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def discover_result_files(results_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Discover all result files in the directory
    
    Returns:
        Dictionary mapping node names to {'standard': path, 'cot': path}
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_dir}")
    
    files = {}
    
    for file_path in results_path.glob("*.json"):
        filename = file_path.stem
        parts = filename.rsplit('_', 1)
        
        if len(parts) == 2:
            node_name, mode = parts
            if mode in ['standard', 'cot']:
                if node_name not in files:
                    files[node_name] = {}
                files[node_name][mode] = str(file_path)
    
    return files


def extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from results"""
    stats = results.get('statistics', {})
    
    return {
        'parse_success_rate': stats.get('parse_success_rate', {}).get('mean', 0),
        'valid_rate': stats.get('valid_rate', {}).get('mean', 0),
        'novelty_rate': stats.get('novelty_rate', {}).get('mean', 0),
        'recovery_rate': stats.get('recovery_rate', {}).get('mean', 0),
        'valid_rate_std': stats.get('valid_rate', {}).get('std', 0),
        'novelty_rate_std': stats.get('novelty_rate', {}).get('std', 0),
        'recovery_rate_std': stats.get('recovery_rate', {}).get('std', 0),
        'total_tokens': results.get('token_usage', {}).get('total_tokens', 0),
        'avg_tokens_per_query': results.get('token_usage', {}).get('avg_tokens_per_query', 0),
        'total_cost': results.get('cost', {}).get('total_cost', 0),
        'avg_cost_per_query': results.get('cost', {}).get('avg_cost_per_query', 0),
        'n_samples': results.get('n_samples', 0),
        'total_errors': results.get('error_summary', {}).get('total_errors', 0),
    }


def print_comparison_table(data: Dict[str, Dict[str, Dict[str, float]]]):
    """Print comparison table for all datasets"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 80)
    
    metrics = [
        ('Parse Success', 'parse_success_rate'),
        ('Valid Rate', 'valid_rate'),
        ('Novelty Rate', 'novelty_rate'),
        ('Recovery Rate', 'recovery_rate'),
    ]
    
    for node_name in sorted(data.keys()):
        print(f"\n{'-' * 80}")
        print(f"Dataset: {node_name.upper()}")
        print('-' * 80)
        
        if 'standard' not in data[node_name] or 'cot' not in data[node_name]:
            print("  Incomplete data - skipping")
            continue
        
        std = data[node_name]['standard']
        cot = data[node_name]['cot']
        
        print(f"{'Metric':<20} {'Standard':>12} {'CoT':>12} {'-':>12} {'-%':>12}")
        print('-' * 80)
        
        for display_name, key in metrics:
            std_val = std[key]
            cot_val = cot[key]
            diff = cot_val - std_val
            pct_change = (diff / std_val * 100) if std_val != 0 else 0
            
            print(f"{display_name:<20} {std_val:>12.4f} {cot_val:>12.4f} "
                  f"{diff:>+12.4f} {pct_change:>+11.2f}%")
        
        print('-' * 80)
        print(f"{'Cost (Total)':<20} ${std['total_cost']:>11.4f} ${cot['total_cost']:>11.4f} "
              f"${cot['total_cost'] - std['total_cost']:>+11.4f}")
        print(f"{'Tokens (Total)':<20} {std['total_tokens']:>12,} {cot['total_tokens']:>12,} "
              f"{cot['total_tokens'] - std['total_tokens']:>+12,}")


def create_visualizations(data: Dict[str, Dict[str, Dict[str, float]]], 
                         output_dir: str = None):
    """Create comprehensive visualization plots"""
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    

    node_names = sorted([k for k in data.keys() if 'standard' in data[k] and 'cot' in data[k]])
    
    if not node_names:
        print("Warning: No complete dataset pairs found for visualization")
        return

    metrics_to_plot = [
        ('Valid Rate', 'valid_rate'),
        ('Novelty Rate', 'novelty_rate'),
        ('Recovery Rate', 'recovery_rate'),
        ('Parse Success', 'parse_success_rate'),
    ]

    fig = plt.figure(figsize=(16, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(node_names))
    width = 0.35
    
    for i, (display_name, key) in enumerate(metrics_to_plot[:3]):
        ax_sub = plt.subplot(2, 3, i + 1)
        
        std_vals = [data[node]['standard'][key] for node in node_names]
        cot_vals = [data[node]['cot'][key] for node in node_names]
        
        bars1 = ax_sub.bar(x - width/2, std_vals, width, label='Standard', alpha=0.8)
        bars2 = ax_sub.bar(x + width/2, cot_vals, width, label='CoT', alpha=0.8)
        
        ax_sub.set_xlabel('Dataset')
        ax_sub.set_ylabel(display_name)
        ax_sub.set_title(f'{display_name} Comparison')
        ax_sub.set_xticks(x)
        ax_sub.set_xticklabels([n.upper() for n in node_names])
        ax_sub.legend()
        ax_sub.grid(axis='y', alpha=0.3)
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax_sub.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax4 = plt.subplot(2, 3, 4)
    std_costs = [data[node]['standard']['total_cost'] for node in node_names]
    cot_costs = [data[node]['cot']['total_cost'] for node in node_names]
    
    bars1 = ax4.bar(x - width/2, std_costs, width, label='Standard', alpha=0.8)
    bars2 = ax4.bar(x + width/2, cot_costs, width, label='CoT', alpha=0.8)
    
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Total Cost ($)')
    ax4.set_title('Cost Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([n.upper() for n in node_names])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.4f}',
                ha='center', va='bottom', fontsize=8)

    ax5 = plt.subplot(2, 3, 5)
    std_tokens = [data[node]['standard']['avg_tokens_per_query'] for node in node_names]
    cot_tokens = [data[node]['cot']['avg_tokens_per_query'] for node in node_names]
    
    bars1 = ax5.bar(x - width/2, std_tokens, width, label='Standard', alpha=0.8)
    bars2 = ax5.bar(x + width/2, cot_tokens, width, label='CoT', alpha=0.8)
    
    ax5.set_xlabel('Dataset')
    ax5.set_ylabel('Avg Tokens per Query')
    ax5.set_title('Token Usage Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels([n.upper() for n in node_names])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    
    improvements = []
    improvement_labels = []
    
    for display_name, key in metrics_to_plot[:3]:
        impr = []
        for node in node_names:
            std_val = data[node]['standard'][key]
            cot_val = data[node]['cot'][key]
            pct = ((cot_val - std_val) / std_val * 100) if std_val > 0 else 0
            impr.append(pct)
        improvements.append(impr)
        improvement_labels.append(display_name)
    
    x_pos = np.arange(len(node_names))
    bar_width = 0.25
    
    for i, (impr, label) in enumerate(zip(improvements, improvement_labels)):
        offset = (i - 1) * bar_width
        bars = ax6.bar(x_pos + offset, impr, bar_width, label=label, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=7)
    
    ax6.set_xlabel('Dataset')
    ax6.set_ylabel('Improvement (%)')
    ax6.set_title('CoT Improvement Over Standard (%)')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([n.upper() for n in node_names])
    ax6.legend()
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plot_path = Path(output_dir) / "comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: {plot_path}")
    
    plt.show()
    
    if len(node_names) >= 2:
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

        node_numbers = []
        for node in node_names:
            try:
                num = int(''.join(filter(str.isdigit, node)))
                node_numbers.append(num)
            except:
                node_numbers.append(0)

        trend_metrics = [
            ('Valid Rate', 'valid_rate'),
            ('Novelty Rate', 'novelty_rate'),
            ('Recovery Rate', 'recovery_rate'),
            ('Cost per Query ($)', 'avg_cost_per_query'),
        ]
        
        for idx, (title, key) in enumerate(trend_metrics):
            ax = axes[idx // 2, idx % 2]
            
            std_vals = [data[node]['standard'][key] for node in node_names]
            cot_vals = [data[node]['cot'][key] for node in node_names]
            
            ax.plot(node_numbers, std_vals, marker='o', label='Standard', linewidth=2, markersize=8)
            ax.plot(node_numbers, cot_vals, marker='s', label='CoT', linewidth=2, markersize=8)
            
            ax.set_xlabel('Number of Nodes')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Dataset Complexity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(node_numbers)
        
        plt.tight_layout()
        
        if output_dir:
            trend_path = Path(output_dir) / "trend_analysis.png"
            plt.savefig(trend_path, dpi=300, bbox_inches='tight')
            print(f"Trend analysis saved to: {trend_path}")
        
        plt.show()


def export_to_csv(data: Dict[str, Dict[str, Dict[str, float]]], output_dir: str):
    """Export comparison data to CSV"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    
    for node_name in sorted(data.keys()):
        for mode in ['standard', 'cot']:
            if mode not in data[node_name]:
                continue
            
            metrics = data[node_name][mode]
            row = {
                'dataset': node_name,
                'mode': mode,
                **metrics
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    csv_path = output_path / "comparison_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nData exported to CSV: {csv_path}")
    
    summary_rows = []
    
    for node_name in sorted(data.keys()):
        if 'standard' not in data[node_name] or 'cot' not in data[node_name]:
            continue
        
        std = data[node_name]['standard']
        cot = data[node_name]['cot']
        
        summary = {
            'dataset': node_name,
            'valid_rate_std': std['valid_rate'],
            'valid_rate_cot': cot['valid_rate'],
            'valid_rate_improvement': cot['valid_rate'] - std['valid_rate'],
            'valid_rate_improvement_pct': ((cot['valid_rate'] - std['valid_rate']) / std['valid_rate'] * 100) if std['valid_rate'] > 0 else 0,
            'novelty_rate_std': std['novelty_rate'],
            'novelty_rate_cot': cot['novelty_rate'],
            'novelty_rate_improvement': cot['novelty_rate'] - std['novelty_rate'],
            'recovery_rate_std': std['recovery_rate'],
            'recovery_rate_cot': cot['recovery_rate'],
            'recovery_rate_improvement': cot['recovery_rate'] - std['recovery_rate'],
            'cost_std': std['total_cost'],
            'cost_cot': cot['total_cost'],
            'cost_increase': cot['total_cost'] - std['total_cost'],
            'cost_increase_pct': ((cot['total_cost'] - std['total_cost']) / std['total_cost'] * 100) if std['total_cost'] > 0 else 0,
        }
        summary_rows.append(summary)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_path / "comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary exported to CSV: {summary_path}")
    
    return df, summary_df


def generate_recommendations(data: Dict[str, Dict[str, Dict[str, float]]]):
    """Generate recommendations based on results"""
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    for node_name in sorted(data.keys()):
        if 'standard' not in data[node_name] or 'cot' not in data[node_name]:
            continue
        
        std = data[node_name]['standard']
        cot = data[node_name]['cot']
        
        print(f"\n{node_name.upper()}:")

        valid_improvement = cot['valid_rate'] - std['valid_rate']
        recovery_improvement = cot['recovery_rate'] - std['recovery_rate']
        cost_increase_pct = ((cot['total_cost'] - std['total_cost']) / std['total_cost'] * 100) if std['total_cost'] > 0 else 0

        if valid_improvement > 0.05 or recovery_improvement > 0.10:
            recommendation = "? RECOMMENDED: CoT shows significant improvement"
            reason = f"Valid rate improved by {valid_improvement*100:.2f}%, Recovery rate improved by {recovery_improvement*100:.2f}%"
        elif valid_improvement > 0 and cost_increase_pct < 100:
            recommendation = "~ CONSIDER: CoT shows moderate improvement with reasonable cost"
            reason = f"Valid rate improved by {valid_improvement*100:.2f}%, cost increased by {cost_increase_pct:.1f}%"
        else:
            recommendation = "? NOT RECOMMENDED: Standard mode is sufficient"
            reason = f"CoT improvement ({valid_improvement*100:.2f}%) doesn't justify cost increase ({cost_increase_pct:.1f}%)"
        
        print(f"  {recommendation}")
        print(f"  Reason: {reason}")
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare CoT and Standard prompting results across multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-dir', type=str, default='results/cot_comparison',
                       help='Directory containing result JSON files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots and CSV files (defaults to results-dir)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting (only print tables)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV export')
    
    args = parser.parse_args()

    print("Discovering result files...")
    result_files = discover_result_files(args.results_dir)
    
    if not result_files:
        print(f"Error: No result files found in {args.results_dir}")
        sys.exit(1)
    
    print(f"Found {len(result_files)} dataset(s):")
    for node_name, modes in result_files.items():
        print(f"  {node_name}: {', '.join(modes.keys())}")

    print("\nLoading results...")
    data = {}
    
    for node_name, mode_files in result_files.items():
        data[node_name] = {}
        for mode, filepath in mode_files.items():
            try:
                results = load_results(filepath)
                data[node_name][mode] = extract_metrics(results)
                print(f"  Loaded {node_name} - {mode}")
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")

    print_comparison_table(data)

    generate_recommendations(data)

    output_dir = args.output_dir or args.results_dir

    if not args.no_csv:
        try:
            export_to_csv(data, output_dir)
        except Exception as e:
            print(f"Warning: Failed to export CSV: {e}")

    if not args.no_plot:
        try:
            create_visualizations(data, output_dir)
        except Exception as e:
            print(f"Warning: Failed to create plots: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
