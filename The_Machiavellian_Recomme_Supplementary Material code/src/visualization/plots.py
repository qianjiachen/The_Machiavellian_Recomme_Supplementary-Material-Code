"""Visualization utilities for experiment results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class TrainingPlotter:
    """Plotter for training curves."""
    
    def __init__(self, save_dir: str = "outputs/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_training_curves(
        self,
        curves: Dict[str, Dict[str, List[float]]],
        metrics: List[str] = None,
        title: str = "Training Curves",
        save_name: str = "training_curves.png"
    ):
        """
        Plot training curves for multiple methods.
        
        Args:
            curves: Dict mapping method names to their training curves
            metrics: List of metrics to plot
            title: Plot title
            save_name: Filename to save
        """
        if metrics is None:
            metrics = ["collusion_rate", "user_utility", "gini_coefficient"]
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            for method_name, method_curves in curves.items():
                if metric in method_curves:
                    values = method_curves[metric]
                    steps = range(len(values))
                    ax.plot(steps, values, label=method_name, linewidth=2)
            
            ax.set_xlabel("Training Steps")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_collusion_emergence(
        self,
        curves: Dict[str, List[float]],
        save_name: str = "collusion_emergence.png"
    ):
        """Plot collusion rate emergence over training."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method_name, values in curves.items():
            steps = range(len(values))
            ax.plot(steps, values, label=method_name, linewidth=2)
        
        ax.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='40% threshold')
        ax.set_xlabel("Training Steps (K)")
        ax.set_ylabel("Collusion Rate")
        ax.set_title("Emergence of Collusion During Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()


class ParetoPlotter:
    """Plotter for Pareto frontier analysis."""
    
    def __init__(self, save_dir: str = "outputs/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_pareto_frontier(
        self,
        results: Dict[str, Tuple[float, float]],
        x_label: str = "User Utility (Efficiency)",
        y_label: str = "1 - Gini (Fairness)",
        title: str = "Efficiency-Fairness Tradeoff",
        save_name: str = "pareto_frontier.png"
    ):
        """
        Plot Pareto frontier for efficiency-fairness tradeoff.
        
        Args:
            results: Dict mapping method names to (efficiency, fairness) tuples
            x_label: X-axis label
            y_label: Y-axis label
            title: Plot title
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot points
        for method_name, (efficiency, fairness) in results.items():
            marker = '*' if method_name == 'SWA' else 'o'
            size = 200 if method_name == 'SWA' else 100
            ax.scatter(efficiency, fairness, s=size, marker=marker, label=method_name)
            ax.annotate(method_name, (efficiency, fairness), 
                       textcoords="offset points", xytext=(5, 5), fontsize=9)
        
        # Plot approximate Pareto frontier
        points = list(results.values())
        points.sort(key=lambda x: x[0])
        
        pareto_x = [points[0][0]]
        pareto_y = [points[0][1]]
        max_y = points[0][1]
        
        for x, y in points[1:]:
            if y > max_y:
                pareto_x.append(x)
                pareto_y.append(y)
                max_y = y
        
        ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, label='Pareto Frontier')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()


class AblationPlotter:
    """Plotter for ablation study results."""
    
    def __init__(self, save_dir: str = "outputs/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_ablation_bars(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "Ablation Study",
        save_name: str = "ablation_study.png"
    ):
        """
        Plot ablation study as grouped bar chart.
        
        Args:
            results: Dict mapping variant names to metric dicts
            metrics: List of metrics to plot
            title: Plot title
            save_name: Filename to save
        """
        if metrics is None:
            metrics = ["defense_rate", "user_utility", "gini_coefficient"]
        
        variants = list(results.keys())
        x = np.arange(len(variants))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, metric in enumerate(metrics):
            values = [results[v].get(metric, 0) for v in variants]
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.replace("_", " ").title())
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel("Variant")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_component_contribution(
        self,
        full_swa: Dict[str, float],
        wo_equity: Dict[str, float],
        wo_welfare: Dict[str, float],
        wo_both: Dict[str, float],
        save_name: str = "component_contribution.png"
    ):
        """Plot contribution of each SWA component."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        variants = ["Full SWA", "w/o Equity", "w/o Welfare", "w/o Both"]
        
        # Defense Rate
        defense_rates = [
            full_swa.get("defense_rate", 0),
            wo_equity.get("defense_rate", 0),
            wo_welfare.get("defense_rate", 0),
            wo_both.get("defense_rate", 0)
        ]
        axes[0].bar(variants, defense_rates, color=['green', 'orange', 'red', 'gray'])
        axes[0].set_ylabel("Defense Rate")
        axes[0].set_title("Defense Rate by Variant")
        axes[0].tick_params(axis='x', rotation=15)
        
        # User Utility
        utilities = [
            full_swa.get("user_utility", 0),
            wo_equity.get("user_utility", 0),
            wo_welfare.get("user_utility", 0),
            wo_both.get("user_utility", 0)
        ]
        axes[1].bar(variants, utilities, color=['green', 'orange', 'red', 'gray'])
        axes[1].set_ylabel("User Utility")
        axes[1].set_title("User Utility by Variant")
        axes[1].tick_params(axis='x', rotation=15)
        
        # Gini Coefficient
        ginis = [
            full_swa.get("gini_coefficient", 0),
            wo_equity.get("gini_coefficient", 0),
            wo_welfare.get("gini_coefficient", 0),
            wo_both.get("gini_coefficient", 0)
        ]
        axes[2].bar(variants, ginis, color=['green', 'orange', 'red', 'gray'])
        axes[2].set_ylabel("Gini Coefficient")
        axes[2].set_title("Gini Coefficient by Variant")
        axes[2].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        plt.close()


def generate_results_table(
    results: Dict[str, Dict[str, float]],
    output_path: str = "outputs/results_table.txt"
) -> str:
    """
    Generate a formatted results table.
    
    Args:
        results: Dict mapping method names to metric dicts
        output_path: Path to save the table
        
    Returns:
        Formatted table string
    """
    headers = ["Method", "Defense Rate", "User Util.", "Gini", "Margin", "HHI", "Surplus"]
    
    rows = []
    for method, metrics in results.items():
        row = [
            method,
            f"{metrics.get('defense_rate', 0):.1%}",
            f"{metrics.get('user_utility', 0):.2f}",
            f"{metrics.get('gini_coefficient', 0):.2f}",
            f"{metrics.get('price_margin', 0):.2f}",
            f"{metrics.get('hhi', 0):.2f}",
            f"{metrics.get('consumer_surplus', 0):.2f}"
        ]
        rows.append(row)
    
    # Format table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    for row in rows:
        lines.append(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))
    
    table_str = "\n".join(lines)
    
    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table_str)
    
    return table_str
