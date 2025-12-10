#!/usr/bin/env python3
"""
Visualize benchmark results
Create charts and tables from benchmark JSON files
"""

import json
import argparse
from pathlib import Path


def print_table(headers, rows):
    """Print a formatted table"""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


def visualize_k_values(results):
    """Visualize K-value benchmark results"""
    print("\n" + "="*60)
    print("K-Value Comparison Results")
    print("="*60 + "\n")
    
    headers = ["K Value", "Avg Time (s)", "Min Time (s)", "Max Time (s)", "Avg Results"]
    rows = []
    
    for key, data in sorted(results.items()):
        if key.startswith('k_'):
            k = data.get('k', key.split('_')[1])
            rows.append([
                k,
                f"{data.get('avg_retrieval_time', 0):.3f}",
                f"{data.get('min_retrieval_time', 0):.3f}",
                f"{data.get('max_retrieval_time', 0):.3f}",
                f"{data.get('avg_results_returned', 0):.1f}"
            ])
    
    print_table(headers, rows)
    
    # Quality metrics if available
    quality_available = any('avg_precision' in v for v in results.values())
    if quality_available:
        print("\n" + "="*60)
        print("Quality Metrics")
        print("="*60 + "\n")
        
        headers = ["K Value", "Precision", "Recall", "F1 Score"]
        rows = []
        
        for key, data in sorted(results.items()):
            if key.startswith('k_') and 'avg_precision' in data:
                k = data.get('k', key.split('_')[1])
                rows.append([
                    k,
                    f"{data.get('avg_precision', 0):.3f}",
                    f"{data.get('avg_recall', 0):.3f}",
                    f"{data.get('avg_f1', 0):.3f}"
                ])
        
        print_table(headers, rows)


def visualize_embedding_models(results):
    """Visualize embedding model comparison results"""
    print("\n" + "="*60)
    print("Embedding Model Comparison Results")
    print("="*60 + "\n")
    
    headers = ["Model", "Load Time (s)", "Encode Time (s)", "Texts/Second", "Dimension"]
    rows = []
    
    for model_name, data in results.items():
        if 'error' in data:
            rows.append([model_name, "ERROR", "-", "-", "-"])
        else:
            rows.append([
                model_name.split('/')[-1][:30],  # Shorten model name
                f"{data.get('load_time', 0):.2f}",
                f"{data.get('encode_time', 0):.2f}",
                f"{data.get('texts_per_second', 0):.1f}",
                data.get('dimension', 0)
            ])
    
    print_table(headers, rows)


def visualize_full_benchmark(results):
    """Visualize full system benchmark results"""
    print("\n" + "="*60)
    print("Full System Benchmark Results")
    print("="*60 + "\n")
    
    print(f"Number of questions: {results.get('num_questions', 0)}")
    print(f"K values tested: {results.get('k_values_tested', [])}\n")
    
    results_by_k = results.get('results_by_k', {})
    
    if results_by_k:
        headers = ["K Value", "Avg Total (s)", "Avg Retrieval (s)", "Avg Generation (s)"]
        rows = []
        
        for key, data in sorted(results_by_k.items()):
            perf = data.get('performance', {})
            rows.append([
                key.split('_')[1],
                f"{perf.get('avg_total_time', 0):.3f}",
                f"{perf.get('avg_retrieval_time', 0):.3f}",
                f"{perf.get('avg_generation_time', 0):.3f}"
            ])
        
        print("Performance Metrics:")
        print_table(headers, rows)
        
        # Quality metrics if available
        quality_data = [(k, v.get('quality', {})) for k, v in results_by_k.items()]
        if any(q for _, q in quality_data):
            print("\nQuality Metrics:")
            headers = ["K Value", "Exact Match", "Avg Token F1"]
            rows = []
            
            for key, quality in quality_data:
                if quality:
                    rows.append([
                        key.split('_')[1],
                        f"{quality.get('exact_match', 0):.3f}",
                        f"{quality.get('avg_token_f1', 0):.3f}"
                    ])
            
            print_table(headers, rows)


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('results_file', type=str, help='Path to benchmark results JSON file')
    parser.add_argument('--format', type=str, choices=['auto', 'k_values', 'embedding', 'full'],
                       default='auto', help='Result format (auto-detect by default)')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    # Auto-detect format if needed
    format_type = args.format
    if format_type == 'auto':
        if 'results_by_k' in results:
            format_type = 'full'
        elif any(key.startswith('k_') for key in results.keys()):
            format_type = 'k_values'
        else:
            format_type = 'embedding'
    
    # Visualize based on format
    if format_type == 'k_values':
        visualize_k_values(results)
    elif format_type == 'embedding':
        visualize_embedding_models(results)
    elif format_type == 'full':
        visualize_full_benchmark(results)
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
