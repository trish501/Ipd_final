import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Compare evaluated models visually")
    parser.add_argument("--results", type=str, default="evaluation_results.json", help="Path to evaluation results JSON")
    parser.add_argument("--output", type=str, default="comparison_chart.png", help="Output image file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        with open(args.results, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.results} not found. Run evaluate_all.py first.")
        return

    models = []
    map50 = []
    map50_95 = []
    
    for model, metrics in results.items():
        if isinstance(metrics, dict) and "mAP50" in metrics and "mAP50-95" in metrics:
            models.append(model)
            map50.append(metrics["mAP50"])
            map50_95.append(metrics["mAP50-95"])
            
    if not models:
        print("No valid models with metrics found in the results file.")
        return
        
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, map50, width, label='mAP@50')
    rects2 = ax.bar(x + width/2, map50_95, width, label='mAP@50-95')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    
    fig.tight_layout()
    plt.savefig(args.output)
    print(f"Comparison chart saved to {args.output}")

if __name__ == '__main__':
    main()
