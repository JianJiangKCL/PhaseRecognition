import json
import matplotlib.pyplot as plt
from collections import Counter
import argparse

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Generate ground truth distribution plot')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to the input JSON file')
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    
    # Read the JSON file
    with open(args.input, 'r') as f:
        data = json.load(f)

    # Get ground truth labels and their descriptions
    ground_truth = data['ground_truth']
    phase_labels = data['phase_labels']

    # Count the frequency of each phase
    phase_counts = Counter(ground_truth.values())

    # Prepare data for plotting
    phases = list(phase_labels.keys())
    counts = [phase_counts[phase] for phase in phases]

    # Create bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(phases, counts)

    # Customize the plot
    plt.title('Distribution of Surgical Phases in Ground Truth Data', fontsize=14, pad=20)
    plt.xlabel('Surgical Phases', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    # Add phase descriptions in legend
    legend_labels = [f"{phase}: {desc}" for phase, desc in phase_labels.items()]
    plt.legend(bars, legend_labels, title="Phase Descriptions", 
              bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot with a name based on the input file
    output_path = args.input.replace('.json', '_distribution.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    main() 