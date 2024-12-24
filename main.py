#!/usr/bin/env python3
import argparse
import json
import os
from src.core.analyzer import ImageAnalyzer
from typing import List, Dict, Any
from src.utils.result_manager import ResultManager

def load_phase_labels(labels_file: str = None) -> dict:
    """Load phase labels from file or use defaults."""
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            return json.load(f)
    
    # Default phase labels from system prompt
    with open('config/prompts/system_prompt.json', 'r') as f:
        return json.load(f)['phase_descriptions']

def load_json_data(json_file: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_video_name(image_path: str) -> str:
    """Extract video name from image path."""
    # Match pattern like "E:\dataset\CholecT50\videos\VID06\000000.png"
    parts = image_path.split('\\')
    for part in parts:
        if part.startswith('VID'):
            return part
    return "unknown"

def create_output_file(output_dir: str, video_name: str, model: str, timestamp: str) -> str:
    """Create output file and initialize with metadata."""
    output_file = os.path.join(
        output_dir,
        f"{video_name}_{model}_analysis_{timestamp}.json"
    )
    
    # Initialize the JSON structure
    initial_data = {
        "model": model,
        "video_name": video_name,
        "total_images": 0,
        "predictions": {},
        "ground_truth": {},
        "accuracy": 0.0,
        "timestamp": timestamp
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, indent=2)
    
    return output_file

def update_results_file(output_file: str, image_name: str, prediction: str, ground_truth: str):
    """Update the results file with new prediction."""
    try:
        # Read current data
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update data
        data['predictions'][image_name] = prediction
        data['ground_truth'][image_name] = ground_truth
        data['total_images'] = len(data['predictions'])
        
        # Calculate running accuracy
        correct_predictions = sum(1 for k in data['predictions'] 
                                if data['predictions'][k] == data['ground_truth'][k])
        data['accuracy'] = correct_predictions / data['total_images']
        
        # Write updated data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"Error updating results file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Surgical Phase Recognition System')
    
    # Add new argument for phase labels file
    parser.add_argument('--phase-labels', default='datasets/phase_choices.json',
                      help='JSON file containing phase labels (default: datasets/phase_choices.json)')
    
    # Required arguments
    parser.add_argument('--input', default="E:\dataset\CholecT50\data_phase\sampling_0.01\VID08_phase.json",
                      help='Input JSON file containing image paths and ground truth')
    
    # Optional arguments
    parser.add_argument('--model', default='openai',
                      choices=['openai', 'anthropic', 'google', 'xai'],
                      help='AI model to use for analysis (default: openai)')
    parser.add_argument('--output-dir', default='results',
                      help='Output directory for results (default: results)')
    parser.add_argument('--resize', action='store_true',
                      help='Resize images before processing')
    parser.add_argument('--max-size', type=int, default=768,
                      help='Maximum dimension for image resize (default: 768)')
    parser.add_argument('--batch-size', type=int, default=10,
                      help='Number of images to process in parallel (default: 10)')
    
    args = parser.parse_args()

    # Load phase labels
    try:
        with open(args.phase_labels, 'r') as f:
            phase_dict = json.load(f)
        print(f"Loaded phase labels from {args.phase_labels}")
    except Exception as e:
        print(f"Error loading phase labels: {str(e)}")
        return

    # Load input data
    try:
        data = load_json_data(args.input)
        print(f"Loaded {len(data)} entries from {args.input}")
    except Exception as e:
        print(f"Error loading input file: {str(e)}")
        return

    # Initialize analyzer and set phase labels
    analyzer = ImageAnalyzer(
        model_name=args.model,
        resize=args.resize,
        max_size=args.max_size
    )
    analyzer.set_phase_labels(phase_dict)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize results tracking
    video_name = None
    result_manager = None

    for entry in data:
        if not entry['images']:
            continue
            
        image_path = entry['images'][0]
        if not video_name:
            video_name = extract_video_name(image_path)
            # Initialize result manager when we have the video name
            result_manager = ResultManager(
                args.output_dir,
                args.model,
                video_name,
                analyzer.get_timestamp()
            )

        try:
            # Use the query from the JSON file
            result = analyzer.analyze_image(image_path, entry['query'])
            image_name = os.path.basename(image_path)
            
            # Update results file in real-time
            result_manager.update_result(image_name, result, entry['response'])
            
            print(f"Processed {image_name}: Predicted={result}, Ground Truth={entry['response']}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    # Final message
    if result_manager:
        output_file = result_manager.get_output_file()
        print(f"\nResults saved to: {output_file}")
        final_data = result_manager.get_final_results()
        print(f"Overall accuracy: {final_data.get('accuracy', 0):.2%}")

if __name__ == "__main__":
    main() 