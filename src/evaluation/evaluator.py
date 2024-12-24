from typing import Dict, List, Any
from datetime import datetime
import json
import os

class PhaseEvaluator:
    def __init__(self, phase_labels: Dict[str, str]):
        """
        Initialize the evaluator with phase labels.
        
        Args:
            phase_labels (Dict[str, str]): Dictionary mapping phase letters to descriptions
        """
        self.phase_labels = phase_labels

    def evaluate_predictions(
        self,
        predictions: Dict[str, str],
        ground_truth: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Evaluate phase recognition predictions against ground truth.
        
        Args:
            predictions (Dict[str, str]): Dictionary of {image_name: predicted_phase}
            ground_truth (Dict[str, str]): Dictionary of {image_name: actual_phase}
            
        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics
        """
        total = len(predictions)
        correct = 0
        phase_wise_accuracy = {
            phase: {'correct': 0, 'total': 0}
            for phase in self.phase_labels.keys()
        }
        
        for img_name, pred_phase in predictions.items():
            if img_name in ground_truth:
                true_phase = ground_truth[img_name]
                phase_wise_accuracy[true_phase]['total'] += 1
                
                if pred_phase == true_phase:
                    correct += 1
                    phase_wise_accuracy[true_phase]['correct'] += 1
        
        # Calculate metrics
        overall_accuracy = correct / total if total > 0 else 0
        phase_accuracy = {}
        for phase in phase_wise_accuracy:
            total_phase = phase_wise_accuracy[phase]['total']
            if total_phase > 0:
                phase_accuracy[phase] = phase_wise_accuracy[phase]['correct'] / total_phase
            else:
                phase_accuracy[phase] = 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'phase_wise_accuracy': phase_accuracy,
            'total_images': total,
            'correct_predictions': correct
        }

    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        output_file: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results (Dict[str, Any]): Evaluation results to save
            output_file (str): Path to save the results
            metadata (Dict[str, Any], optional): Additional metadata to include
        """
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "phase_labels": self.phase_labels,
            "results": results
        }
        
        if metadata:
            output_data["metadata"] = metadata
            
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def load_ground_truth(self, json_file: str) -> Dict[str, str]:
        """
        Load ground truth data from a JSON file.
        
        Args:
            json_file (str): Path to the JSON file containing ground truth data
            
        Returns:
            Dict[str, str]: Dictionary mapping image names to their ground truth phases
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Handle both simple {image: phase} format and complex format
        if isinstance(data, dict):
            return data
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return {
                os.path.basename(item['images'][0]): item['response']
                for item in data
                if 'images' in item and 'response' in item
            }
        else:
            raise ValueError("Unsupported ground truth format") 