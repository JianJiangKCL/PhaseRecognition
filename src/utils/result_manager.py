import os
import json
from typing import Dict, Any

class ResultManager:
    """Manages saving and updating results for surgical phase recognition."""
    
    def __init__(self, output_dir: str, model: str, video_name: str, timestamp: str):
        """Initialize ResultManager with output parameters."""
        self.output_dir = output_dir
        self.model = model
        self.video_name = video_name
        self.timestamp = timestamp
        self.output_file = self._create_output_file()
    
    def _create_output_file(self) -> str:
        """Create and initialize the output JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_file = os.path.join(
            self.output_dir,
            f"{self.video_name}_{self.model}_analysis_{self.timestamp}.json"
        )
        
        initial_data = {
            "model": self.model,
            "video_name": self.video_name,
            "total_images": 0,
            "predictions": {},
            "ground_truth": {},
            "accuracy": 0.0,
            "timestamp": self.timestamp
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2)
        
        return output_file
    
    def update_result(self, image_name: str, prediction: str, ground_truth: str) -> None:
        """Update results file with new prediction."""
        try:
            # Read current data
            with open(self.output_file, 'r', encoding='utf-8') as f:
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
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error updating results file: {str(e)}")
    
    def get_final_results(self) -> Dict[str, Any]:
        """Read and return the final results."""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading final results: {str(e)}")
            return {}
    
    def get_output_file(self) -> str:
        """Return the path to the output file."""
        return self.output_file 