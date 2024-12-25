from typing import Dict, Any, List, Type
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

from ..api.base import BaseAPI
from ..api.openai_api import OpenAIAPI
from ..api.anthropic_api import AnthropicAPI
from ..api.google_api import GoogleAPI
from ..api.xai_api import XAIAPI
from ..evaluation.evaluator import PhaseEvaluator
from src.utils.response_formatter import ResponseFormatter

class ImageAnalyzer:
    API_CLASSES = {
        'openai': OpenAIAPI,
        'anthropic': AnthropicAPI,
        'google': GoogleAPI,
        'xai': XAIAPI
    }

    def __init__(
        self,
        model_name: str,
        strategy: str = 'current',
        resize: bool = False,
        max_size: int = 768
    ):
        """
        Initialize the ImageAnalyzer with specified model.
        
        Args:
            model_name (str): Name of the model to use ('openai', 'anthropic', 'google', 'xai')
            strategy (str): Strategy to use for phase recognition ('current' or 'past-current')
            resize (bool): Whether to resize images before processing
            max_size (int): Maximum dimension for resized images
        """
        self.model_name = model_name.lower()
        self.strategy = strategy
        self.resize = resize
        self.max_size = max_size if resize else None
        self.config = self._load_config()
        self.api = self._initialize_api()
        self.phase_labels = {}
        self.evaluator = None
        self.formatter = None  # Initialize formatter to None, will be set when phase_labels are set
        self.previous_prediction = None  # Track previous frame prediction

    def get_timestamp(self) -> str:
        """Get current timestamp in format suitable for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_config(self) -> Dict[str, Any]:
        """Load API configuration from file."""
        config_path = os.path.join('config', 'api_config.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load API configuration: {str(e)}")

    def _initialize_api(self) -> BaseAPI:
        """Initialize the appropriate API client."""
        if self.model_name not in self.config:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if self.model_name not in self.API_CLASSES:
            raise ValueError(f"No API class implementation for: {self.model_name}")

        model_config = self.config[self.model_name]
        api_class = self.API_CLASSES[self.model_name]
        return api_class(
            api_key=model_config["api_key"],
            model=model_config["model"]
        )

    def set_phase_labels(self, labels_dict: Dict[str, str]) -> None:
        """Set the phase labels for recognition."""
        self.phase_labels = labels_dict
        self.evaluator = PhaseEvaluator(labels_dict)
        # Create formatter with phase labels
        formatter = ResponseFormatter(labels_dict)
        self.formatter = formatter.get_formatter(self.model_name)

    def create_phase_prompt(self, previous_phase: str = None) -> str:
        """Create a formatted prompt with all phase options."""
        # Load system prompt configuration
        with open('config/prompts/system_prompt.json', 'r') as f:
            system_prompt = json.load(f)
        
        if self.strategy == 'current':
            prompt = system_prompt["base_prompt"] + "\n\n"
        else:  # past-current strategy
            if previous_phase:
                phase_name = self.phase_labels.get(previous_phase, "Unknown")
                prompt = f"{system_prompt['base_prompt']}, considering that the previous frame was in phase {previous_phase} ({phase_name}).\n\n"
            else:
                prompt = f"{system_prompt['base_prompt']} (no previous phase information available).\n\n"
        
        prompt += system_prompt["instruction"] + "\n\n"
        
        for key, description in self.phase_labels.items():
            prompt += f"{key}. {description}\n"
        
        prompt += "\n" + system_prompt["format_instruction"]
        return prompt

    def analyze_image(self, image_path: str, prompt: str = None) -> str:
        """Analyze a single image."""
        if prompt is None:
            prompt = self.create_phase_prompt(self.previous_prediction)

        try:
            # Prepare image data
            image_data = self.api.prepare_image(image_path, self.max_size)
            # Process with API
            raw_response = self.api.process_image(image_data, prompt)
            
            # Format the response if formatter is available
            if self.formatter:
                try:
                    # Call the appropriate format method directly
                    formatted_response = self.formatter(raw_response)
                    # Update previous prediction for next frame
                    self.previous_prediction = formatted_response
                    return formatted_response
                except Exception as e:
                    # Log the error and raw response for debugging
                    print(f"\nWarning - Unformattable response (returning 'Z'): {raw_response}")
                    return 'Z'
            
            # If no formatter, try basic letter extraction
            if raw_response:
                # Look for single letters A-G
                match = re.search(r'\b[A-G]\b', raw_response)
                if match:
                    result = match.group(0)
                    self.previous_prediction = result
                    return result
                
                # Look for any letter followed by a dot or parenthesis
                match = re.search(r'([A-G])[\.\)]', raw_response)
                if match:
                    result = match.group(1)
                    self.previous_prediction = result
                    return result
            
            # Return 'Z' for any unhandled cases
            return 'Z'
            
        except Exception as e:
            print(f"Error processing image (returning 'Z'): {str(e)}")
            return 'Z'

    def analyze_folder(
        self,
        folder_path: str,
        output_dir: str = None,
        supported_formats: tuple = ('.jpg', '.jpeg', '.png', '.gif'),
        save_results: bool = True,
        batch_size: int = 10,
        sampling: int = 1
    ) -> Dict[str, str]:
        """
        Analyze all images in a folder using parallel processing.
        
        Args:
            folder_path (str): Path to the folder containing images
            output_dir (str): Directory to save results
            supported_formats (tuple): Supported image file extensions
            save_results (bool): Whether to save results to JSON
            batch_size (int): Number of images to process in parallel
            sampling (int): Process every Nth frame
            
        Returns:
            Dict[str, str]: Dictionary mapping image paths to their analysis results
        """
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")

        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(folder_path, 'results')
        os.makedirs(output_dir, exist_ok=True)

        # Get and sort image files
        def extract_number(filename):
            return int(''.join(filter(str.isdigit, filename)) or 0)

        image_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(supported_formats)
        ]
        image_files.sort(key=extract_number)
        image_files = image_files[::sampling]

        if not image_files:
            raise ValueError(f"No supported image files found in {folder_path}")

        image_paths = [os.path.join(folder_path, f) for f in image_files]
        prompt = self.create_phase_prompt()

        # Extract metadata
        video_name = "unknown"
        if "/videos/" in folder_path:
            video_parts = folder_path.split("/videos/")
            if len(video_parts) > 1:
                video_name = video_parts[1].split("/")[0]

        # Setup output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config[self.model_name]["model"]
        output_file = os.path.join(
            output_dir,
            f"{video_name}_{model_name}_analysis_{timestamp}.json"
        )

        # Initialize results tracking
        results = {}
        start_time = datetime.now()

        def process_image(image_path: str) -> tuple:
            try:
                result = self.analyze_image(image_path, prompt)
                return os.path.basename(image_path), result
            except Exception as e:
                return os.path.basename(image_path), f"Error: {str(e)}"

        # Process images with progress bar
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(process_image, path): path
                for path in image_paths
            }
            
            for future in tqdm(
                as_completed(futures),
                total=len(image_paths),
                desc="Processing images"
            ):
                img_name, result = future.result()
                results[img_name] = result
                
                if save_results:
                    self._save_intermediate_results(
                        results,
                        output_file,
                        start_time,
                        len(image_paths)
                    )

        # Save final results
        if save_results:
            self._save_final_results(
                results,
                output_file,
                start_time,
                len(image_paths)
            )

        return results

    def _save_intermediate_results(
        self,
        results: Dict[str, str],
        output_file: str,
        start_time: datetime,
        total_images: int
    ) -> None:
        """Save intermediate results to file."""
        current_data = {
            "processing_stats": {
                "total_images": total_images,
                "processed_images": len(results),
                "timestamp_start": start_time.isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "phase_labels": self.phase_labels,
            "results": results
        }
        
        # Atomic write
        with open(output_file + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)
        os.replace(output_file + '.tmp', output_file)

    def _save_final_results(
        self,
        results: Dict[str, str],
        output_file: str,
        start_time: datetime,
        total_images: int
    ) -> None:
        """Save final results with complete statistics."""
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        avg_time = total_time / total_images if total_images > 0 else 0

        final_data = {
            "processing_stats": {
                "total_images": total_images,
                "processed_images": len(results),
                "timestamp_start": start_time.isoformat(),
                "timestamp_end": end_time.isoformat(),
                "total_time_seconds": round(total_time, 2),
                "avg_time_per_image_seconds": round(avg_time, 2),
                "successful_frames": len([r for r in results.values() if not str(r).startswith("Error")])
            },
            "phase_labels": self.phase_labels,
            "results": results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False) 