import base64
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
import argparse
import os
import json
import time
from datetime import datetime
from typing import Dict
import concurrent.futures
from anthropic import Anthropic
import google.generativeai as genai
# Import other necessary model SDKs

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry
import re

MAX_WORKERS = 4
ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 8  # Keep under 10 RPM limit
DELAY_BETWEEN_CALLS = 7   # 60 seconds / 8 requests ≈ 7 seconds between calls
RETRY_DELAY = 10         # Base retry delay when hitting rate limits
XAI_MAX_CALLS_PER_HOUR = 45  # Reduce further from 50 to stay well under 60
XAI_MAX_CALLS_PER_SECOND = 1  # Keep 1 RPS limit
XAI_DELAY_BETWEEN_CALLS = 3   # Increase delay to ensure 1 RPS

@sleep_and_retry
@limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
def rate_limited_google_call(model, image_data, prompt):
    """Rate-limited wrapper for Google API calls"""
    time.sleep(DELAY_BETWEEN_CALLS)  # Add delay between each call
    return model.generate_content([
        {
            "mime_type": "image/jpeg",
            "data": image_data
        },
        prompt
    ])

@sleep_and_retry
@limits(calls=XAI_MAX_CALLS_PER_SECOND, period=1)  # 1 request per second
@limits(calls=XAI_MAX_CALLS_PER_HOUR, period=ONE_MINUTE * 60)  # 45 requests per hour
def rate_limited_xai_call(client, image_data, prompt, model_config):
    """Rate-limited wrapper for XAI API calls with exponential backoff"""
    time.sleep(XAI_DELAY_BETWEEN_CALLS)  # Ensure minimum delay between calls
    
    max_retries = 3
    base_delay = 5  # Increase base delay for retries
    
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model_config["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ],
                    }
                ],
                max_tokens=3000,
            )
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                # More aggressive exponential backoff
                wait_time = base_delay * (4 ** attempt)  # Use 4 instead of 2 for more spacing
                print(f"Rate limit hit, waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            raise

class ImageAnalyzer:
    def __init__(self, model_name="openai", resize=False, max_size=768):
        """
        Initialize the ImageAnalyzer with specified model.
        
        Args:
            model_name (str): Name of the model to use ('openai', 'anthropic', 'google', 'xai')
            resize (bool): Whether to resize images before processing
            max_size (int): Maximum dimension (width or height) for resized images
        """
        self.model_name = model_name
        self.resize = resize
        self.max_size = max_size
        self.config = self._load_config()
        self.client = self._initialize_client()
        self.phase_labels = {}
        self.first_image_logged = False  # Add flag for first image logging

    def _load_config(self):
        """Load API configuration from file"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'api_config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            raise Exception(f"Failed to load API configuration: {str(e)}")

    def _initialize_client(self):
        """Initialize the appropriate client based on model_name"""
        if self.model_name not in self.config:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model_config = self.config[self.model_name]
        
        if self.model_name == "openai":
            return OpenAI(api_key=model_config["api_key"])
        elif self.model_name == "anthropic":
            return Anthropic(api_key=model_config["api_key"])
        elif self.model_name == "google":
            genai.configure(api_key=model_config["api_key"])
            return genai
        elif self.model_name == "xai":
            return OpenAI(
                api_key=model_config["api_key"],
                base_url="https://api.x.ai/v1"
            )
        
        raise ValueError(f"Model initialization not implemented: {self.model_name}")

    def _process_image_with_model(self, image_data, prompt):
        """Process image with the selected model"""
        model_config = self.config[self.model_name]

        try:
            if self.model_name == "xai":
                response = rate_limited_xai_call(
                    self.client, 
                    image_data, 
                    prompt, 
                    model_config
                )
                # Just return the cleaned response text, like other models
                return self._clean_response(response.choices[0].message.content)
            
            elif self.model_name == "openai":
                response = self.client.chat.completions.create(
                    model=model_config["model"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    }
                                }
                            ],
                        }
                    ],
                    max_tokens=3000,
                )
                response = self._clean_response(response.choices[0].message.content)
                return response

            elif self.model_name == "anthropic":
                # Modified Anthropic image processing
                response = self.client.messages.create(
                    model=model_config["model"],
                    max_tokens=3000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",  # Explicitly specify media type
                                        "data": image_data
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                response = self._clean_response(response.content[0].text)
                return response

            elif self.model_name == "google":
                max_retries = 3  # Reduced retries to avoid hitting daily limit
                
                for attempt in range(max_retries):
                    try:
                        model = self.client.GenerativeModel(model_config["model"])
                        response = rate_limited_google_call(model, image_data, prompt)
                        response = self._clean_response(response.text)
                        return response
                    except Exception as e:
                        if "429" in str(e) and attempt < max_retries - 1:
                            wait_time = RETRY_DELAY * (attempt + 1)
                            print(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                        raise

        except Exception as e:
            raise Exception(f"Error processing with {self.model_name}: {str(e)}")

    def _clean_response(self, response):
        """Clean up model response to ensure it's just a single letter A-G"""
        # If response is a string, take just the first character
        if isinstance(response, str):
            # Remove any whitespace and get first character
            cleaned = response.strip()[0].upper()
            # Verify it's a valid phase letter
            if cleaned in 'ABCDEFG':
                return cleaned
            
        # If we couldn't clean the response, return error
        raise ValueError(f"Invalid response format: {response}")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def analyze_local_image(self, image_path, prompt=None):
        """Analyze image with retry logic"""
        if prompt is None:
            prompt = self.create_phase_prompt()
        try:
            # Log first image size
            if not self.first_image_logged:
                with Image.open(image_path) as img:
                    print(f"First image original size: {img.size}")
                    if self.resize:
                        # Calculate new dimensions maintaining aspect ratio
                        width, height = img.size
                        if width > height:
                            if width > self.max_size:
                                new_width = self.max_size
                                new_height = int(height * (self.max_size / width))
                        else:
                            if height > self.max_size:
                                new_height = self.max_size
                                new_width = int(width * (self.max_size / height))
                            else:
                                new_width, new_height = width, height
                        print(f"First image resized to: ({new_width}, {new_height})")
                self.first_image_logged = True

            # Read and encode image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if needed
                if self.resize:
                    width, height = img.size
                    if width > self.max_size or height > self.max_size:
                        if width > height:
                            new_width = self.max_size
                            new_height = int(height * (self.max_size / width))
                        else:
                            new_height = self.max_size
                            new_width = int(width * (self.max_size / height))
                        img = img.resize((new_width, new_height))
                
                # Save to bytes
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                image_data = base64.b64encode(img_byte_arr).decode('utf-8')

            return self._process_image_with_model(image_data, prompt)
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def set_phase_labels(self, labels_dict):
        """Set the phase labels for recognition"""
        self.phase_labels = labels_dict

    def create_phase_prompt(self):
        """Create a formatted prompt with all phase options"""
        prompt = "Please analyze this surgical image and select the most appropriate surgical phase.\n\n"
        prompt += "Choose EXACTLY ONE option from the following phases:\n\n"
        
        for key, description in self.phase_labels.items():
            prompt += f"{key}. {description}\n"
        
        prompt += "\nOutput format: Only return the letter (A/B/C/...) corresponding to your choice."
        return prompt

    def analyze_folder(
        self,
        folder_path: str,
        prompt: str,
        output_dir: str = None,
        supported_formats: tuple = ('.jpg', '.jpeg', '.png', '.gif'),
        save_results: bool = True,
        batch_size: int = 10,
        sampling: int = 5
    ) -> Dict[str, str]:
        """
        Analyze all images in a folder using parallel processing.

        Args:
            folder_path (str): Path to the folder containing images
            prompt (str): Question or instruction about the images
            output_dir (str, optional): Directory to save results. If None, uses folder_path/results
            supported_formats (tuple): Tuple of supported image file extensions
            save_results (bool): Whether to save results to a JSON file
            batch_size (int): Number of images to process in parallel
            sampling (int): Process every Nth frame. Default=1 means process all frames.
                           sampling=2 means process every other frame, etc.

        Returns:
            Dict[str, str]: Dictionary mapping image paths to their analysis results
        """
        # Ensure folder path exists
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")

        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(folder_path, 'results')
        os.makedirs(output_dir, exist_ok=True)

        # Get list of image files and sort numerically
        def extract_number(filename):
            # Extract numbers from filename, default to 0 if no numbers found
            return int(''.join(filter(str.isdigit, filename)) or 0)

        image_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith(supported_formats)
        ]
        image_files.sort(key=extract_number)  # Sort based on numerical value in filename

        # After sorting, apply sampling to image_files
        image_files = image_files[::sampling]  # Take every Nth item

        if not image_files:
            raise ValueError(f"No supported image files found in {folder_path}")

        # Create full paths for images (maintaining sorted order)
        image_paths = [os.path.join(folder_path, f) for f in image_files]

        # Extract video name from path
        video_name = "unknown"
        if "/videos/" in folder_path:
            # Extract video name (e.g., VID02) from path like "/data/jj/datasets/ColecT50/videos/VID02/001202.png"
            video_parts = folder_path.split("/videos/")
            if len(video_parts) > 1:
                video_name = video_parts[1].split("/")[0]

        # Modified timestamp and output file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Get exact model name from config
        exact_model_name = self.config[self.model_name]["model"]
        output_file = os.path.join(output_dir, f"{video_name}_{exact_model_name}_analysis_{timestamp}.json")
        
        # Get model info
        model_info = {
            "model_name": self.model_name,
            "model_config": self.config[self.model_name].copy(),
            "resize_enabled": self.resize,
            "max_size": self.max_size if self.resize else None
        }
        # Remove sensitive info like API keys
        if "api_key" in model_info["model_config"]:
            model_info["model_config"]["api_key"] = "***"

        # Extract video info
        video_info = {
            "video_name": video_name,
            "total_frames": len(image_files),
            "sampling_rate": sampling,
            "frame_count_after_sampling": len(image_files),
            "supported_formats": supported_formats,
            "batch_size": batch_size,
            "folder_path": folder_path
        }

        # Initialize results file with enhanced header
        initial_summary = {
            "video_info": video_info,
            "model_info": model_info,
            "processing_stats": {
                "total_images": len(image_files),
                "timestamp_start": datetime.now().isoformat(),
            },
            "phase_labels": self.phase_labels,
            "results": {}
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(initial_summary, f, indent=2, ensure_ascii=False)

        # Initialize progress bar
        pbar = tqdm(total=len(image_paths), desc="Processing images")
        start_time = datetime.now()
        results = {}

        # Initialize token tracking
        token_stats = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'token_usage_by_image': {}
        }

        def process_single_image(image_path):
            max_retries = 3
            retry_delay = 10  # seconds
            
            for attempt in range(max_retries):
                try:
                    result = self.analyze_local_image(image_path, prompt)
                    return os.path.basename(image_path), result
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return os.path.basename(image_path), f"Error processing image after {max_retries} attempts: {str(e)}"

        # Adjust max workers based on model
        effective_max_workers = 1 if self.model_name == "xai" else MAX_WORKERS
        
        # Process images with results saving
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
            futures_to_paths = {
                executor.submit(process_single_image, path): path 
                for path in image_paths
            }
            
            for future in tqdm(concurrent.futures.as_completed(futures_to_paths), 
                             total=len(image_paths), 
                             desc="Processing images"):
                image_name, result = future.result()
                
                results[image_name] = result
                
                # Add extra delay for XAI between batches
                if self.model_name == "xai":
                    time.sleep(XAI_DELAY_BETWEEN_CALLS)
                
                # Save intermediate results (removed token stats)
                if save_results:
                    current_data = {
                        "processing_stats": {
                            "total_images": len(image_paths),
                            "processed_images": len(results),
                            "timestamp_start": start_time.isoformat(),
                            "last_updated": datetime.now().isoformat()
                        },
                        "phase_labels": self.phase_labels,
                        "results": results
                    }
                    
                    # Atomic write to prevent corruption
                    with open(output_file + '.tmp', 'w', encoding='utf-8') as f:
                        json.dump(current_data, f, indent=2, ensure_ascii=False)
                    os.replace(output_file + '.tmp', output_file)

        # Update final statistics
        end_time = time.time()
        total_time = end_time - start_time.timestamp()
        avg_time_per_image = total_time / len(image_files)

        with open(output_file, 'r+', encoding='utf-8') as f:
            final_data = json.load(f)
            final_data["processing_stats"].update({
                "total_time_seconds": round(total_time, 2),
                "avg_time_per_image_seconds": round(avg_time_per_image, 2),
                "timestamp_end": datetime.now().isoformat(),
                "frames_processed": len(results),
                "successful_frames": len([r for r in results.values() if not str(r).startswith("Error")])
            })
            f.seek(0)
            json.dump(final_data, f, indent=2, ensure_ascii=False)
            f.truncate()

        # Print processing summary
        print(f"\nProcessing Summary:")
        print(f"Total images processed: {len(image_files)}")
        print(f"Total processing time: {round(total_time, 2)} seconds")
        print(f"Average time per image: {round(avg_time_per_image, 2)} seconds")
        print(f"Results saved to: {output_file}")

        # Retry failed images
        failed_images = {
            name: path for name, path in zip(image_files, image_paths)
            if "Error processing image" in results.get(name, "")
        }
        
        if failed_images:
            print(f"\nRetrying {len(failed_images)} failed images...")
            for name, path in failed_images.items():
                try:
                    _, result = process_single_image(path)
                    results[name] = result
                    # Update results file
                    if save_results:
                        current_data = {
                            "processing_stats": {
                                "total_images": len(image_paths),
                                "timestamp_start": start_time.isoformat(),
                                "last_updated": datetime.now().isoformat()
                            },
                            "phase_labels": self.phase_labels,
                            "results": results
                        }
                        with open(output_file + '.tmp', 'w', encoding='utf-8') as f:
                            json.dump(current_data, f, indent=2, ensure_ascii=False)
                        os.replace(output_file + '.tmp', output_file)
                except Exception as e:
                    print(f"Final retry failed for {name}: {str(e)}")

        return results

    def evaluate_predictions(self, predictions, ground_truth):
        """
        Evaluate the phase recognition predictions against ground truth.
        
        Args:
            predictions (dict): Dictionary of {image_name: predicted_phase}
            ground_truth (dict): Dictionary of {image_name: actual_phase}
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        total = len(predictions)
        correct = 0
        phase_wise_accuracy = {phase: {'correct': 0, 'total': 0} for phase in self.phase_labels.keys()}
        
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

    def analyze_json_file(
        self,
        json_file_path: str,
        output_dir: str = None,
        save_results: bool = True
    ) -> Dict[str, str]:
        """
        Analyze images listed in a JSON file with their expected responses.
        """
        # Load JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.dirname(json_file_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract video name from the first image path
        video_name = "unknown"
        if data and data[0]["images"]:
            image_path = data[0]["images"][0]
            # Match pattern like "/data/jj/datasets/ColecT50/videos/VID01/000000.png"
            match = re.search(r'/videos/([^/]+)/', image_path)
            if match:
                video_name = match.group(1)
        
        # Get model info
        model_info = {
            "model_name": self.model_name,
            "model_config": self.config[self.model_name].copy(),
            "resize_enabled": self.resize,
            "max_size": self.max_size if self.resize else None
        }
        # Remove sensitive info like API keys
        if "api_key" in model_info["model_config"]:
            model_info["model_config"]["api_key"] = "***"

        # Extract video info
        video_info = {
            "video_name": video_name,
            "total_frames": len(data),
            "source_file": json_file_path,
            "image_base_path": os.path.dirname(data[0]["images"][0]) if data else None
        }

        # Initialize results
        results = {}
        start_time = datetime.now()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Get exact model name from config
        exact_model_name = self.config[self.model_name]["model"]
        output_file = os.path.join(output_dir, f"{video_name}_{exact_model_name}_analysis_{timestamp}.json")
        
        # Create ground truth dictionary early for evaluation
        ground_truth = {
            os.path.basename(entry['images'][0]): entry['response']
            for entry in data
        }
        
        # Process each entry
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for entry in data:
                image_path = entry['images'][0]
                futures.append(
                    executor.submit(self.analyze_local_image, image_path, self.create_phase_prompt())
                )
            
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                          total=len(futures), 
                                          desc="Processing images")):
                image_path = data[i]['images'][0]
                result = future.result()
                image_name = os.path.basename(image_path)
                results[image_name] = result
                
                # Calculate running accuracy metrics
                current_metrics = self.evaluate_predictions(
                    {k: v for k, v in results.items()}, 
                    ground_truth
                )
                
                # Save intermediate results with accuracy metrics
                if save_results:
                    current_data = {
                        "processing_stats": {
                            "total_images": len(data),
                            "processed_images": len(results),
                            "timestamp_start": start_time.isoformat(),
                            "last_updated": datetime.now().isoformat()
                        },
                        "phase_labels": self.phase_labels,
                        "accuracy_metrics": {
                            "overall_accuracy": current_metrics['overall_accuracy'],
                            "phase_wise_accuracy": current_metrics['phase_wise_accuracy'],
                            "total_correct": current_metrics['correct_predictions'],
                            "total_evaluated": current_metrics['total_images']
                        },
                        "results": results,
                        "ground_truth": ground_truth
                    }
                    
                    # Atomic write to prevent corruption
                    with open(output_file + '.tmp', 'w', encoding='utf-8') as f:
                        json.dump(current_data, f, indent=2, ensure_ascii=False)
                    os.replace(output_file + '.tmp', output_file)
        
        # Calculate final accuracy metrics
        final_metrics = self.evaluate_predictions(results, ground_truth)
        
        # Save final results with complete metrics
        if save_results:
            final_data = {
                "processing_stats": {
                    "total_images": len(data),
                    "processed_images": len(results),
                    "timestamp_start": start_time.isoformat(),
                    "timestamp_end": datetime.now().isoformat(),
                    "total_time_seconds": (datetime.now() - start_time).total_seconds()
                },
                "phase_labels": self.phase_labels,
                "accuracy_metrics": {
                    "overall_accuracy": final_metrics['overall_accuracy'],
                    "phase_wise_accuracy": final_metrics['phase_wise_accuracy'],
                    "total_correct": final_metrics['correct_predictions'],
                    "total_evaluated": final_metrics['total_images']
                },
                "results": results,
                "ground_truth": ground_truth
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        return results

#test
def main():
    parser = argparse.ArgumentParser(description='Analyze surgical phases in images using AI models')
    parser.add_argument('--input', required=True, help='Input image file, directory, or JSON file')
    parser.add_argument('--output-dir', help='Output directory for results (optional)')
    parser.add_argument('--labels-file', required=True, help='JSON file containing phase labels')
    parser.add_argument('--ground-truth', help='JSON file containing ground truth labels for evaluation')
    parser.add_argument('--sampling', type=int, default=1, help='Process every Nth frame (only for folder input)')
    parser.add_argument('--model', default='openai', choices=['openai', 'anthropic', 'google', 'xai'],
                      help='AI model to use for analysis')
    parser.add_argument('--resize', action='store_true', help='Resize images before processing')
    parser.add_argument('--max-size', type=int, default=768, 
                      help='Maximum dimension for image resize (default: 768)')
    
    args = parser.parse_args()

    # Initialize the analyzer with specified model and resize options
    analyzer = ImageAnalyzer(
        model_name=args.model,
        resize=args.resize,
        max_size=args.max_size
    )
    
    # Load phase labels
    with open(args.labels_file, 'r') as f:
        phase_labels = json.load(f)
    analyzer.set_phase_labels(phase_labels)

    # Process input based on type
    if os.path.isfile(args.input):
        if args.input.lower().endswith('.json'):
            # Process JSON file
            results = analyzer.analyze_json_file(
                json_file_path=args.input,
                output_dir=args.output_dir,
                save_results=True
            )
        else:
            # Process single image
            result = analyzer.analyze_local_image(args.input)
            print("\nPredicted phase:", result)
            return
    else:
        # Process directory
        results = analyzer.analyze_folder(
            folder_path=args.input,
            prompt=analyzer.create_phase_prompt(),
            output_dir=args.output_dir,
            save_results=True,
            sampling=args.sampling
        )
    
    # Evaluate if ground truth is provided
    if args.ground_truth:
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
        
        eval_results = analyzer.evaluate_predictions(results, ground_truth)
        print("\nEvaluation Results:")
        print(f"Overall Accuracy: {eval_results['overall_accuracy']:.2%}")
        print("\nPhase-wise Accuracy:")
        for phase, accuracy in eval_results['phase_wise_accuracy'].items():
            print(f"Phase {phase}: {accuracy:.2%}")

if __name__ == "__main__":
    main()