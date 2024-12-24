from abc import ABC, abstractmethod
from typing import Dict, Any
import base64
from PIL import Image
from io import BytesIO

class BaseAPI(ABC):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        """Initialize the API client."""
        pass

    @abstractmethod
    def process_image(self, image_data: str, prompt: str) -> str:
        """Process an image with the API and return the result."""
        pass

    def prepare_image(self, image_path: str, max_size: int = None) -> str:
        """Prepare image for API processing."""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if max_size:
                width, height = img.size
                if width > max_size or height > max_size:
                    if width > height:
                        new_width = max_size
                        new_height = int(height * (max_size / width))
                    else:
                        new_height = max_size
                        new_width = int(width * (max_size / height))
                    img = img.resize((new_width, new_height))
            
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            return base64.b64encode(img_byte_arr).decode('utf-8')

    def clean_response(self, response: str) -> str:
        """Clean up model response to ensure it's just a single letter A-G."""
        if isinstance(response, str):
            cleaned = response.strip()[0].upper()
            if cleaned in 'ABCDEFG':
                return cleaned
        raise ValueError(f"Invalid response format: {response}") 