import google.generativeai as genai
from ..utils.rate_limiter import google_rate_limit
from .base import BaseAPI

class GoogleAPI(BaseAPI):
    def _initialize_client(self):
        """Initialize the Google client."""
        genai.configure(api_key=self.api_key)
        return genai

    @google_rate_limit
    def process_image(self, image_data: str, prompt: str) -> str:
        """Process an image with Google's API."""
        try:
            model = self.client.GenerativeModel(self.model)
            response = model.generate_content([
                {
                    "mime_type": "image/jpeg",
                    "data": image_data
                },
                prompt
            ])
            return self.clean_response(response.text)
        except Exception as e:
            raise Exception(f"Error processing with Google: {str(e)}") 