from openai import OpenAI
from ..utils.rate_limiter import openai_rate_limit
from .base import BaseAPI

class OpenAIAPI(BaseAPI):
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        return OpenAI(api_key=self.api_key)

    @openai_rate_limit
    def process_image(self, image_data: str, prompt: str) -> str:
        """Process image with OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
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
                            },
                        ],
                    }
                ],
                max_tokens=50,
            )
            
            # Return the raw response text for the formatter to handle
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Error processing with OpenAI: {str(e)}")

    def clean_response(self, response: str) -> str:
        """Clean the response if needed, but don't validate format."""
        return response.strip() 