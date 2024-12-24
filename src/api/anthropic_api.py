from anthropic import Anthropic
from ..utils.rate_limiter import anthropic_rate_limit
from .base import BaseAPI

class AnthropicAPI(BaseAPI):
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        return Anthropic(api_key=self.api_key)

    @anthropic_rate_limit
    def process_image(self, image_data: str, prompt: str) -> str:
        """Process an image with Anthropic's API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
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
            return self.clean_response(response.content[0].text)
        except Exception as e:
            raise Exception(f"Error processing with Anthropic: {str(e)}") 