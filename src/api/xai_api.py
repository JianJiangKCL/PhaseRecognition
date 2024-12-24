from openai import OpenAI
from ..utils.rate_limiter import xai_rate_limit
from .base import BaseAPI

class XAIAPI(BaseAPI):
    def _initialize_client(self):
        """Initialize the XAI client."""
        return OpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )

    @xai_rate_limit
    def process_image(self, image_data: str, prompt: str) -> str:
        """Process an image with XAI's API."""
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
                            }
                        ],
                    }
                ],
                max_tokens=3000,
            )
            return self.clean_response(response.choices[0].message.content)
        except Exception as e:
            raise Exception(f"Error processing with XAI: {str(e)}") 