import re
from typing import Optional

class ResponseFormatter:
    """Formats and standardizes responses from different AI models."""
    
    @staticmethod
    def format_openai_response(response: str) -> Optional[str]:
        """Format OpenAI response to extract the phase label."""
        try:
            # Remove markdown formatting and clean up the text
            response = re.sub(r'\*\*|\*', '', response)
            response = response.strip()
            
            # Try multiple patterns in order of preference
            patterns = [
                # Match exact single letter A-G
                r'\b([A-G])\b',
                # Match letter followed by dot or parenthesis
                r'([A-G])[\.\)]',
                # Match letter at the start of a line
                r'^([A-G])\.',
                # Match phase label format "X. phase-name"
                r'([A-G])\.\s*(?:gallbladder-dissection|calot-triangle-dissection|clipping-cutting|gallbladder-retraction|cleaning-coagulation|gallbladder-packaging)',
                # Match the letter after common prefixes
                r'(?:phase|option|choice|answer|select|choose)?\s*([A-G])[\.\s\)]',
                # Match the letter in a more relaxed pattern
                r'.*?([A-G])[\.\s].*?(?:gallbladder|calot|clipping|cleaning|preparation)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).upper()
            
            # If no direct letter match, try to find phase names
            phase_mapping = {
                'preparation': 'A',
                'calot-triangle-dissection': 'B',
                'carlot-triangle-dissection': 'B',  # Common misspelling
                'clipping-cutting': 'C',
                'gallbladder-dissection': 'D',
                'gallbladder-packaging': 'E',
                'cleaning-coagulation': 'F',
                'gallbladder-retraction': 'G'
            }
            
            # Convert response to lowercase for case-insensitive matching
            response_lower = response.lower()
            for phase, letter in phase_mapping.items():
                if phase in response_lower:
                    return letter
            
            # Return None to indicate formatting failed
            return None
            
        except Exception as e:
            print(f"Error in formatter (will use 'Z'): {str(e)}")
            return None
    
    @staticmethod
    def format_anthropic_response(response: str) -> Optional[str]:
        """Format Anthropic response to extract the phase label."""
        # Add specific formatting for Anthropic responses
        # Similar pattern matching as OpenAI but adjusted for Anthropic's response format
        pass
    
    @staticmethod
    def format_google_response(response: str) -> Optional[str]:
        """Format Google response to extract the phase label."""
        # Add specific formatting for Google responses
        pass
    
    @staticmethod
    def get_formatter(model: str):
        """Get the appropriate formatter function for the specified model."""
        formatters = {
            'openai': ResponseFormatter.format_openai_response,
            'anthropic': ResponseFormatter.format_anthropic_response,
            'google': ResponseFormatter.format_google_response
        }
        return formatters.get(model.lower()) 