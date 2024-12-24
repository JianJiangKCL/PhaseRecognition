import re
from typing import Optional, Dict

class ResponseFormatter:
    """Formats and standardizes responses from different AI models."""
    
    def __init__(self, phase_labels: Dict[str, str]):
        """Initialize with phase labels mapping."""
        self.phase_labels = phase_labels
        # Create reverse mapping from phase name to letter
        self.phase_mapping = {
            name.lower(): letter 
            for letter, name in phase_labels.items()
        }
        # Add common variations and misspellings
        if 'calot-triangle-dissection' in self.phase_mapping.values():
            self.phase_mapping['carlot-triangle-dissection'] = 'B'  # Common misspelling
    
    def format_openai_response(self, response: str) -> str:
        """Format OpenAI response to extract the phase label."""
        try:
            print(f"Raw OpenAI response: {response}")
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
                # Match "X. xxx" format
                r'([A-G])\.\s*\w+',
                # Match phase label format "X. phase-name"
                r'([A-G])\.\s*(?:' + '|'.join(map(re.escape, self.phase_labels.values())) + ')',
                # Match the letter after common prefixes
                r'(?:phase|option|choice|answer|select|choose)?\s*([A-G])[\.\s\)]',
                # Match the letter in a more relaxed pattern
                r'.*?([A-G])[\.\s].*?(?:' + '|'.join(map(re.escape, self.phase_labels.values())) + ')',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).upper()
            
            # If no direct letter match, try to find phase names
            response_lower = response.lower()
            for phase, letter in self.phase_mapping.items():
                if phase in response_lower:
                    return letter
            
            # Return 'Z' if no match found
            print(f"Warning - No phase match found (returning 'Z'): {response}")
            return 'Z'
            
        except Exception as e:
            print(f"Error in formatter (returning 'Z'): {str(e)}")
            return 'Z'
    
    def format_anthropic_response(self, response: str) -> str:
        """Format Anthropic response to extract the phase label."""
        try:
            print(f"Raw Anthropic response: {response}")
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
                # Match "X. xxx" format
                r'([A-G])\.\s*\w+',
                # Match phase label format "X. phase-name"
                r'([A-G])\.\s*(?:' + '|'.join(map(re.escape, self.phase_labels.values())) + ')',
                # Match the letter after common prefixes
                r'(?:phase|option|choice|answer|select|choose)?\s*([A-G])[\.\s\)]',
                # Match the letter in a more relaxed pattern
                r'.*?([A-G])[\.\s].*?(?:' + '|'.join(map(re.escape, self.phase_labels.values())) + ')',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).upper()
            
            # If no direct letter match, try to find phase names
            response_lower = response.lower()
            for phase, letter in self.phase_mapping.items():
                if phase in response_lower:
                    return letter
            
            # Return 'Z' if no match found
            print(f"Warning - No phase match found (returning 'Z'): {response}")
            return 'Z'
            
        except Exception as e:
            print(f"Error in formatter (returning 'Z'): {str(e)}")
            return 'Z'
    
    def format_google_response(self, response: str) -> Optional[str]:
        """Format Google response to extract the phase label."""
        # Add specific formatting for Google responses
        pass
    
    def get_formatter(self, model: str):
        """Get the appropriate formatter function for the specified model."""
        formatters = {
            'openai': self.format_openai_response,
            'anthropic': self.format_anthropic_response,
            'google': self.format_google_response
        }
        return formatters.get(model.lower()) 