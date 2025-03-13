"""

"""

import json
import os
import requests
from typing import Optional, Dict, Any, List

from mlgym.backend.base import BaseModel, ModelArguments
from mlgym.utils.log import get_logger


class ArgoModel(BaseModel):
    """
    Implementation for interacting with the Argo API service.
    
    This model adheres to the Argo Input/Output standard and extends the BaseModel
    to provide cost tracking and token management.
    """
    
    MODELS = {
        "argo": {
            "context_length": 128000,
            "cost_per_input_token": 0.0,  # Update with actual costs if applicable
            "cost_per_output_token": 0.0,  # Update with actual costs if applicable
        }
    }
    
    SHORTCUTS = {
        "argo-default": "argo"
    }
    
    def __init__(self, args: ModelArguments):
        """
        Initialize the ArgoModel with configuration arguments.
        
        Args:
            args (ModelArguments): Configuration for the model
        """
        super().__init__(args)
        self.model_provider = "Argo"
        
        # Set up API endpoint
        self.api_endpoint = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
        
        # Set up API key
        self.api_key = args.api_key or os.getenv("ARGO_API_KEY")
        if not self.api_key:
            self.logger.warning("No API key provided for Argo. If required, set ARGO_API_KEY env var or pass via api_key")
        
        # Get username for Argo API
        self.username = os.getenv("ANL_USER") or os.getenv("USER")
        if not self.username:
            self.logger.warning("No username found. Set ANL_USER env var for proper identification")
    
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for the given text.
        
        This is a simple approximation; for production use, implement a proper tokenization method.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Estimated token count
        """
        # Simple approximation (~4 chars per token)
        return len(text) // 4
    
    def query(self, history: List[Dict[str, str]]) -> str:
        """
        Query the Argo model with a conversation history.
        
        Args:
            history (List[Dict[str, str]]): List of conversation turns
            
        Returns:
            str: Model's response
        """
        # Format request payload according to Argo Input Standard
        payload = {
            "user": self.username,
            "model": self.api_model,
            "messages": history,
            "stop": self.args.stop,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            **self.args.completion_kwargs  # Add any additional kwargs
        }
        
        # Count input tokens
        input_text = json.dumps(history)
        input_tokens = self._count_tokens(input_text)

        if self.api_model == 'gpto1preview':
            for message in payload['messages']:
                if message['role'] == 'system':
                    message['role'] = 'user'
        
        
        # Make API call
        self.logger.info(f"Sending request to Argo API with {len(history)} messages")
        response = requests.post(
            self.api_endpoint, 
            json=payload,
        )

        
        if response.status_code != 200:
            self.logger.error(f"Error from Argo API: {response.status_code} {response.text}")
            return f"Error: {response.status_code} - {response.text}"
        
        # Parse response
        response_data = response.json()
        response_text = response_data.get("response", "")
        
        # Count output tokens
        output_tokens = self._count_tokens(response_text)
        
        # Update statistics
        self.update_stats(input_tokens, output_tokens)
        
        return response_text