"""
Ollama HTTP Client Interface.
Handles asynchronous REST calls to local Ollama service with health checks and timeouts.
"""

import httpx
from typing import Optional, Dict, Any
from backend.app.core.config import settings
from backend.app.core.logging import logger


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = float(settings.OLLAMA_TIMEOUT_SECONDS)

    async def check_health(self) -> bool:
        """Verifies if local Ollama server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False

    async def generate_completion(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Sends a completion request to Ollama."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Sending prompt to Ollama model '{self.model}' at {url}...")
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("response", "")
                else:
                    logger.error(f"Ollama error status {res.status_code}: {res.text}")
                    return None
        except Exception as e:
            logger.warning(f"Ollama API request failed: {e}")
            return None
