import logging
import time
from typing import Any, List, Mapping, Optional

import jwt
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from llms.zhipu import zhipu_api_key

logger = logging.getLogger(__name__)


def generate_token(apikey: str):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + 300 * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


class ZhiPu(LLM):
    """Endpoint URL to use."""
    url: str = "https://open.bigmodel.cn/api/paas/v3/model-api/chatglm_pro/invoke"
    """Top P for nucleus sampling from 0 to 1"""
    top_p: float = 0.7
    """history content"""
    with_history: bool = False
    history: List[dict] = []
    """Key word arguments to pass to the model."""
    model_kwargs: Optional[dict] = None

    @property
    def _llm_type(self) -> str:
        return "zhipu"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"url": self.url},
            **{"model_kwargs": _model_kwargs},
        }

    def _call(
            self,
            prompt: str,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:

        _model_kwargs = self.model_kwargs or {}

        # HTTP headers for authorization
        headers = {"Content-Type": "application/json", "Authorization": generate_token(zhipu_api_key)}

        payload = {
            "prompt": prompt,
            "top_p": self.top_p
        }
        payload.update(_model_kwargs)
        payload.update(kwargs)

        logger.debug(f"zhipu payload: {payload}")

        # call api
        try:
            response = requests.post(self.url, headers=headers, json=payload)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        logger.debug(f"zhipu response: {response}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")

        parsed_response = response.json()
        choices = parsed_response["data"]["choices"]

        if self.with_history:
            self.history = self.history + choices
        return choices[0]["content"]
