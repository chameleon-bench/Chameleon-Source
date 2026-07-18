

import os
import sys
import time
import json
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import get_logger

# Optional imports, based on provider used
try:
    from openai import OpenAI, AzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """LLM response data class."""
    content: str                          # Generated content
    model: str                            # Model used
    provider: str                         # Provider
    prompt_tokens: int                    # Input token count
    completion_tokens: int                # Output token count
    total_tokens: int                     # Total token count
    cost: float                           # Estimated cost (USD)
    latency: float                        # Response latency (seconds)
    raw_response: Optional[Dict] = None   # Raw response
    prompt_cache_hit_tokens: int = 0      # Cache hit prompt token count (DeepSeek/OpenAI)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': self.content,
            'model': self.model,
            'provider': self.provider,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'cost': self.cost,
            'latency': self.latency,
            'prompt_cache_hit_tokens': self.prompt_cache_hit_tokens,
        }


class LLMClient:
    """
    Unified LLM client
    
    Supports multiple LLM providers with a consistent calling interface
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config_path: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider (openai, azure, anthropic)
            model: Model name
            config_path: Config file path
            api_key: API Key (read from environment variable by default)
            **kwargs: Other configuration parameters
        """
        # Load config
        self.config = self._load_config(config_path)
        
        # Set provider
        self.provider = provider or self.config.get('default_provider', 'openai')
        self.provider_config = self.config['providers'].get(self.provider, {})
        
        if not self.provider_config:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        # Set model
        self.model = model or self.provider_config.get('default_model')
        self.model_config = self.provider_config.get('models', {}).get(self.model, {})
        
        # Set API Key
        self.api_key = api_key or self._get_api_key()
        
        # Initialize client
        self._client = None
        self._init_client()
        
        # Retry config
        self.retry_config = self.config.get('retry', {})
        
        # Output config
        self.output_config = self.config.get('output', {})
        
        # Configure logging
        logger.info(f"LLMClient initialized: provider={self.provider}, model={self.model}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load config file."""
        if config_path is None:
            # Default config file path
            config_path = Path(__file__).parent.parent / 'config' / 'llm_config.yaml'
        
        if not Path(config_path).exists():
            logger.warning(f"Config file not found: {config_path}, using default config")
            return self._default_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict:
        """Default config."""
        return {
            'default_provider': 'openai',
            'providers': {
                'openai': {
                    'api_key_env': 'OPENAI_API_KEY',
                    'base_url': 'https://api.openai.com/v1',
                    'default_model': 'gpt-4-turbo',
                }
            },
            'retry': {
                'max_retries': 3,
                'initial_delay': 1.0,
                'exponential_base': 2.0,
            }
        }
    
    def _get_api_key(self) -> str:
        """Get API Key.
        
        Supports three methods:
        1. Passed via init parameter (highest priority)
        2. Read api_key field directly from config file
        3. Read from environment variable (via api_key_env config)
        """
        # First check if api_key was passed during init
        if hasattr(self, 'api_key') and self.api_key:
            return self.api_key
        
        # Prefer reading direct api_key from config
        api_key = self.provider_config.get('api_key')
        if api_key:
            return api_key
            
        # Get environment variable name from config
        api_key_env = self.provider_config.get('api_key_env', f'{self.provider.upper()}_API_KEY')
        
        # Try to read from environment variable
        api_key = os.getenv(api_key_env)
        
        # If env var not found, check if config value itself is an API Key
        if not api_key:
            # Check if config value looks like an API Key (starts with sk- or length > 20)
            if api_key_env.startswith('sk-') or len(api_key_env) > 20:
                # Assume config value is the API Key itself
                api_key = api_key_env
            else:
                # Config value is not an API Key, need to set environment variable
                raise ValueError(
                    f"API Key not found. Please set environment variable: {api_key_env} "
                    f"or set api_key field in config file"
                )
        
        return api_key
    
    def _init_client(self):
        """Initialize underlying client."""
        if self.provider == 'openai':
            if not HAS_OPENAI:
                raise ImportError("Please install openai library: pip install openai")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider_config.get('base_url')
            )
        
        elif self.provider == 'azure':
            if not HAS_OPENAI:
                raise ImportError("Please install openai library: pip install openai")
            
            self._client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.provider_config.get('base_url'),
                api_version=self.provider_config.get('api_version', '2024-02-01')
            )
        
        elif self.provider == 'anthropic':
            if not HAS_ANTHROPIC:
                raise ImportError("Please install anthropic library: pip install anthropic")
            
            self._client = anthropic.Anthropic(api_key=self.api_key)
        
        elif self.provider == 'deepseek':
            # DeepSeek official API, compatible with OpenAI API format
            if not HAS_OPENAI:
                raise ImportError("Please install openai library: pip install openai")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider_config.get('base_url')
            )
        
        elif self.provider == 'aliyun':
            # Alibaba Cloud Bailian compatible with OpenAI API format, uses OpenAI library
            if not HAS_OPENAI:
                raise ImportError("Please install openai library: pip install openai")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider_config.get('base_url')
            )
        
        elif self.provider == 'siliconflow':
            # SiliconFlow API, compatible with OpenAI API format
            if not HAS_OPENAI:
                raise ImportError("Please install openai library: pip install openai")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider_config.get('base_url')
            )
        
        elif self.provider == 'xiaomi':
            # Xiaomi MiMo API, compatible with OpenAI API format
            if not HAS_OPENAI:
                raise ImportError("Please install openai library: pip install openai")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider_config.get('base_url')
            )
        
        elif self.provider == 'local':
            # Local/intranet API, compatible with OpenAI API format
            if not HAS_OPENAI:
                raise ImportError("Please install openai library: pip install openai")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider_config.get('base_url')
            )
        
        elif self.provider == 'aihubmix':
            # AiHubMix relay platform, compatible with OpenAI API format
            if not HAS_OPENAI:
                raise ImportError("Please install openai library: pip install openai")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider_config.get('base_url')
            )
        
        elif self.provider == 'zhipu':
            # Zhipu AI, compatible with OpenAI API format
            if not HAS_OPENAI:
                raise ImportError("Please install openai library: pip install openai")
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.provider_config.get('base_url')
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    @staticmethod
    def _extract_content(content) -> str:
        """
        Extract plain text content from message.content
        
        Alibaba Cloud qwen3/3.5 series with enable_thinking enabled, content may return as list:
        [
            {"type": "thinking", "thinking": "...thinking process..."},
            {"type": "text", "text": "...formal answer..."}
        ]
        In normal mode, content is a str.
        
        This method always returns str.
        """
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            # Prefer extracting type=text blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                    elif 'text' in block and block.get('type') != 'thinking':
                        text_parts.append(block.get('text', ''))
                elif isinstance(block, str):
                    text_parts.append(block)
            
            if text_parts:
                return '\n'.join(text_parts)
            
            # If no text block, concatenate all non-thinking content
            all_parts = []
            for block in content:
                if isinstance(block, dict):
                    all_parts.append(block.get('text', block.get('thinking', str(block))))
                elif isinstance(block, str):
                    all_parts.append(block)
            return '\n'.join(all_parts) if all_parts else str(content)
        
        return str(content) if content is not None else ''
    
    def complete(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        save_response: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Call LLM to generate text
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt (required)
            temperature: Temperature parameter (creativity)
            max_tokens: Max generation token count
            save_response: Whether to save response
            **kwargs: Other parameters
        
        Returns:
            LLMResponse object
        
        Example:
            response = client.complete(
                system_prompt="You are a SQL database expert.",
                user_prompt="Please briefly introduce the main differences between MySQL and PostgreSQL."
            )
        """
        if user_prompt is None:
            raise ValueError("user_prompt parameter is required")
        
        # Use model config or passed parameters
        temp = temperature if temperature is not None else self.model_config.get('temperature', 0.3)
        max_tok = max_tokens if max_tokens is not None else self.model_config.get('max_tokens', 4096)
        
        # Record start time
        start_time = time.time()
        
        # Retry logic
        last_exception = None
        max_retries = self.retry_config.get('max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                if self.provider in ['openai', 'azure', 'aliyun', 'local', 'deepseek', 'siliconflow', 'xiaomi', 'zhipu', 'aihubmix']:
                    # Alibaba Cloud Bailian / Local API / SiliconFlow / Xiaomi MiMo / AiHubMix compatible with OpenAI API format
                    response = self._call_openai(user_prompt, system_prompt, temp, max_tok, **kwargs)
                elif self.provider == 'anthropic':
                    response = self._call_anthropic(user_prompt, system_prompt, temp, max_tok, **kwargs)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                
                # Calculate latency
                latency = time.time() - start_time
                response.latency = latency
                
                # Save response
                if save_response and self.output_config.get('save_raw_response'):
                    self._save_response(user_prompt, response, system_prompt)
                
                return response
                
            except Exception as e:
                last_exception = e
                # Check if it is a rate limit error (429 / rate limit)
                err_str = str(e).lower()
                is_rate_limit = '429' in err_str or 'rate' in err_str or 'limit' in err_str or 'too many' in err_str or 'quota' in err_str
                
                if is_rate_limit:
                    logger.warning(f"⏳ Rate limited (attempt {attempt + 1}/{max_retries}): {e}")
                else:
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    if is_rate_limit:
                        # Rate limited: wait longer (30s, 60s)
                        delay = 30.0 * (2.0 ** attempt)
                    else:
                        # Normal error: short exponential backoff
                        delay = self.retry_config.get('initial_delay', 1.0) * (
                            self.retry_config.get('exponential_base', 2.0) ** attempt
                        )
                    delay = min(delay, self.retry_config.get('max_delay', 120.0))
                    logger.info(f"  Waiting {delay:.0f}s s before retry...")
                    time.sleep(delay)
        
        # All retries failed
        raise last_exception or Exception("LLM call failed")
    
    def complete_n(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = None,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Call LLM to generate multiple independent completions (OpenAI compatible API only)
        
        By setting the n parameter, one API call returns n independent completions.
        Each completion has a different random seed, naturally providing diversity.
        
        Note: when n > 1, output token cost is multiplied by n.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt (required)
            n: Number of completions (default 1)
            temperature: Temperature parameter
            max_tokens: Max generation token count
            **kwargs: Other parameters
        
        Returns:
            List[LLMResponse]: n independent LLMResponse objects
        """
        if user_prompt is None:
            raise ValueError("user_prompt parameter is required")
        
        if n <= 1:
            return [self.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )]
        
        # n > 1 only supports OpenAI compatible API
        if self.provider not in ['openai', 'azure', 'aliyun', 'local', 'deepseek', 'siliconflow', 'xiaomi', 'aihubmix']:
            logger.warning(
                f"Provider {self.provider} does not support n > 1, falling back to serial calls {n} times"
            )
            results = []
            for _ in range(n):
                resp = self.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                results.append(resp)
            return results
        
        # Each provider's n parameter limit
        max_n_per_call = self.provider_config.get('max_n', 4)  # default 4 (Alibaba Cloud limit)
        
        # If n exceeds provider limit, auto-batch calls
        if n > max_n_per_call:
            logger.info(
                f"n={n} exceeds provider limit {max_n_per_call}，"
                f"auto-batch: {n // max_n_per_call} times ×{max_n_per_call}"
                f"{f' + 1 times ×{n % max_n_per_call}' if n % max_n_per_call else ''}"
            )
            all_results = []
            remaining = n
            while remaining > 0:
                batch_n = min(remaining, max_n_per_call)
                batch_results = self._complete_n_single(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    n=batch_n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                all_results.extend(batch_results)
                remaining -= batch_n
            return all_results
        
        return self._complete_n_single(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def _complete_n_single(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = None,
        n: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Single API call to get n completions (n must be within provider limit)
        """
        temp = temperature if temperature is not None else self.model_config.get('temperature', 0.3)
        max_tok = max_tokens if max_tokens is not None else self.model_config.get('max_tokens', 4096)
        
        start_time = time.time()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        model = self.provider_config.get('deployment_name', self.model)
        
        # Retry logic
        last_exception = None
        max_retries = self.retry_config.get('max_retries', 3)
        
        # Read thinking settings from model config
        # Alibaba Cloud: extra_body={enable_thinking: false/true}
        # DeepSeek: extra_body={"thinking": {"type": "enabled/disabled"}} + reasoning_effort="high"/"max"
        thinking_config = self.model_config.get('thinking', {})
        thinking_enabled = thinking_config.get('enabled', None)
        
        extra_body = kwargs.pop('extra_body', {})
        if thinking_enabled is not None:
            if self.provider in ('aliyun', 'siliconflow'):
                extra_body.setdefault('enable_thinking', thinking_enabled)
                # Thinking depth control: thinking_budget (limit thinking token count)
                thinking_budget = thinking_config.get('thinking_budget', None)
                if thinking_budget is not None and thinking_enabled:
                    extra_body.setdefault('thinking_budget', thinking_budget)
            elif self.provider in ('deepseek', 'xiaomi', 'zhipu'):
                extra_body.setdefault('thinking', {'type': 'enabled' if thinking_enabled else 'disabled'})
        
        for attempt in range(max_retries):
            try:
                create_kwargs = dict(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    n=n,
                    **kwargs
                )
                if extra_body:
                    create_kwargs['extra_body'] = extra_body
                # DeepSeek thinking effort control (top-level parameter, not extra_body)
                if self.provider == 'deepseek' and thinking_enabled:
                    effort = thinking_config.get('effort', 'high')
                    create_kwargs.setdefault('reasoning_effort', effort)
                
                response = self._client.chat.completions.create(**create_kwargs)
                
                latency = time.time() - start_time
                
                # Extract cache hit tokens (DeepSeek / OpenAI compatible)
                cache_hit = self._extract_cache_hit_tokens(response.usage)
                
                # Calculate cost (output tokens is sum of all choices)
                cost = self._calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    cache_hit_tokens=cache_hit
                )
                
                # Create LLMResponse for each choice
                results = []
                for i, choice in enumerate(response.choices):
                    results.append(LLMResponse(
                        content=self._extract_content(choice.message.content),
                        model=self.model,
                        provider=self.provider,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens // n,
                        total_tokens=response.usage.total_tokens // n,
                        cost=cost / n,
                        latency=latency,
                        prompt_cache_hit_tokens=cache_hit,
                        raw_response=None,
                    ))
                
                logger.info(
                    f"complete_n(n={n}): {response.usage.total_tokens} tokens, "
                    f"{latency:.1f}s, ${cost:.4f}"
                )
                
                return results
                
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                is_rate_limit = '429' in err_str or 'rate' in err_str or 'limit' in err_str or 'too many' in err_str or 'quota' in err_str
                
                if is_rate_limit:
                    logger.warning(f"⏳ complete_n Rate limited (attempt {attempt + 1}/{max_retries}): {e}")
                else:
                    logger.warning(f"complete_n call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    if is_rate_limit:
                        delay = 30.0 * (2.0 ** attempt)
                    else:
                        delay = self.retry_config.get('initial_delay', 1.0) * (
                            self.retry_config.get('exponential_base', 2.0) ** attempt
                        )
                    delay = min(delay, self.retry_config.get('max_delay', 120.0))
                    logger.info(f"  Waiting {delay:.0f}s s before retry...")
                    time.sleep(delay)
        
        raise last_exception or Exception("complete_n call failed")
    
    def _call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Call OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Azure uses deployment_name as model
        model = self.provider_config.get('deployment_name', self.model)
        
        # Read thinking settings from model config
        # Alibaba Cloud: extra_body={enable_thinking: false/true}
        # DeepSeek: extra_body={"thinking": {"type": "enabled/disabled"}} + reasoning_effort="high"/"max"
        thinking_config = self.model_config.get('thinking', {})
        thinking_enabled = thinking_config.get('enabled', None)
        
        extra_body = kwargs.pop('extra_body', {})
        if thinking_enabled is not None:
            if self.provider in ('aliyun', 'siliconflow'):
                extra_body.setdefault('enable_thinking', thinking_enabled)
                # Thinking depth control: thinking_budget (limit thinking token count)
                thinking_budget = thinking_config.get('thinking_budget', None)
                if thinking_budget is not None and thinking_enabled:
                    extra_body.setdefault('thinking_budget', thinking_budget)
            elif self.provider in ('deepseek', 'xiaomi', 'zhipu'):
                extra_body.setdefault('thinking', {'type': 'enabled' if thinking_enabled else 'disabled'})
        
        # Reasoning models like gpt-5-nano: use max_completion_tokens instead of max_tokens, do not pass temperature
        use_max_completion_tokens = self.model_config.get('use_max_completion_tokens', False)
        
        if use_max_completion_tokens:
            create_kwargs = dict(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs
            )
        else:
            create_kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        if extra_body:
            create_kwargs['extra_body'] = extra_body
        # DeepSeek thinking effort control (top-level parameter, not extra_body)
        if self.provider == 'deepseek' and thinking_enabled:
            effort = thinking_config.get('effort', 'high')
            create_kwargs.setdefault('reasoning_effort', effort)
        
        # Check if model needs stream call (e.g., qwen3-8b enable_thinking only supports stream)
        use_stream = self.model_config.get('stream', False)
        # If extra_body explicitly disables thinking, no need to force stream
        if extra_body.get('enable_thinking') is False:
            use_stream = False
        
        if use_stream:
            # Stream call, collect complete response
            create_kwargs['stream'] = True
            create_kwargs['stream_options'] = {"include_usage": True}
            
            stream = self._client.chat.completions.create(**create_kwargs)
            
            collected_content = []
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            for chunk in stream:
                # Extract usage (usually in last chunk)
                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens
                
                # Extract content delta
                if chunk.choices:
                    for choice in chunk.choices:
                        delta = choice.delta
                        if delta and delta.content:
                            collected_content.append(delta.content)
            
            full_content = ''.join(collected_content)
            
            # Calculate cost
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            
            return LLMResponse(
                content=full_content,
                model=self.model,
                provider=self.provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
                latency=0.0,  # will be set by outer layer
                prompt_cache_hit_tokens=0,
                raw_response=None
            )
        
        response = self._client.chat.completions.create(**create_kwargs)
        
        # Extract cache hit tokens (DeepSeek / OpenAI compatible)
        cache_hit = self._extract_cache_hit_tokens(response.usage)
        
        # Calculate cost
        cost = self._calculate_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            cache_hit_tokens=cache_hit
        )
        
        return LLMResponse(
            content=self._extract_content(response.choices[0].message.content),
            model=self.model,
            provider=self.provider,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost=cost,
            latency=0.0,  # will be set by outer layer
            prompt_cache_hit_tokens=cache_hit,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )
    
    def _call_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Call Anthropic API."""
        messages = [{"role": "user", "content": prompt}]
        
        response = self._client.messages.create(
            model=self.model,
            messages=messages,
            system=system_prompt or "",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Calculate cost
        cost = self._calculate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            cost=cost,
            latency=0.0,
            prompt_cache_hit_tokens=0,  # Anthropic cache priced separately by prompt caching
            raw_response={
                'id': response.id,
                'model': response.model,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            }
        )
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int,
                        cache_hit_tokens: int = 0) -> float:
        """
        Calculate API call cost, supports differential pricing for cache hit tokens
        
        General logic:
          - input_tokens includes sum of cache_hit and cache_miss
          - cache_hit_tokens portion uses cost_per_1k_input_cached (cheaper)
          - Remaining (input_tokens - cache_hit_tokens) uses cost_per_1k_input (normal price)
          - If cached price not configured, all at normal price (compatible with all providers)
        """
        cached_cost_per_1k = self.model_config.get('cost_per_1k_input_cached', None)
        
        if cache_hit_tokens > 0 and cached_cost_per_1k is not None:
            # Cache hit with cache price: split into two parts
            cache_miss_tokens = input_tokens - cache_hit_tokens
            cache_hit_cost = (cache_hit_tokens / 1000) * cached_cost_per_1k
            cache_miss_cost = (cache_miss_tokens / 1000) * self.model_config.get('cost_per_1k_input', 0)
            input_cost = cache_hit_cost + cache_miss_cost
        else:
            # No cache hit or no cache price: all at normal price
            input_cost = (input_tokens / 1000) * self.model_config.get('cost_per_1k_input', 0)
        
        output_cost = (output_tokens / 1000) * self.model_config.get('cost_per_1k_output', 0)
        return input_cost + output_cost
    
    def _extract_cache_hit_tokens(self, usage) -> int:
        """
        Extract cache hit token count from OpenAI compatible usage object
        
        Compatible with multiple return formats:
          - DeepSeek: prompt_cache_hit_tokens (top-level field)
          - OpenAI: prompt_tokens_details.cached_tokens
          - Other providers: may be None or not exist
        """
        try:
            # DeepSeek style: top-level field
            hit = getattr(usage, 'prompt_cache_hit_tokens', None)
            if hit is not None and hit > 0:
                return hit
            
            # OpenAI style: prompt_tokens_details.cached_tokens
            details = getattr(usage, 'prompt_tokens_details', None)
            if details is not None:
                cached = getattr(details, 'cached_tokens', None)
                if cached is not None and cached > 0:
                    return cached
        except Exception:
            pass
        
        return 0
    
    def _save_response(self, user_prompt: str, response: LLMResponse, system_prompt: Optional[str] = None):
        """Save response to file."""
        if not self.output_config.get('save_raw_response'):
            return
        
        save_dir = Path(self.output_config.get('response_dir', './output/llm_responses'))
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        # Model name may contain / (e.g., Qwen/Qwen3-8B), replace with _ to avoid path errors
        safe_model_name = self.model.replace('/', '_')
        filename = f"{self.provider}_{safe_model_name}_{timestamp}.json"
        filepath = save_dir / filename
        
        data = {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'response': response.to_dict(),
            'raw_response': response.raw_response,
            'timestamp': timestamp
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Response saved: {filepath}")
    
    @classmethod
    def for_task(cls, task_type: str, **kwargs) -> 'LLMClient':
        """
        Create client based on task type
        
        Args:
            task_type: Task type (simple, standard, complex, creative)
        """
        config_path = Path(__file__).parent.parent / 'config' / 'llm_config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        task_model = config.get('task_models', {}).get(task_type, 'gpt-4-turbo')
        
        # Parse model name, determine provider
        if task_model.startswith('claude'):
            provider = 'anthropic'
        elif 'aliyun' in task_model.lower() or 'qwen' in task_model.lower():
            provider = 'aliyun'
        else:
            provider = 'openai'
        
        return cls(provider=provider, model=task_model, **kwargs)
    
    @classmethod
    def simple_complete(
        cls,
        system_prompt: str,
        user_prompt: str,
        provider: str = "openai",
        model: str = "gpt-4-turbo",
        **kwargs
    ) -> str:
        """
        Simple LLM call, returns only content string
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            provider: Provider
            model: Model
            **kwargs: Other parameters
        
        Returns:
            Generated text content
        """
        client = cls(provider=provider, model=model, **kwargs)
        response = client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs
        )
        return response.content


# Convenience functions
def quick_complete(
    prompt: str,
    provider: str = "openai",
    model: str = "gpt-4-turbo",
    **kwargs
) -> str:
    """
    Quick LLM call, returns only content string (legacy API, compatibility)
    
    Args:
        prompt: Prompt
        provider: Provider
        model: Model
        **kwargs: Other parameters
    
    Returns:
        Generated text content
    
    Note:
        Recommend using LLMClient.simple_complete(system_prompt, user_prompt)
    """
    client = LLMClient(provider=provider, model=model)
    response = client.complete(prompt, **kwargs)
    return response.content
