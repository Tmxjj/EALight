"""
Model configuration for VLMAgent.
"""

MODEL_CONFIG = {
    # Default API type to use: "openai_sdk", "gemini_sdk", "requests", or "vertex_sdk"
    "api_type": "requests", 
    
    # Global generation settings
    "temperature": 0, # inference 0；generate golden data for training 0.7
    "max_new_tokens": 4096, # Max tokens to generate

    # Configuration for specific backends
    # "local_model": {
    #     "model_path": "models/Qwen3-VL-8B-SFT-Merged",
    #     "device": "cuda"
    # },
    "openai_sdk": {
        "api_key": "YOUR_OPENAI_API_KEY",
        "base_url": None, # Optional: "https://api.openai.com/v1"
        "model_name": "gpt-4-vision-preview"
    },

    "vertex_sdk": {
        "project_id": "YOUR_VERTEX_PROJECT_ID",
        "location": "global",
        "model_name": "gemini-3.1-pro-preview",
        "credentials_path": "configs/project-api.json" 
    },
    
    "gemini_sdk": {
        "api_key": "YOUR_GEMINI_API_KEY",  # the api don't need to upload to github
        "model_name": "gemini-3.1-pro-preview" # gemini-2.5-pro  gemini-3-pro-preview gemini-3.1-pro-preview
    },
    
    "requests": {
        # 基于 VLLM 的本地服务地址，其中 model_name 应与启动脚本中 --served-model-name 保持一致
        "url": "http://localhost:8000/v1/chat/completions",
        "headers": {"Content-Type": "application/json"},
        "model_name": "qwen3-vl-8b"
    }
}
