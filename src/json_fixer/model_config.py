class MODEL_CONFIG:
    SmolLM2_135M = "unsloth/smollm2-135m-instruct"
    Gemma3_270M = "unsloth/gemma-3-270m-it"
    SmolLM2_360M = "unsloth/smollm2-360m-instruct"
    Qwen3_0_6B = "unsloth/Qwen3-0.6B"
    LFM2_700M = "unsloth/lfm2-700m"


    def get_output_name(model_id):
        return f"{model_id}-json-fixer"