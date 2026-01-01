# intel-xpu-fine-tune

Demonstrates how to fine-tune a small language model on any Intel Arc GPU using standard `trl` and `torch+xpu`. 

# Resources

I have written two blog articles to get started with setting up the development environment to inference and fine-tune on Intel Arc GPUs:

* [vLLM on Intel Arc GPUs (xpu)](https://www.roger.lol/blog/accessible-ai-vllm-on-intel-arc)
* [Fine-Tuning LLMs on any Intel Arc GPU](https://www.roger.lol/blog/fine-tuning-llms-on-any-intel-arc-gpu)

# Task

Fine-tune Qwen3-0.6B to format and fix invalid JSON.

# Results

The test dataset contains 46 different examples. I performed inference on the base model first to understand the baseline performance. After fine-tuning, I performed inference on the fine-tuned model to see the improvement. 

Results show baseline performance of **54.35%** accuracy and after fine-tuning, **86.97%**. 

<img width="1182" height="882" alt="image" src="https://github.com/user-attachments/assets/60d42422-d535-4171-9410-8b00b0aa5468" />
