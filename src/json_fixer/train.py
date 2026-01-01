from datasets import Dataset
from json_fixer.convert_to_conversation import convert_to_conversation
import jsonlines
from peft import get_peft_model, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

training_configuration = {
  "lora": {
    "rank": 32,
    "alpha": 32,
    "dropout": 0.0,
    "target_modules": [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ]
  },
  "train": {
    "eval_accumulation_steps": 1, 
    "eval_steps": 100,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2.5e-5,
    "learning_rate_scheduler_type": "cosine",
    "logging_steps": 4,
    "max_length": 2048,
    "num_train_epochs": 6,
    "output_dir": "checkpoints",
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "save_steps": 100,
    "warmup_ratio": 0.05
  }
}

model_id = "unsloth/Qwen3-0.6B"
fine_tuned_model_id = "Qwen3-0.6B-finetuned"
train_dataset_path = "/home/rngo/code/intel-gpu-fine-tune/dataset/train_data.jsonl"
eval_dataset_path = "/home/rngo/code/intel-gpu-fine-tune/dataset/eval_data.jsonl"

with jsonlines.open(train_dataset_path) as j:
  train_dataset = list(j)
converted_train_dataset = [convert_to_conversation(example) for example in train_dataset]

with jsonlines.open(eval_dataset_path) as j:
  eval_dataset = list(j)
converted_eval_dataset = [convert_to_conversation(example) for example in eval_dataset]

model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype=torch.bfloat16
)
# Enable gradient checkpointing compatability with LoRA
model.enable_input_require_grads()

tokenizer = AutoTokenizer.from_pretrained(
  model_id
)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
  r=training_configuration["lora"]["rank"],
  lora_alpha=training_configuration["lora"]["alpha"],
  lora_dropout=training_configuration["lora"]["dropout"],
  bias="none",
  target_modules=training_configuration["lora"]["target_modules"]
)

model = get_peft_model(model, lora_config)


def formatting_prompts_func(examples):
  conversations = examples["conversations"]

  texts = [
    tokenizer.apply_chat_template(
      conversation,
      tokenize=False,
      add_generation_prompt=False
    ) for conversation in conversations
  ]

  return {"text": texts}

train_dataset = Dataset.from_list(converted_train_dataset).map(
  formatting_prompts_func,
  batched=True
)
eval_dataset = Dataset.from_list(converted_eval_dataset).map(
  formatting_prompts_func,
  batched=True
)

trainer = SFTTrainer(
  model=model,
  processing_class=tokenizer,
  train_dataset=train_dataset,
  eval_dataset=eval_dataset,
  args=SFTConfig(
    dataset_text_field="text",
    eval_accumulation_steps=training_configuration["train"]["eval_accumulation_steps"],
    eval_strategy="steps",
    eval_steps=training_configuration["train"]["eval_steps"],
    gradient_accumulation_steps=training_configuration["train"]["gradient_accumulation_steps"],

    # Use this to save some VRAM - instead of saving all the activations, we will recompute dynamically.
    gradient_checkpointing=True,

    # Do not use reentrant way of gradient checkpointing.
    gradient_checkpointing_kwargs={"use_reentrant": False},

    learning_rate=training_configuration["train"]["learning_rate"],
    logging_steps=training_configuration["train"]["logging_steps"],
    lr_scheduler_type=training_configuration["train"]["learning_rate_scheduler_type"],
    max_length=training_configuration["train"]["max_length"],
    num_train_epochs=training_configuration["train"]["num_train_epochs"],
    optim="adamw_torch",
    output_dir=training_configuration["train"]["output_dir"],
    per_device_eval_batch_size=training_configuration["train"]["per_device_eval_batch_size"],
    per_device_train_batch_size=training_configuration["train"]["per_device_train_batch_size"],
    save_steps=training_configuration["train"]["save_steps"],
    save_strategy="steps",
    warmup_ratio=training_configuration["train"]["warmup_ratio"],
    weight_decay=0.01,

    # save some more VRAM
    prediction_loss_only=True
  )
)

trainer.train()

# Save LoRA adapters
model.save_pretrained(fine_tuned_model_id)

# merge LoRA adapters
merged_model = model.merge_and_unload()

# save the full merged model
merged_model.save_pretrained(fine_tuned_model_id)
tokenizer.save_pretrained(fine_tuned_model_id)
