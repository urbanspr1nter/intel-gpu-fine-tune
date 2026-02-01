from unsloth import FastLanguageModel
from datasets import Dataset
import jsonlines
from trl import SFTConfig, SFTTrainer

from json_fixer.convert_to_conversation import convert_to_conversation
from json_fixer.model_config import MODEL_CONFIG
from json_fixer.training_config import training_configuration

model_id = MODEL_CONFIG.SmolLM2_135M
fine_tuned_model_id = MODEL_CONFIG.SmolLM2_135M

train_dataset_path = "/home/rngo/code/intel-gpu-fine-tune/dataset/train_data.jsonl"
eval_dataset_path = "/home/rngo/code/intel-gpu-fine-tune/dataset/eval_data.jsonl"

with jsonlines.open(train_dataset_path) as j:
  train_dataset = list(j)
converted_train_dataset = [convert_to_conversation(example) for example in train_dataset]

with jsonlines.open(eval_dataset_path) as j:
  eval_dataset = list(j)
converted_eval_dataset = [convert_to_conversation(example) for example in eval_dataset]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_id,
    max_seq_length=2048,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False  
)

model = FastLanguageModel.get_peft_model(
    model,
    r=training_configuration["lora"]["rank"],
    lora_alpha=training_configuration["lora"]["alpha"],
    lora_dropout=training_configuration["lora"]["dropout"],
    bias="none",
    target_modules=training_configuration["lora"]["target_modules"],
    use_gradient_checkpointing="unsloth",
    use_rslora=False,
    loftq_config=None
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        ) for convo in convos
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
    weight_decay=training_configuration["train"]["weight_decay"],

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