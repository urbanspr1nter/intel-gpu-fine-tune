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
    "learning_rate": 2e-4,
    "learning_rate_scheduler_type": "cosine",
    "logging_steps": 4,
    "max_length": 2048,
    "num_train_epochs": 6,
    "output_dir": "checkpoints",
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "save_steps": 100,
    "warmup_ratio": 0.03,
    "weight_decay": 0.001
  }
}
