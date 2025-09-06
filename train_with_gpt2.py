import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os

# Set wandb to offline mode
os.environ["WANDB_MODE"] = "offline"

def main():
    """
    LoRA fine-tuning with GPT-2 (better for instruction following)
    """
    print("🚀 Starting LoRA Fine-tuning with GPT-2")
    print("="*50)
    
    # Step 1: Load GPT-2 model and tokenizer
    model_name = "gpt2-medium"  # Much better than gpt2 for instruction following
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        use_cache=False  # Required for training
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 2: Load and prepare dataset
    print("\n📊 Loading dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")  # Use full dataset for best results
    
    # Split the dataset 90/10 for train/eval
    print("Splitting dataset into train/eval (90/10)...")
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print(f"✅ Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")
    
    def formatting_prompts_func(example):
        # Simple, clear format for instruction following
        if example['context']:
            text = f"Human: {example['instruction']}\n\nContext: {example['context']}\n\nAssistant: {example['response']}{tokenizer.eos_token}"
        else:
            text = f"Human: {example['instruction']}\n\nAssistant: {example['response']}{tokenizer.eos_token}"
        return {"text": text}
    
    # Format both train and eval datasets
    formatted_train = train_dataset.map(formatting_prompts_func)
    formatted_eval = eval_dataset.map(formatting_prompts_func)
    print(f"✅ Train formatted: {len(formatted_train)} examples")
    print(f"✅ Eval formatted: {len(formatted_eval)} examples")
    
    # Step 3: Tokenize datasets
    print("\n🔤 Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=256,
        )
    
    tokenized_train = formatted_train.map(
        tokenize_function, 
        batched=True,
        remove_columns=formatted_train.column_names
    )
    tokenized_eval = formatted_eval.map(
        tokenize_function, 
        batched=True,
        remove_columns=formatted_eval.column_names
    )
    
    # Step 4: Configure LoRA for GPT-2
    print("\n⚙️ Configuring LoRA for GPT-2...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn"],  # GPT-2 attention modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Step 5: Training arguments
    print("\n🏋️ Setting up training...")
    training_args = TrainingArguments(
        output_dir="./gpt2-medium-fine-tuned",
        per_device_train_batch_size=1,  # Reduced for larger model
        gradient_accumulation_steps=16,  # Increased to maintain effective batch size
        learning_rate=1e-5,  # Lower learning rate for larger model
        num_train_epochs=2,  # Reduced epochs due to larger dataset
        
        # === EVALUATION PARAMETERS ===
        eval_strategy="steps",          # Evaluate every `eval_steps`
        eval_steps=250,                 # Run evaluation every 250 steps
        load_best_model_at_end=True,    # Load the best model at the end of training
        # =============================
        
        logging_steps=250,              # Log metrics at the same frequency as evaluation
        save_steps=250,                 # Save a checkpoint at the same frequency
        fp16=False,
        bf16=False,
        save_total_limit=3,             # Keep more checkpoints
        dataloader_pin_memory=False,
        warmup_ratio=0.05,  # Slightly more warmup
        weight_decay=0.01,
    )
    
    # Step 6: Data collator (THE FIX!)
    # ✅ Using the correct data collator for language modeling.
    # mlm=False specifies that we are doing Causal Language Modeling (not Masked Language Modeling)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train,  # Use the train split
        eval_dataset=tokenized_eval,    # Provide the eval split
        args=training_args,
        data_collator=data_collator,
    )
    
    # Step 7: Train
    print("\n🚀 Starting training...")
    print(f"Training on {len(tokenized_train)} examples")
    print(f"Evaluating on {len(tokenized_eval)} examples")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Evaluation every {training_args.eval_steps} steps")
    
    trainer.train()
    print("✅ Training completed!")
    
    # Step 8: Save model
    print("\n💾 Saving model...")
    model.save_pretrained("./gpt2-medium-fine-tuned")
    tokenizer.save_pretrained("./gpt2-medium-fine-tuned")
    print("✅ Model saved to ./gpt2-medium-fine-tuned/")
    
    # Step 9: Test the model
    print("\n🧪 Testing the model...")
    test_prompt = "Human: What is machine learning?\n\nAssistant:"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test prompt: {test_prompt}")
    print(f"Model response: {response}")

if __name__ == "__main__":
    main()
