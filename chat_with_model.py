import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Set wandb to offline mode
os.environ["WANDB_MODE"] = "offline"

def load_fine_tuned_model():
    """Load the fine-tuned model with LoRA adapters"""
    print("Loading fine-tuned model...")
    
    # Load the base model
    base_model_name = "gpt2-medium"
    
    # Try MPS first, fallback to CPU if there are issues
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto"
        )
        print("✅ Model loaded on MPS device")
    except Exception as e:
        print(f"⚠️  MPS failed, using CPU: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="cpu"
        )
        print("✅ Model loaded on CPU")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the LoRA adapters
    model = PeftModel.from_pretrained(model, "./gpt2-medium-fine-tuned")
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, instruction, context="", max_length=150):
    """Generate a response from the model"""
    
    # Format the prompt in the same way as training data
    if context:
        prompt = f"Human: {instruction}\nContext: {context}\nAssistant:"
    else:
        prompt = f"Human: {instruction}\nAssistant:"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Reduce repetition
            no_repeat_ngram_size=2   # Avoid repeating 2-grams
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (after "Assistant:")
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    else:
        # If format is not found, just take everything after the prompt
        response = response[len(prompt):].strip()
    
    return response

def chat_with_model():
    """Interactive chat with the fine-tuned model"""
    
    # Load the model
    model, tokenizer = load_fine_tuned_model()
    
    print("\n" + "="*60)
    print("🤖 CHAT WITH YOUR FINE-TUNED MODEL")
    print("="*60)
    print("Type 'quit' to exit, 'help' for examples")
    print("="*60)
    
    while True:
        print("\n" + "-"*40)
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'help':
            print("\n📝 Example questions you can ask:")
            print("• What is machine learning?")
            print("• Write a Python function to calculate factorial")
            print("• Explain photosynthesis in simple terms")
            print("• What are the benefits of exercise?")
            print("• How do I create a list in Python?")
            print("• What is the capital of France?")
            continue
        elif not user_input:
            print("Please enter a question!")
            continue
        
        print("\n🤖 Model: ", end="", flush=True)
        
        try:
            response = generate_response(model, tokenizer, user_input)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Try asking a different question.")

def compare_with_original():
    """Compare fine-tuned model with original model"""
    
    print("Loading both models for comparison...")
    
    # Load fine-tuned model
    fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model()
    
    # Load original model
    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
    original_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token
    
    print("✅ Both models loaded!")
    
    print("\n" + "="*60)
    print("🆚 COMPARISON: FINE-TUNED vs ORIGINAL")
    print("="*60)
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        print("\n" + "-"*40)
        user_input = input("Your question: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif not user_input:
            print("Please enter a question!")
            continue
        
        print(f"\n📝 Question: {user_input}")
        print("\n" + "="*50)
        
        # Fine-tuned model response
        print("🤖 Fine-tuned Model:")
        try:
            ft_response = generate_response(fine_tuned_model, fine_tuned_tokenizer, user_input)
            print(ft_response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-"*30)
        
        # Original model response
        print("🔵 Original Model:")
        try:
            orig_response = generate_response(original_model, original_tokenizer, user_input)
            print(orig_response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    print("Choose your mode:")
    print("1. Chat with fine-tuned model only")
    print("2. Compare fine-tuned vs original model")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        compare_with_original()
    else:
        chat_with_model()
