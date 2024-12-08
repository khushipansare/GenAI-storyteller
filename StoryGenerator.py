from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Load the GPT-2 model
model_name = "gpt2"  # You can use "gpt2-medium" or other variants for better quality
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
def generate_story(prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generate a story based on a prompt using GPT-2.
    """
    # Tokenize the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
    )
    
    # Decode the output
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story
if __name__ == "__main__":
    print("Welcome to the AI Story Generator!")
    prompt = input("Enter a starting line for your story: ")
    max_length = int(input("Enter the maximum story length (e.g., 100): "))
    
    # Generate the story
    story = generate_story(prompt, max_length=max_length)
    print("\nGenerated Story:\n")
    print(story)
