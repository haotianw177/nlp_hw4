import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
output_file = open('part1_outputs.txt', 'w')
output_file2 = open('part2_outputs.txt', 'w')
output_file3 = open('part3_outputs.txt', 'w')

class LanguageModel:
    
    def __init__(self, model_name='gpt2', device=None, mode='greedy', k=None, p=None, temperature=1.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.mode = mode
        self.k = k
        self.p = p
        self.temperature = temperature
    
    def start(self, text):
        """Tokenize input string and return model-ready tensors."""
        inputs = self.tokenizer(text, return_tensors='pt') # adds BOS/EOS by default, returns numberized tokens
        return {k: v.to(self.device) for k, v in inputs.items()}

    def step(self, state):
        """Perform one decoding step given current state."""
        with torch.no_grad():
            outputs = self.model(**state)
            next_token = self.decoding_algorithm(outputs)
            # Append new token to input
            state['input_ids'] = torch.cat([state['input_ids'], next_token.unsqueeze(0)], dim=1)
            state['attention_mask'] = torch.cat(
                [state['attention_mask'], torch.ones((1,1), device=self.device)], dim=1
            )
        return state

    def decoding_algorithm(self, outputs):
        """Choose the next token according to the selected decoding strategy."""
        
        # TODO: use self.temperature to incorporate temperature sampling
        logits = outputs.logits[:, -1, :]
        logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        if self.mode == 'greedy':
            # finds the index of the token with the highest probability
            next_token = torch.argmax(logits, dim=-1)
        
        elif self.mode == 'sampling':
            # Apply ancestral sampling - sample from the full probability distribution
            # torch.multinomial samples from the probability distribution
            # num_samples=1 means we sample one token according to the probabilities
            next_token = torch.multinomial(probs, num_samples=1)

        elif self.mode == 'top-k':
    # Get top k tokens - torch.topk returns (values, indices) of top k
            top_k_probs, top_k_indices = torch.topk(probs, self.k)
            # Sample from top k (no renormalization needed since multinomial handles it)
            selected_index = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_indices[selected_index]

        elif self.mode == 'top-p':
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            # Find where cumulative probability first exceeds p
            # Keep tokens up to that point
            keep_mask = cumulative_probs <= self.p
            # Ensure we keep at least one token
            if not keep_mask.any():
                keep_mask[0] = True
            # Get probabilities of kept tokens and sample
            kept_probs = sorted_probs[keep_mask]
            selected_index = torch.multinomial(kept_probs, num_samples=1)
            next_token = sorted_indices[keep_mask][selected_index]

        return next_token

    # The `generate()` method below is NOT HuggingFace's built-in `.generate()`.
    # It simply runs our custom decoding loop using your implementation of greedy search, sampling, top-k, and top-p. 
    # You may NOT use `model.generate()` from the HuggingFace Transformers library.
    def generate(self, prompt, max_new_tokens=40):
        """Generate a continuation from a given prompt."""
        state = self.start(prompt)
        for _ in range(max_new_tokens):
            state = self.step(state)
        output_ids = state['input_ids'].squeeze().tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

if __name__ == '__main__':
    with open('storycloze-2018/short_context_data.txt') as f:
        contexts = [line.strip() for line in f if line.strip()]
    # TODO: run the model with different decoding methods and print the outputs (as outlined in the assignment)
    # lm = LanguageModel(mode=...)
    lm_greedy = LanguageModel(mode='greedy')
    # lm.generate(...)

    output_file.write("=== GREEDY SEARCH RESULTS (Part 1) ===\n\n")
    for i, context in enumerate(contexts[:10]):
        result = lm_greedy.generate(context, max_new_tokens=40)
        print(f"Example {i+1}:")
        print(f"Context: {context}")
        print(f"Generated: {result}\n")
        
        output_file.write(f"Example {i+1}:\n")
        output_file.write(f"Context: {context}\n")
        output_file.write(f"Generated: {result}\n\n")
    
    # output_file.close()
    # print("\nOutputs also saved to 'part1_outputs.txt'")

    lm_sampling = LanguageModel(mode='sampling')
    output_file.write("\n\n=== ANCESTRAL SAMPLING RESULTS (Part 1) ===\n\n")
    for i, context in enumerate(contexts[:10]):
        result = lm_sampling.generate(context, max_new_tokens=40)
        print(f"Example {i+1} (Sampling):")
        print(f"Context: {context}")
        print(f"Generated: {result}\n")
        
        output_file.write(f"Example {i+1}:\n")
        output_file.write(f"Context: {context}\n")
        output_file.write(f"Generated: {result}\n\n")
    
    output_file.close()
    print("\nOutputs also saved to 'part1_outputs.txt'")

    k_value = 50
    p_value = 0.9
    
    # Top-k decoding
    lm_topk = LanguageModel(mode='top-k', k=k_value)
    output_file2.write(f"=== Part 2 TOP-K DECODING RESULTS (k={k_value}) ===\n\n")
    for i, context in enumerate(contexts[:10]):
        result = lm_topk.generate(context, max_new_tokens=40)
        print(f"Example {i+1} (Top-k):")
        print(f"Context: {context}")
        print(f"Generated: {result}\n")
        
        output_file2.write(f"Example {i+1}:\n")
        output_file2.write(f"Context: {context}\n")
        output_file2.write(f"Generated: {result}\n\n")
    
    # Top-p decoding
    lm_topp = LanguageModel(mode='top-p', p=p_value)
    output_file2.write(f"\n\n=== Part 2 TOP-P DECODING RESULTS (p={p_value}) ===\n\n")
    for i, context in enumerate(contexts[:10]):
        result = lm_topp.generate(context, max_new_tokens=40)
        print(f"Example {i+1} (Top-p):")
        print(f"Context: {context}")
        print(f"Generated: {result}\n")
        
        output_file2.write(f"Example {i+1}:\n")
        output_file2.write(f"Context: {context}\n")
        output_file2.write(f"Generated: {result}\n\n")
    
    output_file2.close()
    print("\nPart 2 outputs saved to 'part2_output.txt'")


    #part 3
    temperatures_values = [0.5, 1.0, 2.0]  # Low, normal, high temperature
    k_values = [5, 20, 50]        # Small, medium, large k

    output_file3.write("=== PART 3: HYPERPARAMETER TUNING (Temperature + Top-k) ===\n\n")

    for temp in temperatures_values:
        for k in k_values:
            output_file3.write(f"=== Temperature: {temp}, k: {k} ===\n\n")
            print(f"Temperature: {temp}, k: {k}")
            lm = LanguageModel(mode = 'top-k', temperature=temp, k=k)
            for idx,context in enumerate(contexts[:5], 1):
                result = lm.generate(context, max_new_tokens=40)
                print(result)
                output_file3.write(f"Example {idx}:\n")
                output_file3.write(f"Context: {context}\n")
                output_file3.write(f"Generated: {result}\n\n")
        
            output_file3.write("-" * 80 + "\n\n")
                
    
    evaluation_features = [
        "relevance",      # Is it relevant to the context?
        "creativity",     # Is it creative and not repetitive?
        "grammar",        # Is it grammatically correct?
    ]
    
    output_file3.close()
    print("\nPart 3 outputs saved to 'part3_outputs.txt'")

