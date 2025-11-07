import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from typing import Optional

# proof of concept implementation

class MaxPLogitsWarper(LogitsProcessor):
    """
    Max-P sampler: Caps maximum probability (Winsorization) and redistributes excess proportionally.
    
    Args:
        max_p (float): Maximum allowed probability for any token (0 < max_p < 1)
    """
    def __init__(self, max_p: float):
        if not 0 < max_p < 1:
            raise ValueError(f"max_p must be between 0 and 1, got {max_p}")
        self.max_p = max_p
        self.call_count = 0
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.call_count += 1
        
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)
                
        # Find tokens exceeding cap
        mask = probs > self.max_p
        num_exceeding = mask.sum().item()
                
        # If no tokens exceed cap, return original scores
        if not mask.any():
            return scores
        
        # Calculate new probability distribution
        excess = torch.clamp(probs - self.max_p, min=0.0)
        total_excess = excess.sum(dim=-1, keepdim=True)
        
        # Get uncapped token probabilities
        uncapped_mask = ~mask
        uncapped_probs = probs * uncapped_mask.float()
        uncapped_sum = uncapped_probs.sum(dim=-1, keepdim=True)
                
        # Avoid division by zero - if all tokens are capped, uniform distribution among them
        if (uncapped_sum < 1e-10).any():
            num_tokens = probs.shape[-1]
            final_probs = torch.full_like(probs, 1.0 / num_tokens)
        else:
            # Redistribute to uncapped tokens proportionally
            scale_factor = (uncapped_sum + total_excess) / uncapped_sum
                        
            final_probs = torch.where(
                uncapped_mask,
                uncapped_probs * scale_factor,
                torch.tensor(self.max_p, device=probs.device, dtype=probs.dtype)
            )
                
        # Normalize and ensure valid probabilities
        final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)
        final_probs = torch.clamp(final_probs, min=1e-10, max=1.0)
        
        # Convert to logits
        final_logits = torch.log(final_probs)
        
        return final_logits

def generate_custom(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    max_p: Optional[float] = None,
    min_p: Optional[float] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Generate text with custom sampling parameters.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (applied first)
        max_p: Maximum probability cap (applied after temperature)
        min_p: Minimum probability threshold relative to top token
        device: Device to use
    """
    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Build logits processor list
    logits_processor = LogitsProcessorList()
    
    # Add max_p warper if specified
    if max_p is not None:
        logits_processor.append(MaxPLogitsWarper(max_p=max_p))
        
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        logits_processor=logits_processor if len(logits_processor) > 0 else None,
        min_p=min_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Example usage
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name,
        device_map="cuda",
        low_cpu_mem_usage=True)
    model.tie_weights()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    token_spew = 128

    # Test prompt
    prompts = ["Once upon a time, in a cyberpunk dystopia,"]
    
    print("Testing different sampling configurations:")
    
    # Configuration: Standard sampling
    print("\nStandard (temp=1.0, min_p=0.1):")
    output = generate_custom(
        model, tokenizer, prompts,
        max_new_tokens=token_spew,
        temperature=1.0,
        max_p=None,
        min_p=0.10,
        device=device,
    )
    print(output)
       
    # Configuration: Full stack (Temperature + Max-P + Min-P)
    print("\nHot Creative (temp=1.2, max_p=0.9, min_p=0.05):")
    output = generate_custom(
        model, tokenizer, prompts,
        max_new_tokens=token_spew,
        temperature=1.2,
        max_p=0.90,
        min_p=0.05,
        device=device,
    )
    print(output)
