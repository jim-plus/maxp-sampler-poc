# Max P Sampler Proof of Concept

Max P consists of a dynamic token filter which applies Winsorization to cap the probabilties of top tokens.
Specifically, a base probability in the range of [0,1] is used to cap individual token probability; the sampler then redistributes excess proportionally.
This proof of concept is implemented in PyTorch and transformers.

## Goals and Motivations
- Prevent overconfident predictions without breaking coherence
- Indirectly reduce long-range literal repetition by creating small variations that compound
- Nudge toward creativity without resorting to token bans or repetition penalties
- Complement min-p sampling in conjunction with temperature
- Be simple, principled mathematically, stateless, and composable
- Easy to implement and control (only one hyperparameter)

## Code

This proof of concept is implemented in PyTorch and designed to interact with the Hugging Face transformers pipeline.
```
pip install torch transformers
```

The core Max P logic is located in the repository's single executable Python file, which incorporates a working demonstration:
```
python maxp_sampler.py
```

## Mathematical Basis

Max P operates on the probability distribution $P$ of the next token, using a single hyperparameter, $\mathbf{p_{max}} \in [0, 1]$, as an absolute probability ceiling.
The process is simple, principled, and executed in three steps:
- Capping: For every token $i$, if its probability $p_i > \mathbf{p_{max}}$, its new probability is set to $p'_{i} = \mathbf{p_{max}}$.
- Excess Collection: The total probability mass $E$ that was clipped is calculated from all capped tokens: $$E = \sum_{i | p_i > p_{max}} (p_i - p_{max})$$
- Proportional Redistribution: The excess mass $E$ is then proportionally distributed across all tokens whose probabilities were not capped.

This ensures the resulting distribution remains normalized ($\sum p'_i = 1$) while dynamically suppressing overconfident predictions.
