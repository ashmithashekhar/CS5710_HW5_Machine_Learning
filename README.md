# CS5710 Machine Learning - Homework 5

## Student Information
- **Name**: Ashmitha Kumbham
- **Course**: CS5710 Machine Learning

## Assignment Overview
This repository contains the solution for Homework 5 for the CS5710 Machine Learning course. The assignment covers both theoretical and practical aspects of transformer models, including understanding attention mechanisms, multi-head attention, and implementing coding tasks such as computing scaled dot-product attention and building a simple transformer encoder block using PyTorch.

## Assignment Parts
### Part A — Short Answer
This part includes theoretical questions on:
- Positional encoding concepts in transformer models.
- Attention mechanisms and multi-head attention.
- Ethical foundations of AI.
- Types of AI harms, dataset biases, and safety/security/privacy issues.

#### **Positional Encoding Concepts**
1. **Why do we need positional encodings in transformer models?**
   - Transformers process input tokens in parallel, which means they do not have an inherent sense of the order of tokens in a sequence. Positional encodings are introduced to provide the model with information about the relative or absolute position of tokens in a sequence, allowing the model to differentiate between tokens that occur at different positions.

2. **Describe two key requirements for a good positional encoding scheme.**
   - **Uniqueness**: Each position in the sequence should have a distinct encoding. This ensures that the model can differentiate between different positions.
   - **Smoothness**: The encoding for each position should be smoothly related to nearby positions. For example, the difference in encoding between positions 1 and 2 should be similar to the difference between positions 3 and 4.

3. **What does it mean for the positional encoding matrix to be unitary and norm-preserving?**
   - **Unitary**: The encoding matrix is unitary if the rows (or columns) are orthogonal to each other. This means that each positional encoding is distinct and independent, and they do not overlap or interfere with each other.
   - **Norm-preserving**: The length (norm) of each positional encoding vector remains constant across all positions. This helps maintain a consistent scale across all tokens, ensuring no positional encoding dominates due to its magnitude.

#### **Attention Mechanism**
1. **Define “attention score” and explain how it determines the weight of each token.**
   - The attention score measures how much focus (or "attention") the model should place on a particular token when processing another token. It is typically calculated by taking the dot product of the query vector and the key vector, with higher scores indicating a stronger relationship between tokens. The attention score is used to assign weights to the value vectors, which are then aggregated to form the context for the token.

2. **What mathematical operation is applied to convert alignment scores into attention weights?**
   - The alignment scores are converted into attention weights using the **softmax** function. This operation normalizes the scores so that they sum to 1, turning them into a probability distribution. The softmax ensures that tokens with higher alignment scores have higher attention weights.

3. **How is the context vector computed from these weights and values?**
   - The context vector is computed by taking a weighted sum of the value vectors. The attention weights determine how much influence each value vector has on the final context, with higher weights leading to a greater influence from the corresponding value vector.

#### **Multi-Head Attention**
1. **What is the main advantage of using multiple attention heads?**
   - The main advantage is that multiple attention heads allow the model to capture different types of relationships and dependencies between tokens. Each attention head learns to focus on different aspects of the input, enabling the model to capture more complex interactions in the data.

2. **How does splitting Q, K, and V across different subspaces improve model representation?**
   - By splitting the query (Q), key (K), and value (V) matrices across different subspaces, the model is able to focus on different parts of the sequence and learn diverse representations. Each subspace may focus on different aspects of the input, allowing the model to capture richer information.

3. **After multi-head attention, why is concatenation followed by another linear projection necessary?**
   - After multi-head attention, the outputs from all the attention heads are concatenated. This concatenation combines the information learned from each head. A subsequent linear projection is then applied to transform the concatenated output back into the desired dimensionality, which is the same as the input size, allowing the model to maintain consistency and continue processing.

#### **Ethical Foundations**
1. **Explain why ethics is not the same as laws or feelings.**
   - Ethics refers to a set of moral principles that guide human behavior, aiming to define what is right and wrong in various contexts. Laws are societal rules that govern behavior, while feelings are personal emotional responses to situations. Ethics helps guide decisions about what is morally justifiable, whereas laws and feelings may be influenced by societal norms or personal emotions.

2. **Briefly describe two classical ethical theories (e.g., utilitarianism and deontology) and how they would handle an AI decision scenario.**
   - **Utilitarianism**: This ethical theory focuses on maximizing overall happiness or well-being. In an AI decision scenario, it would prioritize outcomes that result in the greatest benefit for the greatest number, even if individual rights are compromised.
   - **Deontology**: This theory emphasizes duties and rules. In an AI context, it would focus on following ethical principles or rules, regardless of the outcomes. For example, an AI might be programmed to follow strict privacy guidelines even if breaking those guidelines could lead to a better overall outcome.

3. **Why do philosophers argue that no single ethical theory clearly “wins” in all contexts?**
   - Different ethical theories prioritize different values (e.g., outcomes vs. principles), and what is considered morally right can vary depending on the specific context. Philosophers argue that no single theory can account for all situations, as the "best" ethical approach often depends on the nature of the decision being made.

---

### **Part B — Coding**
This part includes two coding tasks:
1. **Compute Scaled Dot-Product Attention**: A Python function to compute scaled dot-product attention using NumPy, including softmax normalization.
2. **Implement Simple Transformer Encoder Block**: Implementing a simplified transformer encoder block in PyTorch, including multi-head self-attention, feed-forward network, and layer normalization.

## Setup and Installation

### Requirements:
- Python 3.x
- NumPy (for Q1)
- PyTorch (for Q2)

### Installation Steps:
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/[YourGitHubUsername]/CS5710_HW5_Machine_Learning.git
