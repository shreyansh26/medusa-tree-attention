# tree_attention.py
import torch
from typing import List, Optional, Tuple

# _make_causal_mask remains the same as the previous working version
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """Make a causal mask for self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device, dtype=torch.bool))
    mask = torch.log(mask.to(dtype)) # log(1)=0, log(0)=-inf

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
            dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, past_key_values_length + tgt_len)

# generate_medusa_tree_attn_mask remains the same
def generate_medusa_tree_attn_mask(
    medusa_choices: List[Tuple[int, ...]],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates the attention mask specific to the Medusa tree structure."""
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    print("sorted_medusa_choices: ", sorted_medusa_choices)
    medusa_len = len(sorted_medusa_choices) + 1
    medusa_attn_mask = torch.eye(medusa_len, medusa_len, dtype=dtype, device=device)
    medusa_attn_mask[:, 0] = 1.0
    choice_to_idx = {choice: i + 1 for i, choice in enumerate(sorted_medusa_choices)}
    for i, current_choice in enumerate(sorted_medusa_choices):
        current_idx = i + 1
        for k in range(1, len(current_choice)):
            ancestor_choice = current_choice[:k]
            if ancestor_choice in choice_to_idx:
                ancestor_idx = choice_to_idx[ancestor_choice]
                medusa_attn_mask[current_idx, ancestor_idx] = 1.0
                print(current_choice, ancestor_choice, ancestor_idx, current_idx)
    return medusa_attn_mask.unsqueeze(0).unsqueeze(0)

# --- apply_medusa_tree_attention: FINAL CORRECTED IMPLEMENTATION ---
def apply_medusa_tree_attention(
    base_attention_mask: torch.Tensor,
    medusa_tree_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Applies the Medusa tree structure constraints to a base causal attention mask.
    FINAL CORRECTED IMPLEMENTATION.

    Args:
        base_attention_mask (torch.Tensor): The original attention mask, usually causal.
                                            Shape: (bsz, 1, seq_len, full_seq_len).
                                            Values are 0.0 (allowed) or -inf (masked).
        medusa_tree_mask (torch.Tensor): The mask defining the Medusa tree structure.
                                         Shape: (1, 1, medusa_len, medusa_len).
                                         Values are 1.0 (allowed) or 0.0 (masked by tree).

    Returns:
        torch.Tensor: The combined attention mask incorporating Medusa tree constraints.
                      Shape is the same as base_attention_mask.
                      Values are 0.0 (allowed) or -inf (masked).
    """
    medusa_len = medusa_tree_mask.shape[-1]
    if medusa_len == 0:
        return base_attention_mask

    bsz, _, tgt_len, src_len = base_attention_mask.shape
    assert tgt_len == medusa_len, f"Target length ({tgt_len}) != Medusa length ({medusa_len})"
    assert src_len >= medusa_len, f"Source length ({src_len}) < Medusa length ({medusa_len})"

    combined_mask = base_attention_mask.clone()
    min_val = torch.finfo(combined_mask.dtype).min

    # 1. Convert Medusa Tree Mask to additive format (0.0 allows, -inf blocks)
    # Shape: (1, 1, medusa_len, medusa_len)
    medusa_tree_additive_mask = torch.log(medusa_tree_mask) # log(1)=0, log(0)=-inf

    # 2. Select the Medusa-to-Medusa submatrix from the base mask
    # Shape: (bsz, 1, medusa_len, medusa_len)
    base_medusa_submatrix = combined_mask[:, :, -medusa_len:, -medusa_len:]

    # 3. Combine the base causal mask's submatrix with the tree additive mask.
    #    Taking the minimum ensures that if *either* mask is -inf, the result is -inf.
    #    We need broadcasting for the batch dimension.
    # Shape: (bsz, 1, medusa_len, medusa_len)
    combined_submatrix = base_medusa_submatrix + medusa_tree_additive_mask
    # combined_submatrix = torch.minimum(base_medusa_submatrix, medusa_tree_additive_mask)

    # 4. Place the combined submatrix back into the full mask
    combined_mask[:, :, -medusa_len:, -medusa_len:] = combined_submatrix

    return combined_mask

TOPK = 10

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        }
    
    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers

# --- Runnable Example ---
if __name__ == "__main__":
    # Define the Medusa tree structure (e.g., from medusa_choices.py)
    medusa_choices = [(0,), (1,), (0,0), (0,1), (1,0), (1,1), (0,0,0)] # Tuples are hashable

    # Simulation parameters
    batch_size = 1
    prefix_len = 5 # Shorter prefix for easier viewing
    dtype = torch.float32
    device = torch.device("cpu") # or "cuda" if available

    # Calculate Medusa length
    num_medusa_candidates = len(medusa_choices)
    medusa_len = 1 + num_medusa_candidates

    print(f"--- Simulation Parameters ---")
    print(f"Batch Size: {batch_size}")
    print(f"Prefix Length: {prefix_len}")
    print(f"Medusa Choices: {medusa_choices}")
    print(f"Total Medusa Nodes (Root + Candidates): {medusa_len}")
    print(f"Total Sequence Length (Prefix + Medusa Nodes): {prefix_len + medusa_len}")
    print("-" * 30)

    # 1. Generate the Medusa tree attention mask (1 allows, 0 blocks within the tree)
    medusa_tree_mask = generate_medusa_tree_attn_mask(medusa_choices, dtype=dtype, device=device)
    print(f"--- Medusa Tree Attention Mask (1 allows, 0 blocks) ---")
    print(f"Shape: {medusa_tree_mask.shape}")
    print(medusa_tree_mask.squeeze().int())
    print("-" * 30)

    # 2. Generate the base causal attention mask
    base_causal_mask = _make_causal_mask(
        input_ids_shape=(batch_size, medusa_len),
        dtype=dtype,
        device=device,
        past_key_values_length=prefix_len,
    )
    print(f"--- Base Causal Attention Mask (0 allows, -1 for -inf blocks) ---")
    print(f"Shape: {base_causal_mask.shape}")
    print(torch.where(torch.isneginf(base_causal_mask), -1, 0).squeeze())
    print("-" * 30)

    # 3. Apply the Medusa tree constraints to the base causal mask
    final_attention_mask = apply_medusa_tree_attention(
        base_causal_mask,
        medusa_tree_mask # Float mask (0 blocks, 1 allows)
    )
    print(f"--- Final Combined Attention Mask (0 allows, -1 for -inf blocks) ---")
    print(f"Shape: {final_attention_mask.shape}")
    # Use float formatting for clarity, showing -inf for masked
    print(final_attention_mask.squeeze().to(torch.float16)) # Use float16 for concise print
    print("-" * 30)

    # Verification
    if (0, 0) in medusa_choices:
        sorted_choices_list = sorted(medusa_choices, key=lambda x: (len(x), x))
        choice_idx_in_list = sorted_choices_list.index((0, 0))
        node_idx = choice_idx_in_list + 1
        print(f"Verification for node index {node_idx} (corresponds to choice (0,0)):")
        final_mask_row_int = torch.where(torch.isneginf(final_attention_mask[0, 0, node_idx, :]), -1, 0)
        print(f"Final Mask Row (Prefix + Medusa Nodes):\n{final_mask_row_int}")

        # Recalculate expected based on combined logic
        expected_medusa_block = [0, 0, -1, 0, -1, -1, -1, -1] # Causal + Tree blocked indices
        print(f"\nExpected medusa block: {expected_medusa_block}")

        actual_medusa_block = final_mask_row_int[prefix_len:].tolist()
        print(f"Actual medusa block:   {actual_medusa_block}")

        if actual_medusa_block == expected_medusa_block:
             print("Verification PASSED for node (0,0)")
        else:
             print("Verification FAILED for node (0,0)")

        print(generate_medusa_buffers(medusa_choices, device=device))