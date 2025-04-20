# tree_attention.py
import torch
from typing import List, Optional, Tuple

# Define TOPK as used in the original utils
TOPK = 10

# Helper function from utils.py
def pad_path(path, length, pad_value=-2):
    return path + [pad_value] * (length - len(path))

# _make_causal_mask remains the same
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

# --- Combined Buffer Generation ---
def generate_medusa_buffers(
    medusa_choices: List[Tuple[int, ...]],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Generates buffers for the Medusa structure based on the provided choices.

    Args:
        medusa_choices (List[Tuple[int,...]]): A list of tuples defining the paths
                                              in the Medusa tree.
        dtype (torch.dtype): Data type for attention mask. Others use torch.long.
        device (torch.device): Device for the tensors.

    Returns:
        dict: A dictionary containing:
              - medusa_attn_mask: Attention mask for tree attention (additive, -inf blocks).
              - tree_indices: Maps nodes in the flattened tree to indices in the
                              candidate token list.
              - medusa_position_ids: Relative position IDs for nodes in the tree.
              - retrieve_indices: Indices to reconstruct full paths for evaluation.
    """
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Create depth counts for easier iteration
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth > len(depth_counts):
            depth_counts.extend([0] * (depth - len(depth_counts)))
        depth_counts[depth - 1] += 1
        prev_depth = depth
    num_medusa_heads = len(depth_counts) # Directly get number of heads from max depth

    # 1. Medusa Attention Mask (Additive version: 0 allows, -inf blocks)
    medusa_attn_mask_logical = torch.eye(medusa_len, medusa_len, dtype=torch.bool, device=device)
    medusa_attn_mask_logical[:, 0] = True # All attend to root
    choice_to_idx = {choice: i + 1 for i, choice in enumerate(sorted_medusa_choices)}
    for i, current_choice in enumerate(sorted_medusa_choices):
        current_idx = i + 1
        for k in range(1, len(current_choice)):
            ancestor_choice = current_choice[:k]
            if ancestor_choice in choice_to_idx:
                ancestor_idx = choice_to_idx[ancestor_choice]
                medusa_attn_mask_logical[current_idx, ancestor_idx] = True
    # Convert to additive float mask
    medusa_attn_mask = torch.log(medusa_attn_mask_logical.to(dtype))
    medusa_attn_mask = medusa_attn_mask.unsqueeze(0).unsqueeze(0) # Add batch and head dims

    # 2. Tree Indices
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long, device=device)
    medusa_tree_indices[0] = 0 # Root maps to 0
    start = 0
    for head_idx, count in enumerate(depth_counts): # head_idx is 0-based head index (depth-1)
        for j in range(count):
            # Get the actual choice tuple for the current node
            current_choice = sorted_medusa_choices[start + j]
            # Calculate index in the flattened candidate list
            # path[-1] is rank, head_idx is the head, +1 for base token
            tree_idx_val = current_choice[-1] + TOPK * head_idx + 1
            medusa_tree_indices[start + j + 1] = tree_idx_val
        start += count

    # 3. Medusa Position IDs
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long, device=device)
    start = 0
    for depth_minus_1, count in enumerate(depth_counts):
        # Nodes at depth d (head_idx d-1) get position ID d
        medusa_position_ids[start + 1: start + count + 1] = depth_minus_1 + 1
        start += count

    # 4. Retrieve Indices
    retrieve_indices_nest = []
    retrieve_paths = set() # Use set for faster lookups
    # Iterate backwards to ensure longest paths are processed first
    for i in range(len(sorted_medusa_choices) - 1, -1, -1):
        cur_medusa_choice = sorted_medusa_choices[i]
        if cur_medusa_choice in retrieve_paths:
            continue

        retrieve_indice_row = []
        # Add indices for all prefixes of this path
        for c in range(len(cur_medusa_choice)):
            prefix_choice = cur_medusa_choice[:c+1]
            if prefix_choice in choice_to_idx: # Should always be true if logic is sound
                 # Get the index in the flattened sequence (node index)
                retrieve_indice_row.append(choice_to_idx[prefix_choice])
                retrieve_paths.add(prefix_choice) # Mark as added
            else:
                 # This case should ideally not happen with sorted choices
                 print(f"Warning: Prefix {prefix_choice} not found in choice_to_idx.")

        retrieve_indices_nest.append(retrieve_indice_row)

    # Pad paths to the maximum length found
    if not retrieve_indices_nest: # Handle empty case
         max_length = 0
         retrieve_indices = torch.empty((0, 1), dtype=torch.long, device=device)
    else:
        max_length = max(len(x) for x in retrieve_indices_nest) if retrieve_indices_nest else 0
        # Pad with a value that won't cause issues (e.g., -1, though 0 is used later)
        # The padding value itself doesn't matter much as it's mainly for tensor creation
        retrieve_indices_padded = [pad_path(path, max_length, pad_value=-1) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices_padded, dtype=torch.long, device=device)

    # Prepend a column of zeros for the root/initial state
    retrieve_indices = torch.cat(
        [torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long, device=device), retrieve_indices],
        dim=1
    )

    # Final dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask,
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        "sorted_medusa_choices": sorted_medusa_choices # Also useful to return
    }
    return medusa_buffers


# apply_medusa_tree_attention remains the same as the last working version
def apply_medusa_tree_attention(
    base_attention_mask: torch.Tensor,
    medusa_attn_mask_additive: torch.Tensor, # Expect additive mask now
) -> torch.Tensor:
    """Applies Medusa tree constraints to a base causal attention mask."""
    medusa_len = medusa_attn_mask_additive.shape[-1]
    if medusa_len == 0:
        return base_attention_mask

    bsz, _, tgt_len, src_len = base_attention_mask.shape
    assert tgt_len == medusa_len, f"Target length ({tgt_len}) != Medusa length ({medusa_len})"
    assert src_len >= medusa_len, f"Source length ({src_len}) < Medusa length ({medusa_len})"

    combined_mask = base_attention_mask.clone()

    # Select the Medusa-to-Medusa submatrix from the base mask
    base_medusa_submatrix = combined_mask[:, :, -medusa_len:, -medusa_len:]

    # Combine using minimum (preserves -inf from either mask)
    combined_submatrix = torch.minimum(base_medusa_submatrix, medusa_attn_mask_additive)

    # Place the combined submatrix back into the full mask
    combined_mask[:, :, -medusa_len:, -medusa_len:] = combined_submatrix

    return combined_mask


# --- Runnable Example ---
if __name__ == "__main__":
    medusa_choices = [(0,), (1,), (0,0), (0,1), (1,0), (1,1), (0,0,0)]

    batch_size = 1
    prefix_len = 5
    dtype = torch.float32
    device = torch.device("cpu")

    num_medusa_candidates = len(medusa_choices)
    medusa_len = 1 + num_medusa_candidates

    print(f"--- Simulation Parameters ---")
    print(f"Batch Size: {batch_size}")
    print(f"Prefix Length: {prefix_len}")
    print(f"Medusa Choices: {medusa_choices}")
    print(f"TOPK: {TOPK}")
    print(f"Total Medusa Nodes (Root + Candidates): {medusa_len}")
    print(f"Total Sequence Length (Prefix + Medusa Nodes): {prefix_len + medusa_len}")
    print("-" * 30)

    # 1. Generate ALL Medusa buffers
    medusa_buffers = generate_medusa_buffers(
        medusa_choices, dtype=dtype, device=device
    )
    medusa_attn_mask_additive = medusa_buffers["medusa_attn_mask"]
    tree_indices = medusa_buffers["tree_indices"]
    medusa_position_ids = medusa_buffers["medusa_position_ids"]
    retrieve_indices = medusa_buffers["retrieve_indices"]
    sorted_choices = medusa_buffers["sorted_medusa_choices"] # Get sorted choices

    print(f"--- Medusa Tree Attention Mask (Additive: 0 allows, -inf blocks) ---")
    print(f"Shape: {medusa_attn_mask_additive.shape}")
    # Print mask in a readable format (0 or -1 for -inf)
    print(torch.where(torch.isneginf(medusa_attn_mask_additive), -1, 0).squeeze())
    print("-" * 30)

    print(f"--- Tree Indices ---")
    print(f"Shape: {tree_indices.shape}")
    print(tree_indices)
    print("-" * 30)

    print(f"--- Medusa Position IDs ---")
    print(f"Shape: {medusa_position_ids.shape}")
    print(medusa_position_ids)
    print("-" * 30)

    print(f"--- Retrieve Indices ---")
    print(f"Shape: {retrieve_indices.shape}")
    print(retrieve_indices)
    print("-" * 30)

    # 2. Generate the base causal attention mask
    base_causal_mask = _make_causal_mask(
        input_ids_shape=(batch_size, medusa_len),
        dtype=dtype,
        device=device,
        past_key_values_length=prefix_len,
    )
    # print(f"--- Base Causal Attention Mask (0 allows, -1 for -inf blocks) ---")
    # print(f"Shape: {base_causal_mask.shape}")
    # print(torch.where(torch.isneginf(base_causal_mask), -1, 0).squeeze())
    # print("-" * 30)

    # 3. Apply the Medusa tree constraints to the base causal mask
    final_attention_mask = apply_medusa_tree_attention(
        base_causal_mask,
        medusa_attn_mask_additive # Use the additive mask directly
    )
    print(f"--- Final Combined Attention Mask (0 allows, -1 for -inf blocks) ---")
    print(f"Shape: {final_attention_mask.shape}")
    print(torch.where(torch.isneginf(final_attention_mask), -1, 0).squeeze())
    print("-" * 30)

    # Verification (using sorted_choices from buffer output)
    target_choice = (0,0)
    if target_choice in sorted_choices:
        choice_idx_in_list = sorted_choices.index(target_choice)
        node_idx = choice_idx_in_list + 1 # +1 for root

        print(f"Verification for node index {node_idx} (corresponds to choice {target_choice}):")
        final_mask_row_int = torch.where(torch.isneginf(final_attention_mask[0, 0, node_idx, :]), -1, 0)
        print(f"Final Mask Row (Prefix + Medusa Nodes):\n{final_mask_row_int}")

        # Manually determine expected allowed indices for the medusa block part
        # node_idx = 3, choice=(0,0). Ancestor choice is (0,) at index 1. Root is 0.
        expected_medusa_allowed_indices = {0, 1, 3} # Root, Ancestor (0,), Self (0,0)
        expected_medusa_block = [-1] * medusa_len
        for idx in range(medusa_len):
            # Must be causally allowed (idx <= node_idx) AND allowed by tree
            if idx <= node_idx and idx in expected_medusa_allowed_indices:
                expected_medusa_block[idx] = 0
            else:
                expected_medusa_block[idx] = -1

        print(f"\nExpected medusa block: {expected_medusa_block}")

        actual_medusa_block = final_mask_row_int[prefix_len:].tolist()
        print(f"Actual medusa block:   {actual_medusa_block}")

        if actual_medusa_block == expected_medusa_block:
             print("Verification PASSED for node (0,0)")
        else:
             print("Verification FAILED for node (0,0)")
    else:
        print(f"Choice {target_choice} not found in medusa_choices for verification.")