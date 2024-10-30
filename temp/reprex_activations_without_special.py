from typing import Set

import torch
from sae_lens import SAE, ActivationsStore
from transformer_lens import HookedTransformer


def get_batch_without_special_token_activations(
    activations_store: ActivationsStore,
    special_tokens: Set[int | None],
    device: str,
) -> torch.Tensor:
    """
    Get a batch of activations from the ActivationsStore, removing special tokens.

    Args:
        activations_store: SAE Lens ActivationsStore instance
        special_tokens: Set of token IDs to remove (e.g., BOS, EOS, PAD tokens)
        device: Device to run computations on ('cpu', 'cuda', 'mps')

    Returns:
        torch.Tensor: Activations with special tokens removed, shape [batch_size, 1, d_in]
    """
    # Get a batch of tokens
    batch_tokens = activations_store.get_batch_tokens().to(device)

    # Get activations for these tokens
    with torch.no_grad():
        activations = activations_store.get_activations(batch_tokens).to(device)

    # Create mask for non-special tokens
    non_special_mask = ~torch.isin(
        batch_tokens, torch.tensor(list(special_tokens), device=device)
    )

    # Remove special token activations
    activations = activations[non_special_mask]

    # Reshape to match next_batch() output format
    activations = activations.reshape(-1, 1, activations.shape[-1])

    # Apply normalization if configured
    if activations_store.normalize_activations == "expected_average_only_in":
        activations = activations_store.apply_norm_scaling_factor(activations)

    # Get correct batch size and return
    train_batch_size = activations_store.train_batch_size_tokens
    return activations[:train_batch_size]


def main():
    # Initialize device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and SAE
    print("Loading model and SAE...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_18/width_16k/canonical",
        device=device,
    )

    # Set up activation store
    print("Setting up ActivationsStore...")
    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        store_batch_size_prompts=8,
        train_batch_size_tokens=512,
        n_batches_in_buffer=16,
        device=device,
    )

    # Get special tokens
    special_tokens = {
        model.tokenizer.bos_token_id,  # type: ignore
        model.tokenizer.eos_token_id,  # type: ignore
        model.tokenizer.pad_token_id,  # type: ignore
    }
    print(f"Special tokens: {special_tokens}")

    # Get batch with and without special tokens
    print("\nGetting regular batch...")
    regular_batch = activation_store.next_batch()
    print(f"Regular batch shape: {regular_batch.shape}")

    print("\nGetting batch without special tokens...")
    filtered_batch = get_batch_without_special_token_activations(
        activation_store, special_tokens, device
    )
    print(f"Filtered batch shape: {filtered_batch.shape}")

    # Compare a few statistics
    print("\nBatch Statistics:")
    print(f"Regular batch mean: {regular_batch.mean().item():.4f}")
    print(f"Filtered batch mean: {filtered_batch.mean().item():.4f}")
    print(f"Regular batch std: {regular_batch.std().item():.4f}")
    print(f"Filtered batch std: {filtered_batch.std().item():.4f}")


if __name__ == "__main__":
    main()
