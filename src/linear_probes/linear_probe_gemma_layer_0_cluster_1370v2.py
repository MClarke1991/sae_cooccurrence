import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sae_lens import SAE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoTokenizer

from sae_cooccurrence.utils.set_paths import get_git_root


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ActivationDataset(Dataset):
    def __init__(self, activations, labels, device):
        self.activations = activations.to(device)
        self.labels = labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


# def generate_examples(n_samples=1000) -> tuple[list[str], list[int]]:
#     """Generate example sentences with number words (one-ten) and without"""
    
#     templates = [
#         "I ate {} {} for lunch",
#         "There are {} {} on the shelf",
#         "We walked {} {} in the park",
#         "She bought {} {} at the store",
#         "They have {} {} at home",
#         "I waited {} {} for the bus",
#         "The garden has {} {} planted",
#         "He scored {} {} in the game",
#         "We saw {} {} in the tree",
#         "The class has {} {} enrolled",
#     ]

#     number_words = [
#         "one", "two", "three", "four", "five",
#         "six", "seven", "eight", "nine", "ten"
#     ]
    
#     objects = [
#         "cookies", "books", "miles", "shirts", "cats",
#         "minutes", "flowers", "goals", "birds", "students",
#         "apples", "boxes", "papers", "pictures", "tasks"
#     ]

#     examples = []
#     labels = []
    
#     for _ in range(n_samples):
#         template = np.random.choice(templates)
#         object_word = np.random.choice(objects)
        
#         # Randomly decide whether to use a number word or not
#         if np.random.random() < 0.5:  # 50% chance for each class
#             # Use number word
#             number = np.random.choice(number_words)
#             label = 1
#         else:
#             # Use either a digit or "some"
#             if np.random.random() < 0.7:  # 70% chance for digit within negative class
#                 number = str(np.random.randint(1, 11))
#             else:  # 30% chance for "some" within negative class
#                 number = np.random.choice(["some", "many", "several", "few"])
#             label = 0
            
#         example = template.format(number, object_word)
#         examples.append(example)
#         labels.append(label)
    
#     # Shuffle examples and labels together
#     indices = np.random.permutation(len(examples))
#     examples = [examples[i] for i in indices]
#     labels = [labels[i] for i in indices]
    
#     return examples, labels

@dataclass
class TextGenerationConfig:
    """Configuration for text generation parameters"""
    number_word_prob: float = 0.5  # Probability of using a number word
    digit_prob: float = 0.7  # Probability of using digit vs other words within negative class
    min_number: int = 1
    max_number: int = 11
    templates: list[str] | None = None
    number_words: list[str] | None = None
    objects: list[str] | None = None
    
    def __post_init__(self):
        # Default templates with more variety and complexity
        self.templates = self.templates or [
            "I ate {} {} for lunch today",
            "There are {} {} on the shelf now",
            "We walked {} {} in the beautiful park",
            "She bought {} {} at the local store",
            "They have {} {} at their home",
            "I waited {} {} for the express bus",
            "The garden has {} {} planted in rows",
            "He scored {} {} during the game",
            "We saw {} {} in the old tree",
            "The class has {} {} enrolled this term",
            "The recipe calls for {} {} to be added",
            "They collected {} {} during the drive",
            "I noticed {} {} in the corner",
            "The shelf holds {} {} neatly arranged",
            "We counted {} {} at the event"
        ]
        
        # Extended number words list
        self.number_words = self.number_words or [
            "one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine", "ten",
            # "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            # "twenty", "thirty", "forty", "fifty"
        ]
        
        # Extended objects list with more variety
        self.objects = self.objects or [
            "cookies", "books", "miles", "shirts", "cats",
            "minutes", "flowers", "goals", "birds", "students",
            "apples", "boxes", "papers", "pictures", "tasks",
            "pencils", "notebooks", "chairs", "plates", "bottles",
            "emails", "messages", "documents", "reports", "cards",
            "packages", "letters", "folders", "files", "tools"
        ]

def generate_balanced_examples(
    n_samples: int = 1000,
    config: TextGenerationConfig | None = None,
    seed: int | None = None
) -> tuple[list[str], list[int]]:
    """
    Generate balanced example sentences with number words and without.
    
    Args:
        n_samples: Number of examples to generate
        config: Configuration object for text generation parameters
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (examples, labels) where examples are sentences and labels are 1 for
        number words and 0 for digits or other quantifiers
    """
    if seed is not None:
        np.random.seed(seed)
        
    config = config or TextGenerationConfig()
    
    # Ensure balanced classes
    n_per_class = n_samples // 2
    examples = []
    labels = []
    
    # Helper function to generate a single example
    def generate_single_example(use_number_word: bool) -> tuple[str, int]:
        template = np.random.choice(config.templates)  # type: ignore
        object_word = np.random.choice(config.objects)  # type: ignore
        
        if use_number_word:
            number = np.random.choice(config.number_words)  # type: ignore
            label = 1
        else:
            if np.random.random() < config.digit_prob:
                number = str(np.random.randint(config.min_number, config.max_number))
            else:
                quantifiers = ["some", "many", "several", "few", "numerous", "various", 
                             "countless", "multiple", "abundant", "sparse"]
                number = np.random.choice(quantifiers)
            label = 0
            
        return template.format(number, object_word), label

    # Generate positive examples (with number words)
    for _ in tqdm(range(n_per_class), desc="Generating positive examples"):
        example, label = generate_single_example(use_number_word=True)
        examples.append(example)
        labels.append(label)
        
    # Generate negative examples (with digits or quantifiers)
    for _ in tqdm(range(n_per_class), desc="Generating negative examples"):
        example, label = generate_single_example(use_number_word=False)
        examples.append(example)
        labels.append(label)

    # Shuffle examples and labels together
    indices = np.random.permutation(len(examples))
    examples = [examples[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return examples, labels

def validate_examples(examples: list[str], labels: list[int]) -> bool:
    """
    Validate generated examples to ensure they meet requirements.
    
    Args:
        examples: List of generated sentences
        labels: List of corresponding labels
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    if len(examples) != len(labels):
        print("Error: Mismatch between examples and labels length")
        return False
        
    # Check class balance
    positive_count = sum(labels)
    if positive_count != len(labels) // 2:
        print(f"Error: Unbalanced classes. Positive examples: {positive_count}, "
              f"Expected: {len(labels) // 2}")
        return False
    
    return True


def get_layer_activations(
    model: HookedTransformer, tokenizer, texts: list[str], layer_idx=-1, is_hooked_transformer=False, device=None
) -> np.ndarray:
    """Get activations from a specific layer for a batch of texts"""
    activations = []
    device = device or get_device()

    with torch.no_grad():
        for text in tqdm(texts, desc="Getting activations"):
            if is_hooked_transformer:
                # HookedTransformer interface
                tokens = model.to_tokens(text, truncate=True)
                tokens = tokens.to(device)
                _, cache = model.run_with_cache(tokens)
                layer_output = cache["blocks." + str(layer_idx) + ".hook_resid_pre"]
            else:
                # Hugging Face model interface
                inputs = tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                layer_output = outputs.hidden_states[layer_idx]

            # Average over sequence length to get one vector per example
            mean_activation = layer_output.mean(dim=1)
            activations.append(mean_activation.cpu().numpy().squeeze())

    return np.stack(activations)


class LinearProbe(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)  # Remove sigmoid from forward pass


def evaluate_probe(probe: nn.Module, 
                   test_loader: DataLoader):
    """Evaluate probe performance with multiple metrics"""
    probe.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_activations, batch_labels in test_loader:
            outputs = probe(batch_activations)
            predicted = (outputs >= 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

def plot_metrics(metrics_history: dict, loss_history: dict, out_dir: str) -> None:
    """Plot training metrics and loss over epochs"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot metrics for each split
    for split in metrics_history:
        for metric in metrics_history[split][0].keys():
            values = [m[metric] for m in metrics_history[split]]
            ax1.plot(range(1, len(values) + 1), values, 
                    label=f"{split}_{metric}", marker='o')
    
    ax1.set_title('Probe Performance Metrics Over Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True)
    
    # Plot losses
    for split, losses in loss_history.items():
        ax2.plot(range(1, len(losses) + 1), losses, 
                label=f'{split}_loss', marker='o')
    
    ax2.set_title('Training and Validation Loss Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"metrics_n_examples_{n_examples}.png"))
    plt.close()


def save_metrics(metrics_history: dict, loss_history: dict, out_dir: str) -> None:
    """Save training metrics and loss to a text file"""
    metrics_file = os.path.join(out_dir, f"metrics_n_examples_{n_examples}.txt")
    
    with open(metrics_file, "w") as f:
        headers = ["Epoch", "Split", "Loss", "Accuracy", "Precision", "Recall", "F1"]
        f.write("\t".join(headers) + "\n")
        
        n_epochs = len(metrics_history['train'])
        for epoch in range(n_epochs):
            for split in metrics_history:
                metrics = metrics_history[split][epoch]
                loss = loss_history.get(split, [0] * n_epochs)[epoch]
                f.write(f"{epoch+1}\t{split}\t{loss:.4f}\t"
                       f"{metrics['accuracy']:.4f}\t"
                       f"{metrics['precision']:.4f}\t"
                       f"{metrics['recall']:.4f}\t"
                       f"{metrics['f1']:.4f}\n")


def train_probe(
    model_name,
    out_dir,
    config: TextGenerationConfig,
    layer_idx=0,
    n_samples=1000,
    batch_size=32,
    n_epochs=10,
    learning_rate=0.001,
    weight_decay=0.01,
):
    """Train a linear probe to detect patterns"""
    device = get_device()

    # Generate example sentences
    print(f"Generating {n_samples} examples...")
    examples, labels = generate_balanced_examples(
        n_samples,
        config=config,
        seed=42  # for reproducibility
    )

    # Load model and tokenizer
    is_hooked_transformer = False
    try:
        model = HookedTransformer.from_pretrained(model_name, device=device)
        tokenizer = model.tokenizer
        is_hooked_transformer = True
    except Exception as e:
        print(f"Failed to load with HookedTransformer: {e}")
        print("Falling back to AutoModel")
        model = AutoModel.from_pretrained(model_name, device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get activations
    activations = get_layer_activations(
        model,
        tokenizer,
        examples,
        layer_idx,
        is_hooked_transformer=is_hooked_transformer,
        device=device,
    )

    # Update the data split to include validation
    X_train, X_temp, y_train, y_temp = train_test_split(
        activations, labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Create datasets and dataloaders for all splits
    train_dataset = ActivationDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1), device
    )
    val_dataset = ActivationDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1), device
    )
    test_dataset = ActivationDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test).reshape(-1, 1), device
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize probe, criterion and optimizer
    probe = LinearProbe(activations.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Add validation metrics tracking
    metrics_history = {'train': [], 'val': [], 'test': []}
    loss_history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    best_probe_state = None
    patience = 5
    patience_counter = 0

    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Training"):
        # Training phase
        probe.train()
        total_train_loss = 0

        for batch_activations, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = probe(batch_activations)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        loss_history['train'].append(avg_train_loss)

        # Validation phase
        probe.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_activations, batch_labels in val_loader:
                logits = probe(batch_activations)
                loss = criterion(logits, batch_labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        loss_history['val'].append(avg_val_loss)

        # Evaluation on all sets
        train_metrics = evaluate_probe(probe, train_loader)
        val_metrics = evaluate_probe(probe, val_loader)
        test_metrics = evaluate_probe(probe, test_loader)
        
        metrics_history['train'].append(train_metrics)
        metrics_history['val'].append(val_metrics)
        metrics_history['test'].append(test_metrics)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_probe_state = probe.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        print(
            f"Epoch {epoch+1}/{n_epochs}\n"
            f"Train - Loss: {avg_train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}\n"
            f"Val   - Loss: {avg_val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}\n"
            f"Test  - Acc: {test_metrics['accuracy']:.4f}"
        )

    # Load best model
    if best_probe_state is not None:
        probe.load_state_dict(best_probe_state)

    # Update plotting and saving functions to handle the new metrics structure
    plot_metrics(metrics_history, loss_history, out_dir)
    save_metrics(metrics_history, loss_history, out_dir)

    return probe, model, tokenizer


def analyze_neurons(probe, n_top=10):
    """Analyze which neurons have the highest weights in the probe"""
    weights = probe.linear.weight.detach().cpu().numpy().squeeze()
    top_neurons = np.argsort(np.abs(weights))[-n_top:]

    # Save top neurons to CSV
    neuron_data = {
        "Neuron Index": top_neurons,
        "Weight": weights[top_neurons]
    }
    pd.DataFrame(neuron_data).to_csv(os.path.join(out_dir, f"top_neurons_n_examples_{n_examples}.csv"), index=False)

    # Plot top neuron weights
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_top), weights[top_neurons])
    plt.title("Top Neuron Weights for 'one of' Detection")
    plt.xlabel("Neuron Index")
    plt.ylabel("Weight")
    plt.xticks(range(n_top), [str(n) for n in top_neurons])  # type: ignore
    plt.show()

    return top_neurons, weights[top_neurons]


def load_sae(
    sae_release="gemma-scope-2b-pt-res-canonical", sae_id="layer_0/width_16k/canonical"
):
    # device = get_device()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    sae.W_dec.norm(dim=-1).mean()
    sae.fold_W_dec_norm()

    return sae


def find_similar_sae_features(probe: nn.Module, sae: SAE, out_dir: str, top_k=10):
    """Find SAE features most similar to the linear probe weights using cosine similarity"""
    # Get probe weights and normalize them
    probe_weights = probe.linear.weight.detach().squeeze()
    probe_weights_norm = probe_weights / torch.norm(probe_weights)

    # Get SAE decoder weights and normalize them
    sae_weights = sae.W_dec.squeeze()
    sae_weights_norm = sae_weights / torch.norm(sae_weights, dim=1, keepdim=True)

    # Compute cosine similarities
    similarities = torch.matmul(sae_weights_norm, probe_weights_norm)

    # Get top-k most similar features
    top_similarities, top_indices = torch.topk(similarities, k=top_k)

    # Save top features to CSV
    feature_data = {
        "Feature Index": top_indices.cpu().numpy(),
        "Cosine Similarity": top_similarities.detach().cpu().numpy()
    }
    pd.DataFrame(feature_data).to_csv(os.path.join(out_dir, f"top_similar_sae_features_n_examples_{n_examples}.csv"), index=False)

    # Plot the top similar features
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_k), top_similarities.detach().cpu().numpy(), tick_label=top_indices.cpu().numpy())
    plt.title("Top Similar SAE Features to Linear Probe Weights")
    plt.xlabel("SAE Feature Index")
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"top_similar_sae_features_n_examples_{n_examples}.png"))
    plt.close()

    return top_indices.tolist(), top_similarities.tolist()


# Example usage
if __name__ == "__main__":
    
    model_name = "gemma-2-2b"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = "layer_0/width_16k/canonical"
    sae_id_safe = sae_id.replace("/", "_").replace(".", "_")
    layer_idx = 0
    n_examples = 100
    
    config = TextGenerationConfig(
        number_word_prob=0.5,
        digit_prob=0.7,
        min_number=1,
        max_number=11,
        # Optionally override templates, number_words, or objects
    )
    
    out_dir = os.path.join(get_git_root(), 
                           "results", 
                           "linear_probes", 
                           model_name, 
                           sae_release, 
                           sae_id_safe, 
                           f"n_examples_{n_examples}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Train the probe
    probe, model, tokenizer = train_probe(model_name=model_name, 
                                          layer_idx=layer_idx, 
                                          out_dir=out_dir, 
                                          n_samples=n_examples,
                                          config=config,
                                          weight_decay=0.01)
    # probe, model, tokenizer = train_probe(model_name="gpt2-small", layer_idx=0)

    # Load SAE
    sae = load_sae(
        sae_release=sae_release,
        sae_id=sae_id,
    )
    # sae = load_sae(sae_release="gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre")

    # Analyze the most important neurons
    top_neurons, top_weights = analyze_neurons(probe)

    # Find similar SAE features
    top_features, similarities = find_similar_sae_features(probe, sae, out_dir)

    # Print results
    print("\nTop neurons for 'one of' detection:")
    for neuron, weight in zip(top_neurons, top_weights):
        print(f"Neuron {neuron}: weight = {weight:.4f}")

    print("\nMost similar SAE features:")
    for feature, similarity in zip(top_features, similarities):
        print(f"Feature {feature}: similarity = {similarity:.4f}")
