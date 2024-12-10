import os

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


def generate_examples(n_samples=1000) -> tuple[list[str], list[int]]:
    """Generate example sentences with number words (one-ten) and without"""
    
    templates = [
        "I ate {} {} for lunch",
        "There are {} {} on the shelf",
        "We walked {} {} in the park",
        "She bought {} {} at the store",
        "They have {} {} at home",
        "I waited {} {} for the bus",
        "The garden has {} {} planted",
        "He scored {} {} in the game",
        "We saw {} {} in the tree",
        "The class has {} {} enrolled",
    ]

    number_words = [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten"
    ]
    
    objects = [
        "cookies", "books", "miles", "shirts", "cats",
        "minutes", "flowers", "goals", "birds", "students",
        "apples", "boxes", "papers", "pictures", "tasks"
    ]

    examples = []
    labels = []
    
    for _ in range(n_samples):
        template = np.random.choice(templates)
        object_word = np.random.choice(objects)
        
        # Randomly decide whether to use a number word or not
        if np.random.random() < 0.5:  # 50% chance for each class
            # Use number word
            number = np.random.choice(number_words)
            label = 1
        else:
            # Use either a digit or "some"
            if np.random.random() < 0.7:  # 70% chance for digit within negative class
                number = str(np.random.randint(1, 11))
            else:  # 30% chance for "some" within negative class
                number = np.random.choice(["some", "many", "several", "few"])
            label = 0
            
        example = template.format(number, object_word)
        examples.append(example)
        labels.append(label)
    
    # Shuffle examples and labels together
    indices = np.random.permutation(len(examples))
    examples = [examples[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return examples, labels


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

def plot_metrics(metrics_history: list[dict[str, float]], loss_history: list[float], out_dir: str) -> None:
    """Plot training metrics and loss over epochs"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot metrics
    for metric in metrics_history[0].keys():
        values = [m[metric] for m in metrics_history]
        ax1.plot(range(1, len(values) + 1), values, label=metric, marker='o')
    
    ax1.set_title('Probe Performance Metrics Over Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(range(1, len(loss_history) + 1), loss_history, label='Loss', marker='o', color='red')
    ax2.set_title('Training Loss Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics.png"))
    plt.close()


def save_metrics(metrics_history: list[dict[str, float]], 
                loss_history: list[float], 
                out_dir: str) -> None:
    """Save training metrics and loss to a text file"""
    metrics_file = os.path.join(out_dir, "metrics.txt")
    
    with open(metrics_file, "w") as f:
        f.write("Epoch\tLoss\tAccuracy\tPrecision\tRecall\tF1\n")
        for epoch, (metrics, loss) in enumerate(zip(metrics_history, loss_history), 1):
            f.write(f"{epoch}\t{loss:.4f}\t{metrics['accuracy']:.4f}\t"
                   f"{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t"
                   f"{metrics['f1']:.4f}\n")


def train_probe(
    model_name,
    out_dir,
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
    examples, labels = generate_examples(n_samples)

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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = ActivationDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1), device
    )
    test_dataset = ActivationDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test).reshape(-1, 1), device
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize probe, criterion and optimizer
    probe = LinearProbe(activations.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Add metrics and loss history tracking
    metrics_history = []
    loss_history = []

    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Training"):
        probe.train()
        total_loss = 0

        for batch_activations, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = probe(batch_activations)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Evaluation
        metrics = evaluate_probe(probe, test_loader)
        metrics_history.append(metrics)
        
        print(
            f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
        )

    # Plot metrics and save to file after training
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
    pd.DataFrame(neuron_data).to_csv("top_neurons.csv", index=False)

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


def find_similar_sae_features(probe, sae, top_k=10):
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
    pd.DataFrame(feature_data).to_csv("top_similar_sae_features.csv", index=False)

    # Plot the top similar features
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_k), top_similarities.detach().cpu().numpy(), tick_label=top_indices.cpu().numpy())
    plt.title("Top Similar SAE Features to Linear Probe Weights")
    plt.xlabel("SAE Feature Index")
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return top_indices.tolist(), top_similarities.tolist()


# Example usage
if __name__ == "__main__":
    
    model_name = "gemma-2-2b"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = "layer_0/width_16k/canonical"
    sae_id_safe = sae_id.replace("/", "_").replace(".", "_")
    layer_idx = 0
    n_examples = 100
    
    out_dir = os.path.join(get_git_root(), "results", "linear_probes", model_name, sae_release, sae_id_safe)
    os.makedirs(out_dir, exist_ok=True)
    
    # Train the probe
    probe, model, tokenizer = train_probe(model_name=model_name, 
                                          layer_idx=layer_idx, 
                                          out_dir=out_dir, 
                                          n_samples=n_examples,
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
    top_features, similarities = find_similar_sae_features(probe, sae)

    # Print results
    print("\nTop neurons for 'one of' detection:")
    for neuron, weight in zip(top_neurons, top_weights):
        print(f"Neuron {neuron}: weight = {weight:.4f}")

    print("\nMost similar SAE features:")
    for feature, similarity in zip(top_features, similarities):
        print(f"Feature {feature}: similarity = {similarity:.4f}")
