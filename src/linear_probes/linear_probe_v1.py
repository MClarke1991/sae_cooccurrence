import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoTokenizer


class ActivationDataset(Dataset):
    def __init__(self, activations, labels):
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


def generate_examples(n_samples=1000):
    """Generate example sentences with and without 'one of'"""
    templates_with = [
        "This is one of the best {}",
        "That was one of my favorite {}",
        "She is one of the most talented {}",
        "It's one of those {} that everyone loves",
        "He became one of the leading {}",
    ]

    templates_without = [
        "This is the best {}",
        "That was my favorite {}",
        "She is the most talented {}",
        "It's the kind of {} that everyone loves",
        "He became the leading {}",
    ]

    nouns = [
        "books",
        "movies",
        "artists",
        "scientists",
        "musicians",
        "athletes",
        "teachers",
        "stories",
        "places",
        "experiences",
    ]

    examples = []
    labels = []

    for _ in range(n_samples // 2):
        # Generate positive example (with "one of")
        template = np.random.choice(templates_with)
        noun = np.random.choice(nouns)
        examples.append(template.format(noun))
        labels.append(1)

        # Generate negative example (without "one of")
        template = np.random.choice(templates_without)
        noun = np.random.choice(nouns)
        examples.append(template.format(noun))
        labels.append(0)

    return examples, labels


def get_layer_activations(
    model, tokenizer, texts, layer_idx=-1, is_hooked_transformer=False
):
    """Get activations from a specific layer for a batch of texts"""
    activations = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for text in texts:
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
        return torch.sigmoid(self.linear(x))


def train_probe(
    model_name="gpt2",
    layer_idx=0,
    n_samples=1000,
    batch_size=32,
    n_epochs=10,
    learning_rate=0.001,
):
    """Train a linear probe to detect 'one of' patterns"""

    # Generate example sentences
    examples, labels = generate_examples(n_samples)

    # Load model and tokenizer
    is_hooked_transformer = False
    try:
        model = HookedTransformer.from_pretrained(model_name)
        tokenizer = model.tokenizer
        is_hooked_transformer = True
    except Exception as e:
        print(f"Failed to load with HookedTransformer: {e}")
        print("Falling back to AutoModel")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get activations
    activations = get_layer_activations(
        model,
        tokenizer,
        examples,
        layer_idx,
        is_hooked_transformer=is_hooked_transformer,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = ActivationDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1)
    )
    test_dataset = ActivationDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test).reshape(-1, 1)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize probe and optimizer
    probe = LinearProbe(activations.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
        probe.train()
        total_loss = 0

        for batch_activations, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = probe(batch_activations)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        probe.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_activations, batch_labels in test_loader:
                outputs = probe(batch_activations)
                predicted = (outputs >= 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        accuracy = correct / total
        print(
            f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}, "
            f"Accuracy: {accuracy:.4f}"
        )

    return probe, model, tokenizer


def analyze_neurons(probe, n_top=10):
    """Analyze which neurons have the highest weights in the probe"""
    weights = probe.linear.weight.detach().numpy().squeeze()
    top_neurons = np.argsort(np.abs(weights))[-n_top:]

    # Plot top neuron weights
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_top), weights[top_neurons])
    plt.title("Top Neuron Weights for 'one of' Detection")
    plt.xlabel("Neuron Index")
    plt.ylabel("Weight")
    plt.xticks(range(n_top), [str(n) for n in top_neurons])  # type: ignore
    plt.show()

    return top_neurons, weights[top_neurons]


# Example usage
if __name__ == "__main__":
    # Train the probe
    probe, model, tokenizer = train_probe()

    # Analyze the most important neurons
    top_neurons, top_weights = analyze_neurons(probe)

    # Print results
    print("\nTop neurons for 'one of' detection:")
    for neuron, weight in zip(top_neurons, top_weights):
        print(f"Neuron {neuron}: weight = {weight:.4f}")
