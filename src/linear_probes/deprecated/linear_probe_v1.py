import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sae_lens import SAE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoTokenizer


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


# def generate_examples(n_samples=1000):
#     """Generate example sentences with and without 'one of'"""
#     templates_with = [
#         "This is one of the best {}",
#         "That was one of my favorite {}",
#         "She is one of the most talented {}",
#         "It's one of those {} that everyone loves",
#         "He became one of the leading {}",
#     ]

#     templates_without = [
#         "This is the best {}",
#         "That was my favorite {}",
#         "She is the most talented {}",
#         "It's the kind of {} that everyone loves",
#         "He became the leading {}",
#     ]

#     nouns = [
#         "books",
#         "movies",
#         "artists",
#         "scientists",
#         "musicians",
#         "athletes",
#         "teachers",
#         "stories",
#         "places",
#         "experiences",
#     ]

#     examples = []
#     labels = []

#     for _ in range(n_samples // 2):
#         # Generate positive example (with "one of")
#         template = np.random.choice(templates_with)
#         noun = np.random.choice(nouns)
#         examples.append(template.format(noun))
#         labels.append(1)

#         # Generate negative example (without "one of")
#         template = np.random.choice(templates_without)
#         noun = np.random.choice(nouns)
#         examples.append(template.format(noun))
#         labels.append(0)

#     return examples, labels


# def generate_examples(n_samples=1000):
#     """Generate example sentences with and without 'monday'"""
#     templates_with = [
#         "I have a meeting on Monday {}",
#         "Monday {} is always busy",
#         "Let's schedule it for Monday {}",
#         "The Monday {} session was productive",
#         "I'll start the project on Monday {}",
#         "Monday {} is the deadline",
#         "We always have team meetings on Monday {}",
#         "The Monday {} report needs to be finished",
#     ]

#     templates_without = [
#         "I have a meeting on Tuesday {}",
#         "The weekend {} is always busy",
#         "Let's schedule it for tomorrow {}",
#         "The weekly {} session was productive",
#         "I'll start the project tomorrow {}",
#         "Friday {} is the deadline",
#         "We always have team meetings on Wednesday {}",
#         "The daily {} report needs to be finished",
#     ]

#     time_phrases = [
#         "morning",
#         "afternoon",
#         "evening",
#         "next week",
#         "this month",
#         "at 2 PM",
#         "after lunch",
#         "before noon",
#         "during the meeting",
#         "at the office",
#     ]

#     examples = []
#     labels = []

#     for _ in range(n_samples // 2):
#         # Generate positive example (with "monday")
#         template = np.random.choice(templates_with)
#         time_phrase = np.random.choice(time_phrases)
#         examples.append(template.format(time_phrase))
#         labels.append(1)

#         # Generate negative example (without "monday")
#         template = np.random.choice(templates_without)
#         time_phrase = np.random.choice(time_phrases)
#         examples.append(template.format(time_phrase))
#         labels.append(0)

#     # Shuffle the examples and labels together
#     combined = list(zip(examples, labels))
#     np.random.shuffle(combined)
#     examples, labels = zip(*combined)

#     return list(examples), list(labels)


def generate_examples(n_samples=1000):
    """Generate example sentences with number words (one-ten) and without"""

    templates_with = [
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

    templates_without = [
        "I ate {} {} for lunch",  # Will use digits instead of words
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
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ]

    objects = [
        "cookies",
        "books",
        "miles",
        "shirts",
        "cats",
        "minutes",
        "flowers",
        "goals",
        "birds",
        "students",
        "apples",
        "boxes",
        "papers",
        "pictures",
        "tasks",
    ]

    examples = []
    labels = []

    for _ in range(n_samples // 2):
        # Generate positive example (with number word)
        template = np.random.choice(templates_with)
        number = np.random.choice(number_words)
        object_word = np.random.choice(objects)
        examples.append(template.format(number, object_word))
        labels.append(1)

        # Generate negative example (with digit or no number)
        template = np.random.choice(templates_without)
        if np.random.random() < 0.5:
            # Use digit
            number = str(np.random.randint(1, 11))
        else:
            # Use phrase without number
            number = "some"
        object_word = np.random.choice(objects)
        examples.append(template.format(number, object_word))
        labels.append(0)

    # Shuffle the examples and labels together
    combined = list(zip(examples, labels))
    np.random.shuffle(combined)
    examples, labels = zip(*combined)

    return list(examples), list(labels)


def get_layer_activations(
    model, tokenizer, texts, layer_idx=-1, is_hooked_transformer=False, device=None
):
    """Get activations from a specific layer for a batch of texts"""
    activations = []
    device = device or get_device()

    with torch.no_grad():
        for text in tqdm(texts):
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
    device = get_device()

    # Generate example sentences
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

    # model = model.to(device)

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

    # Initialize probe and optimizer
    probe = LinearProbe(activations.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Training"):
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
    weights = probe.linear.weight.detach().cpu().numpy().squeeze()
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

    return top_indices.tolist(), top_similarities.tolist()


# Example usage
if __name__ == "__main__":
    # Train the probe
    probe, model, tokenizer = train_probe(model_name="gemma-2-2b", layer_idx=0)
    # probe, model, tokenizer = train_probe(model_name="gpt2-small", layer_idx=0)

    # Load SAE
    sae = load_sae(
        sae_release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_0/width_16k/canonical",
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
