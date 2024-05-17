# MechInterp project: Finding positional subspaces in the residual stream of LLMs
#
# 1. Synthetic model and data setting
#
# In this first experiment, we give a model some entangled input of token embeddings
# and positional encodings and train it to predict the position, like positional
# neurons do in Voita et al. (2023) (see https://arxiv.org/pdf/2309.04827).
# We then identify positional neurons and construct a subspace from their weight
# vectors (using PCA) to see whether this subspace is roughly the embedding space of
# the positional information.
#
# To establish a ground truth, we create the training dataset by embedding token
# information and positional information in the input space using two matrices spanning
# orthogonal spaces.
#
# In a subsequent experiment, we will apply the same subspace-retrieving mechanism to
# the residual stream of a pre-trained LLM (of course, we don't know the "real"
# positional subspace in this case).
#
# Steps:
# 1. Create token and position subspaces in the embedding space
# 2. Create dataset and embed it using the subspace projections
# 3. Train FFN to predict position of tokens
# 4. Identify positional neurons using the method in Voita et al. (2023)
# 5. Gather all weight vectors of the positional neurons
# 6. Calculate the best-fitting subspace for these vectors
# 7. Check if this subspace aligns with the true positional subspace
# 8. Compare the alignment with that of non-positional neurons


# %% Imports
from jaxtyping import Float

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import matplotlib.pyplot as plt
import einops
from scipy.linalg import null_space


# %% Model
class SimpleNN(nn.Module):
    def __init__(self, d_model, seq_len, n_hidden):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(d_model, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, seq_len)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# %% Dataset generation
class ActivationDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def generate_data(n_samples, n_vocab, d_tokens, seq_len, d_pos, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    d_model = d_tokens + d_pos

    # Define random matrices to embed tokens and positions into the residual stream
    W_embed = torch.rand(n_vocab, d_model)
    W_pos = torch.rand(seq_len, d_model)

    # Separate tokens and positions into orthogonal subspaces of dimension d_tokens and
    # d_pos, respectively
    S_tokens = torch.rand(d_model, d_tokens)
    S_pos = torch.tensor(null_space(S_tokens.T))

    # Define the corresponding projection matrices
    P_tokens = S_tokens @ torch.linalg.inv(S_tokens.T @ S_tokens) @ S_tokens.T
    P_pos = S_pos @ torch.linalg.inv(S_pos.T @ S_pos) @ S_pos.T

    # For our later analysis, calculate orthonormal bases for both subspaces
    B_tokens, _ = torch.linalg.qr(S_tokens)
    B_pos, _ = torch.linalg.qr(S_pos)

    # Combine everything to get the final embedding matrices
    W_embed = W_embed @ P_tokens
    W_pos = W_pos @ P_pos

    # Generate tokens and embed them
    tokens = torch.randint(low=0, high=n_vocab, size=(n_samples, seq_len))
    tokens_one_hot = torch.nn.functional.one_hot(tokens, num_classes=n_vocab).float()

    token_embeddings = einops.einsum(
        W_embed,
        tokens_one_hot,
        'n_vocab d_model, n seq_len n_vocab -> n seq_len d_model'
    )

    # Generate positional information and embed it
    positions_one_hot = torch.eye(seq_len)
    position_embeddings = einops.einsum(
        W_pos,
        positions_one_hot,
        'seq_len d_model, seq_len seq_len -> seq_len d_model'
    )

    # Create the dataset
    X = token_embeddings + position_embeddings  # Note that this summation is reversible since the subspaces are orthogonal
    y = torch.arange(seq_len).repeat(n_samples, 1)

    # Return everything we need for training and analysis
    return ActivationDataset(X, y), B_tokens, B_pos


def generate_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# %% Training
def train(model, dataloader, num_epochs):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss/10:.4f}')
                running_loss = 0.0


# %% Analysis
def calculate_mutual_information(
        activations: Float[torch.Tensor, 'batch seq_len neurons']
        ):
    activation_frequency = (activations > 0).float()

    frequency_position = activation_frequency.mean(dim=0)  # shape = (seq_len, neurons)
    frequency_overall = activation_frequency.mean(dim=(0, 1))  # shape = (neurons, )

    return (frequency_position * torch.nan_to_num(
        torch.log(frequency_position / frequency_overall), nan=0)
        + (1 - frequency_position) *
        torch.nan_to_num(
            torch.log((1 - frequency_position) / (1 - frequency_overall)), nan=0
        )).mean(dim=0)


def determine_positional_neurons(
        activations: Float[torch.Tensor, 'batch seq_len neurons'],
        threshold: Float = 0.05
        ):
    mutual_information = calculate_mutual_information(activations)

    positional_neurons = torch.where(mutual_information > threshold)[0]
    return positional_neurons.tolist(), mutual_information


# %%
def plot_layer_activation_frequency_per_position(
        activations: Float[Tensor, 'batch seq_len'], neurons=None
        ):
    _, _, n_neurons = activations.shape

    if neurons is None:
        neurons = list(range(n_neurons))

    activation_frequency = (activations > 0).float()
    positional_neurons, mutual_information = determine_positional_neurons(activations)

    _, axes = plt.subplots(nrows=1, ncols=len(neurons), figsize=(len(neurons) * 3, 3))

    for neuron, ax, mutual_information_ in zip(neurons, axes, mutual_information):
        neuron_activation_frequency = activation_frequency[:, :, neuron]
        neuron_activation_frequency_per_position = neuron_activation_frequency.mean(
            dim=0
            )
        color = 'red' if neuron in positional_neurons else 'black'

        ax.plot(neuron_activation_frequency_per_position, color=color, linestyle='-')
        ax.set_title(f'Neuron {neuron} (I={mutual_information_:.3f})')
        ax.set_xlabel('Position')
        ax.set_ylabel('Activation Frequency')

    plt.tight_layout()
    plt.show()


# %%
# Check if neuron weight vector is mostly aligned with positional subspace
def analyze_neuron_weights(weights, neurons=None):
    n_neurons, _ = weights.shape

    if neurons is None:
        neurons = list(range(n_neurons))

    for neuron in neurons:
        # Get weight vector of the neuron
        v = weights[neuron]

        # Project the weight vector onto both subspaces
        Proj_tokens = B_tokens @ B_tokens.T @ v
        Proj_pos = B_pos @ B_pos.T @ v

        # Calculate the respective magnitudes of the projections
        magnitude_tokens = torch.linalg.norm(Proj_tokens)
        magnitude_pos = torch.linalg.norm(Proj_pos)
        total_magnitude = torch.linalg.norm(v)

        # Calculate the relative magnitudes
        proportion_tokens = (magnitude_tokens ** 2) / (total_magnitude ** 2)
        proportion_pos = (magnitude_pos ** 2) / (total_magnitude ** 2)

        print(f'Neuron {neuron}:')
        print(f'Proportion in tokens: {proportion_tokens:.3f}')
        print(f'Proportion in pos: {proportion_pos:.3f}\n')


# %%
# Calculate k-dimensional subspace optimally aligned with weight vectors of given
# neurons
def fit_subspace(weights, neurons, subspace_dimension=None):
    _, d_model = weights.shape

    # Get all weight vectors
    weight_vectors = weights[neurons]

    # Create subspace
    # Dimension of the subspace
    if subspace_dimension is None:
        subspace_dimension = d_model // 2

    # Step 1: Center the data
    mean_vector = weight_vectors.mean(dim=0)
    centered_data = weight_vectors - mean_vector
    cov_matrix = torch.mm(
        centered_data.t(), centered_data
    ) / (weight_vectors.shape[0] - 1)

    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Step 4: Select the top-k eigenvectors (largest eigenvalues)
    topk_indices = torch.argsort(eigenvalues, descending=True)[:subspace_dimension]
    principal_components = eigenvectors[:, topk_indices]

    return principal_components


def calculate_positional_subspace(weights, activations, positional_neurons=None,
                                  subspace_dimension=None):
    if positional_neurons is None:
        positional_neurons, _ = determine_positional_neurons(activations)

    if subspace_dimension is None:
        subspace_dimension = d_model // 2

    return fit_subspace(weights,
                        neurons=positional_neurons,
                        subspace_dimension=subspace_dimension
                        )


# %%
# Check overlap between calculated positional subspace and ground truth
def calculate_subspace_alignment(calculated_subspace, true_subspace):
    singular_values = torch.linalg.svdvals(torch.mm(calculated_subspace.t(),
                                                    true_subspace))

    # Sum of cosines of the principal angles (a measure of overlap)
    overlap = singular_values.sum().item()

    # Frobenius norm as an alternative measure
    frobenius_norm = torch.linalg.norm(torch.mm(
        calculated_subspace.t(), true_subspace), 'fro'
        ).item()

    print(f'Overlap (sum of cosines): {overlap}')
    print(f'Frobenius norm of U^T V: {frobenius_norm}')


# %%
# Experiment

# Config
n_samples = 1024
batch_size = 32
n_vocab = 10
d_tokens = 3
seq_len = 20
d_pos = 5

d_model = d_tokens + d_pos

# Data
dataset, B_tokens, B_pos = generate_data(n_samples,
                                         n_vocab,
                                         d_tokens,
                                         seq_len,
                                         d_pos,
                                         seed=42
                                         )
dataloader = generate_dataloader(dataset, batch_size)

# Model
model = SimpleNN(d_model, seq_len, n_hidden=4 * d_model)

# Training
train(model, dataloader, num_epochs=500)

# Analysis
with torch.no_grad():
    activations = model.fc1(dataset.inputs)  # shape = (batch, seq_len, neurons)
    weights = model.fc1.weight.data

positional_neurons, _ = determine_positional_neurons(activations)
n_neurons = activations.shape[-1]
non_positional_neurons = list(set(range(n_neurons)) - set(positional_neurons))

print(f'Positional neurons: {positional_neurons}, \
      non-positional neurons: {non_positional_neurons}')

calculated_positional_subspace = fit_subspace(weights,
                                              neurons=positional_neurons,
                                              subspace_dimension=5)
calculated_non_positional_subspace = fit_subspace(weights,
                                                  neurons=non_positional_neurons,
                                                  subspace_dimension=5)

print('Alignment between calculated positional subspace and B_pos:')
calculate_subspace_alignment(calculated_positional_subspace, B_pos)

print('Alignment between calculated positional subspace and B_tokens:')
calculate_subspace_alignment(calculated_positional_subspace, B_tokens)

print()

print('Alignment between calculated non-positional subspace and B_pos:')
calculate_subspace_alignment(calculated_non_positional_subspace, B_pos)

print('Alignment between calculated non-positional subspace and B_tokens:')
calculate_subspace_alignment(calculated_non_positional_subspace, B_tokens)
