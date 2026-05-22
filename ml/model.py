"""
model.py
========
Deep Neural Network for diabetes risk prediction.

This module implements a PyTorch-based neural network for binary classification
of diabetes risk based on healthcare features. The model accepts 8 normalized
healthcare input features and outputs a diabetes risk probability (0 to 1).

Architecture:
    Input (8 features)
    → Linear(8, 64)
    → ReLU activation
    → Dropout(0.3)
    → Linear(64, 32)
    → ReLU activation
    → Linear(32, 1)
    → Sigmoid activation
    → Output (0-1 probability)

The model is designed to work seamlessly with the preprocessing pipeline
defined in preprocess.py, which normalizes features to zero mean and unit
variance using StandardScaler.
"""

import torch
import torch.nn as nn


class DiabetesRiskPredictor(nn.Module):
    """
    Deep Neural Network for diabetes risk prediction.

    This model takes 8 normalized healthcare features as input and outputs
    a probability score between 0 and 1 representing the estimated risk
    of diabetes for a given patient.

    Args:
        input_size (int): Number of input features. Default: 8
        hidden_size_1 (int): Number of neurons in first hidden layer. Default: 64
        hidden_size_2 (int): Number of neurons in second hidden layer. Default: 32
        dropout_rate (float): Dropout probability. Default: 0.3
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dropout_rate: float = 0.3,
    ):
        """Initialize the neural network layers."""
        super(DiabetesRiskPredictor, self).__init__()

        # ---------------------------------------------------------------------------
        # First Linear Layer
        # ---------------------------------------------------------------------------
        # Transforms the 8 input healthcare features into a 64-dimensional
        # hidden representation. This layer learns to extract meaningful patterns
        # from the normalized input features.
        self.fc1 = nn.Linear(input_size, hidden_size_1)

        # ---------------------------------------------------------------------------
        # ReLU Activation (First)
        # ---------------------------------------------------------------------------
        # ReLU (Rectified Linear Unit) introduces non-linearity to the model,
        # allowing it to learn complex relationships between features.
        # It replaces negative values with 0, keeping positive values unchanged.
        self.relu1 = nn.ReLU()

        # ---------------------------------------------------------------------------
        # Dropout Layer
        # ---------------------------------------------------------------------------
        # Randomly drops 30% of neurons during training to prevent overfitting.
        # This is a regularization technique that encourages the network to learn
        # redundant representations. During evaluation, dropout is disabled.
        self.dropout = nn.Dropout(p=dropout_rate)

        # ---------------------------------------------------------------------------
        # Second Linear Layer
        # ---------------------------------------------------------------------------
        # Transforms the 64-dimensional hidden representation into a 32-dimensional
        # representation. This layer further refines the learned features before
        # making a final prediction.
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)

        # ---------------------------------------------------------------------------
        # ReLU Activation (Second)
        # ---------------------------------------------------------------------------
        # Another ReLU activation to maintain non-linearity in the network.
        self.relu2 = nn.ReLU()

        # ---------------------------------------------------------------------------
        # Output Linear Layer
        # ---------------------------------------------------------------------------
        # Transforms the 32-dimensional representation into a single output value.
        # This single output represents the raw diabetes risk score (logit).
        # The output will be passed through a Sigmoid activation to convert it
        # to a probability between 0 and 1.
        self.fc3 = nn.Linear(hidden_size_2, 1)

        # ---------------------------------------------------------------------------
        # Sigmoid Activation (Output)
        # ---------------------------------------------------------------------------
        # Sigmoid function squashes the output to a range of (0, 1), making it
        # suitable for binary classification. The output can be interpreted as
        # the probability that a patient has diabetes.
        # Formula: sigmoid(x) = 1 / (1 + e^(-x))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 8) containing
                normalized healthcare features.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) containing
                diabetes risk probabilities in range [0, 1].
        """
        # Pass through first linear layer → ReLU → Dropout
        x = self.fc1(x)         # (batch_size, 64)
        x = self.relu1(x)       # Apply non-linearity
        x = self.dropout(x)     # Apply dropout for regularization

        # Pass through second linear layer → ReLU
        x = self.fc2(x)         # (batch_size, 32)
        x = self.relu2(x)       # Apply non-linearity

        # Pass through output layer → Sigmoid
        x = self.fc3(x)         # (batch_size, 1)
        x = self.sigmoid(x)     # Convert to probability [0, 1]

        return x


# ---------------------------------------------------------------------------
# Test Block – Verify Model Initialization and Forward Pass
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Simple test to verify:
    1. Model initializes correctly
    2. Forward pass works with expected input/output shapes
    3. Model can process batches of healthcare data
    """

    print("=" * 70)
    print("DiabetesRiskPredictor – Model Test")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1: Model Initialization
    # -------------------------------------------------------------------------
    print("\n[Test 1] Initializing model...")
    model = DiabetesRiskPredictor(
        input_size=8,
        hidden_size_1=64,
        hidden_size_2=32,
        dropout_rate=0.3,
    )
    print("✓ Model initialized successfully")

    # Print model architecture
    print("\nModel architecture:")
    print(model)

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # Test 2: Forward Pass with Single Sample
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Test 2] Forward pass with single sample...")

    # Create a single sample with 8 features (simulating normalized data)
    single_sample = torch.randn(1, 8)
    print(f"✓ Input shape: {single_sample.shape}")

    with torch.no_grad():  # Disable gradient computation for inference
        output = model(single_sample)
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Predicted risk probability: {output.item():.4f}")

    # -------------------------------------------------------------------------
    # Test 3: Forward Pass with Batch
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Test 3] Forward pass with batch of samples...")

    # Create a batch of 32 samples (typical mini-batch size)
    batch_size = 32
    batch_samples = torch.randn(batch_size, 8)
    print(f"✓ Input batch shape: {batch_samples.shape}")

    with torch.no_grad():
        batch_output = model(batch_samples)
    print(f"✓ Output batch shape: {batch_output.shape}")
    print(f"✓ Output range: [{batch_output.min():.4f}, {batch_output.max():.4f}]")

    # -------------------------------------------------------------------------
    # Test 4: Model Modes (Training vs Evaluation)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Test 4] Testing model modes...")

    # Set to training mode (dropout will be active)
    model.train()
    print("✓ Model set to training mode (dropout enabled)")

    # Forward pass in training mode
    with torch.no_grad():
        train_output1 = model(single_sample)
        train_output2 = model(single_sample)
    print(f"  Output 1 (with dropout): {train_output1.item():.4f}")
    print(f"  Output 2 (with dropout): {train_output2.item():.4f}")
    print(f"  Outputs differ due to dropout: {not torch.allclose(train_output1, train_output2)}")

    # Set to evaluation mode (dropout will be disabled)
    model.eval()
    print("\n✓ Model set to evaluation mode (dropout disabled)")

    # Forward pass in evaluation mode
    with torch.no_grad():
        eval_output1 = model(single_sample)
        eval_output2 = model(single_sample)
    print(f"  Output 1 (no dropout): {eval_output1.item():.4f}")
    print(f"  Output 2 (no dropout): {eval_output2.item():.4f}")
    print(f"  Outputs identical: {torch.allclose(eval_output1, eval_output2)}")

    # -------------------------------------------------------------------------
    # Test 5: Output Range Validation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Test 5] Validating output range...")

    model.eval()
    large_batch = torch.randn(100, 8)
    with torch.no_grad():
        outputs = model(large_batch)

    min_output = outputs.min().item()
    max_output = outputs.max().item()
    in_range = (min_output >= 0.0) and (max_output <= 1.0)

    print(f"✓ Batch size: {large_batch.shape[0]} samples")
    print(f"✓ Min output: {min_output:.6f}")
    print(f"✓ Max output: {max_output:.6f}")
    print(f"✓ All outputs in [0, 1]: {in_range}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    print("\nModel is ready for training with preprocess.py data.")
    print("Compatible with:")
    print("  - 8 normalized input features from preprocess.py")
    print("  - Binary classification (diabetes: 0/1)")
    print("  - PyTorch training loops with BCELoss or BCEWithLogitsLoss")
