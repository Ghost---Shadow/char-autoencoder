import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lstm_cell import CustomLSTM
from preprocess import stringsToArray


class Seq2SeqAutoencoder(nn.Module):
    """
    Sequence-to-sequence autoencoder for character-level sequences.
    Encodes input sequences to a latent representation and decodes back.
    """

    def __init__(self, state_size=128, time_steps=10, input_size=27, unk_token=26):
        """
        Initialize the autoencoder.

        Args:
            state_size: Dimension of LSTM hidden state
            time_steps: Length of input sequences
            input_size: Size of input vocabulary (27 = 26 letters + space)
            unk_token: Token used for padding/unknown characters
        """
        super(Seq2SeqAutoencoder, self).__init__()

        self.UNK_TOKEN = unk_token
        self.state_size = state_size
        self.time_steps = time_steps
        self.input_size = input_size
        self.interp_steps = 9  # 10x10 grid (increased from 4 for 5x5)

        # Custom LSTM
        self.lstm = CustomLSTM(input_size, state_size)

        # Prediction layers
        self.W1 = nn.Parameter(torch.randn(state_size, state_size // 2))
        self.B1 = nn.Parameter(torch.randn(state_size // 2))

        self.W2 = nn.Parameter(torch.randn(state_size // 2, input_size))
        self.B2 = nn.Parameter(torch.randn(input_size))

    def _get_predictions(self, outputs):
        """
        Convert LSTM outputs to character predictions.

        Args:
            outputs: List of LSTM outputs, each (batch_size, state_size)

        Returns:
            predictions: Tensor of shape (batch_size, time_steps, input_size)
        """
        # Stack outputs: (time_steps, batch_size, state_size)
        outputs = torch.stack(outputs, dim=0)
        # Transpose to: (batch_size, time_steps, state_size)
        outputs = outputs.transpose(0, 1)

        # First prediction layer with sigmoid activation
        layer1 = torch.sigmoid(outputs @ self.W1 + self.B1)

        # Second prediction layer (logits)
        predictions = layer1 @ self.W2 + self.B2

        return predictions

    def _get_y(self, x):
        """
        Get target by reversing input sequence.

        Args:
            x: Input tensor (batch_size, time_steps, input_size)

        Returns:
            y: Reversed input (batch_size, time_steps, input_size)
        """
        return torch.flip(x, dims=[1])

    def forward(self, x):
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor of shape (batch_size, time_steps) with integer indices

        Returns:
            predictions: Logits of shape (batch_size, time_steps, input_size)
            y: Target (reversed one-hot encoded input)
        """
        # Convert to one-hot encoding
        x_onehot = F.one_hot(
            x, num_classes=self.input_size
        ).float()  # (batch_size, time_steps, input_size)

        # Get reversed target
        y = self._get_y(x_onehot)

        # Unstack along time dimension to list
        x_list = [x_onehot[:, t, :] for t in range(self.time_steps)]

        # Autoencode
        outputs = self.lstm.autoencode(x_list)

        # Get predictions
        predictions = self._get_predictions(outputs)

        return predictions, y

    def encode(self, x):
        """
        Encode input sequence to latent representation.

        Args:
            x: Input tensor of shape (batch_size, time_steps) with integer indices

        Returns:
            s: Hidden state (batch_size, state_size)
            c: Cell state (batch_size, state_size)
        """
        # Convert to one-hot encoding
        x_onehot = F.one_hot(x, num_classes=self.input_size).float()

        # Unstack along time dimension
        x_list = [x_onehot[:, t, :] for t in range(self.time_steps)]

        # Encode
        s, c = self.lstm.encode(x_list)

        return s, c

    def decode(self, s, c):
        """
        Decode from latent representation.

        Args:
            s: Hidden state (batch_size, state_size)
            c: Cell state (batch_size, state_size)

        Returns:
            predictions: Argmax predictions (batch_size, time_steps)
        """
        # Decode
        outputs = self.lstm.decode(s, c, self.time_steps)

        # Get predictions
        predictions = self._get_predictions(outputs)

        # Argmax to get character indices
        predictions = torch.argmax(predictions, dim=2)

        return predictions

    def compute_loss(self, predictions, y):
        """
        Compute cross-entropy loss.

        Args:
            predictions: Logits (batch_size, time_steps, input_size)
            y: Target one-hot vectors (batch_size, time_steps, input_size)

        Returns:
            loss: Scalar loss value
        """
        # Reshape for cross entropy
        predictions_flat = predictions.reshape(-1, self.input_size)
        y_indices = torch.argmax(y, dim=2).reshape(-1)

        loss = F.cross_entropy(predictions_flat, y_indices)

        return loss

    def compute_accuracy(self, predictions, y):
        """
        Compute character-level accuracy.

        Args:
            predictions: Logits (batch_size, time_steps, input_size)
            y: Target one-hot vectors (batch_size, time_steps, input_size)

        Returns:
            accuracy: Scalar accuracy value
        """
        pred_indices = torch.argmax(predictions, dim=2)
        y_indices = torch.argmax(y, dim=2)

        correct = (pred_indices == y_indices).float()
        accuracy = correct.mean()

        return accuracy

    def argmax_to_string(self, v):
        """
        Convert argmax indices to string.

        Args:
            v: Numpy array of character indices

        Returns:
            s: Decoded string
        """
        s = ""
        for i in v:
            if i == self.UNK_TOKEN:
                s += " "
            else:
                s += chr(i + ord("a"))

        # Reverse the string (since output is reversed)
        s = s[::-1]
        return s

    def _get_grid(self, a, b, c):
        """
        Create interpolation grid between three vectors.

        Args:
            a, b, c: Vectors to interpolate between

        Returns:
            d: Grid of interpolated vectors
        """
        d = np.zeros([self.interp_steps + 1, self.interp_steps + 1, a.shape[0]])

        ab = b - a
        ac = c - a

        for i in range(self.interp_steps + 1):
            t1 = i / self.interp_steps
            for j in range(self.interp_steps + 1):
                t2 = j / self.interp_steps
                d[i, j] = a + (t1 * ab + t2 * ac)

        return d

    def _get_interpolated_vectors(self, s, c):
        """
        Get interpolated vectors for visualization.

        Args:
            s: Hidden states for 3 examples
            c: Cell states for 3 examples

        Returns:
            s_m: Grid of interpolated hidden states
            c_m: Grid of interpolated cell states
        """
        batch_size = (self.interp_steps + 1) ** 2
        s_m = np.reshape(
            self._get_grid(s[0], s[1], s[2]), [batch_size, self.state_size]
        )
        c_m = np.reshape(
            self._get_grid(c[0], c[1], c[2]), [batch_size, self.state_size]
        )
        return s_m, c_m

    @torch.no_grad()
    def generate_text_summary(self, strings, device="cpu"):
        """
        Generate interpolated text for visualization.

        Args:
            strings: List of 3 strings to interpolate between
            device: Device to run on

        Returns:
            result_strings: Grid of interpolated strings
        """
        assert len(strings) == 3

        # Encode given strings
        v = stringsToArray(strings)
        v = torch.from_numpy(v).squeeze(-1).long().to(device)
        s, c = self.encode(v)

        # Convert to numpy for interpolation
        s = s.cpu().numpy()
        c = c.cpu().numpy()

        # Interpolate in latent space
        s_interp, c_interp = self._get_interpolated_vectors(s, c)

        # Convert back to tensors
        s_interp = torch.from_numpy(s_interp).float().to(device)
        c_interp = torch.from_numpy(c_interp).float().to(device)

        # Decode
        indices_batch = self.decode(s_interp, c_interp)
        indices_batch = indices_batch.cpu().numpy()

        # Convert to strings
        result_strings = []
        for indices in indices_batch:
            result_strings.append(self.argmax_to_string(indices))

        result_strings = np.reshape(
            np.array(result_strings), [self.interp_steps + 1, self.interp_steps + 1]
        )

        return result_strings

    @torch.no_grad()
    def compute_parallelogram_error(self, strings, device="cpu"):
        """
        Compute the "wobbliness" of the parallelogram by measuring L2 distance
        between ideal interpolated points and actual decoded/re-encoded points.

        Args:
            strings: List of 3 strings to interpolate between
            device: Device to run on

        Returns:
            Dictionary with error statistics and visualization data
        """
        assert len(strings) == 3

        # Encode given strings
        v = stringsToArray(strings)
        v = torch.from_numpy(v).squeeze(-1).long().to(device)
        s_orig, c_orig = self.encode(v)

        # Convert to numpy for interpolation
        s_orig_np = s_orig.cpu().numpy()
        c_orig_np = c_orig.cpu().numpy()

        # Interpolate in latent space (IDEAL parallelogram)
        s_ideal, c_ideal = self._get_interpolated_vectors(s_orig_np, c_orig_np)

        # Convert back to tensors
        s_ideal_torch = torch.from_numpy(s_ideal).float().to(device)
        c_ideal_torch = torch.from_numpy(c_ideal).float().to(device)

        # Decode (this introduces quantization error)
        indices_batch = self.decode(s_ideal_torch, c_ideal_torch)

        # Re-encode the decoded strings to get ACTUAL positions
        s_actual, c_actual = self.encode(indices_batch)
        s_actual_np = s_actual.cpu().numpy()
        c_actual_np = c_actual.cpu().numpy()

        # Compute L2 distances
        s_distances = np.linalg.norm(s_ideal - s_actual_np, axis=1)
        c_distances = np.linalg.norm(c_ideal - c_actual_np, axis=1)
        total_distances = np.sqrt(s_distances**2 + c_distances**2)

        # Reshape for grid
        s_distances_grid = s_distances.reshape(
            self.interp_steps + 1, self.interp_steps + 1
        )
        c_distances_grid = c_distances.reshape(
            self.interp_steps + 1, self.interp_steps + 1
        )
        total_distances_grid = total_distances.reshape(
            self.interp_steps + 1, self.interp_steps + 1
        )

        return {
            "mean_error": float(np.mean(total_distances)),
            "max_error": float(np.max(total_distances)),
            "min_error": float(np.min(total_distances)),
            "std_error": float(np.std(total_distances)),
            "error_grid": total_distances_grid.tolist(),
            "s_ideal": s_ideal,  # For visualization
            "c_ideal": c_ideal,
            "s_actual": s_actual_np,
            "c_actual": c_actual_np,
        }
