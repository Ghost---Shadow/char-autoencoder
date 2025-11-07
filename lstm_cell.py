import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLSTM(nn.Module):
    """
    Custom LSTM implementation for sequence-to-sequence autoencoder.
    Implements LSTM from scratch with explicit gate calculations.
    """

    def __init__(self, input_size, state_size):
        """
        Initialize LSTM cell.

        Args:
            input_size: Dimension of input vectors
            state_size: Dimension of hidden state
        """
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.state_size = state_size

        # LSTM parameters for each gate (input, forget, output, candidate)
        # W: input-to-hidden weights
        # U: hidden-to-hidden weights
        # B: biases
        self.W_i = nn.Parameter(torch.empty(input_size, state_size).uniform_(-0.1, 0.1))
        self.U_i = nn.Parameter(torch.empty(state_size, state_size).uniform_(-0.1, 0.1))
        self.B_i = nn.Parameter(torch.empty(state_size).uniform_(-0.1, 0.1))

        self.W_f = nn.Parameter(torch.empty(input_size, state_size).uniform_(-0.1, 0.1))
        self.U_f = nn.Parameter(torch.empty(state_size, state_size).uniform_(-0.1, 0.1))
        self.B_f = nn.Parameter(torch.empty(state_size).uniform_(-0.1, 0.1))

        self.W_o = nn.Parameter(torch.empty(input_size, state_size).uniform_(-0.1, 0.1))
        self.U_o = nn.Parameter(torch.empty(state_size, state_size).uniform_(-0.1, 0.1))
        self.B_o = nn.Parameter(torch.empty(state_size).uniform_(-0.1, 0.1))

        self.W_g = nn.Parameter(torch.empty(input_size, state_size).uniform_(-0.1, 0.1))
        self.U_g = nn.Parameter(torch.empty(state_size, state_size).uniform_(-0.1, 0.1))
        self.B_g = nn.Parameter(torch.empty(state_size).uniform_(-0.1, 0.1))

    def _step(self, x_t, s_t, c_t):
        """
        Single LSTM step.

        Args:
            x_t: Input at time t (batch_size, input_size)
            s_t: Hidden state at time t (batch_size, state_size)
            c_t: Cell state at time t (batch_size, state_size)

        Returns:
            o_t: Output gate activation (batch_size, state_size)
            s_next: Next hidden state (batch_size, state_size)
            c_next: Next cell state (batch_size, state_size)
        """
        # Input gate
        i = torch.sigmoid(x_t @ self.W_i + s_t @ self.U_i + self.B_i)

        # Forget gate
        f = torch.sigmoid(x_t @ self.W_f + s_t @ self.U_f + self.B_f)

        # Output gate
        o = torch.sigmoid(x_t @ self.W_o + s_t @ self.U_o + self.B_o)

        # Candidate cell state
        g = torch.tanh(x_t @ self.W_g + s_t @ self.U_g + self.B_g)

        # Update cell state
        c_next = c_t * f + g * i

        # Update hidden state
        s_next = torch.tanh(c_next) * o

        return o, s_next, c_next

    def _unroll(self, x, s, c):
        """
        Unroll LSTM over sequence.

        Args:
            x: List of inputs at each time step, each (batch_size, input_size)
            s: Initial hidden state (batch_size, state_size)
            c: Initial cell state (batch_size, state_size)

        Returns:
            outputs: List of output gate activations
            s_next: Final hidden state
            c_next: Final cell state
        """
        time_steps = len(x)
        outputs = []
        s_t = s
        c_t = c

        for t in range(time_steps):
            o_t, s_t, c_t = self._step(x[t], s_t, c_t)
            outputs.append(o_t)

        return outputs, s_t, c_t

    def encode(self, x):
        """
        Encode input sequence to latent representation.

        Args:
            x: List of inputs at each time step, each (batch_size, input_size)

        Returns:
            s: Final hidden state (batch_size, state_size)
            c: Final cell state (batch_size, state_size)
        """
        batch_size = x[0].shape[0]
        device = x[0].device

        # Initialize states to zero
        s = torch.zeros(batch_size, self.state_size, device=device)
        c = torch.zeros(batch_size, self.state_size, device=device)

        # Unroll to get final states
        _, s, c = self._unroll(x, s, c)

        return s, c

    def decode(self, s, c, time_steps):
        """
        Decode from latent representation to output sequence.

        Args:
            s: Initial hidden state (batch_size, state_size)
            c: Initial cell state (batch_size, state_size)
            time_steps: Number of time steps to decode

        Returns:
            outputs: List of output gate activations
        """
        batch_size = s.shape[0]
        device = s.device

        # Use zero inputs for decoder
        x = [
            torch.zeros(batch_size, self.input_size, device=device)
            for _ in range(time_steps)
        ]

        # Unroll to get outputs
        outputs, _, _ = self._unroll(x, s, c)

        return outputs

    def autoencode(self, x):
        """
        Encode and decode sequence (autoencoder).

        Args:
            x: List of inputs at each time step, each (batch_size, input_size)

        Returns:
            outputs: List of output gate activations
        """
        time_steps = len(x)
        s, c = self.encode(x)
        outputs = self.decode(s, c, time_steps)
        return outputs
