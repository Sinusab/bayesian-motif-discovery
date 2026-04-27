"""
Position Weight Matrix and Position Frequency Matrix
for representing DNA sequence motifs.
"""

import numpy as np

# Standard nucleotide order
NUCLEOTIDES = ['A', 'C', 'G', 'T']
NUC_TO_IDX = {nuc: i for i, nuc in enumerate(NUCLEOTIDES)}
IDX_TO_NUC = {i: nuc for i, nuc in enumerate(NUCLEOTIDES)}


class PositionWeightMatrix:
    """
    A PWM stores the probability of each nucleotide at each position.
    
    Attributes:
        counts: numpy array of shape (motif_length, 4) with observed counts
        pseudocount: float, the prior weight added to every count
        probs: numpy array of shape (motif_length, 4) with posterior probabilities
    """
    
    def __init__(self, pseudocount=1.0):
        """
        Initialize with a pseudocount (our prior).
        
        Args:
            pseudocount: added to each count. Higher = more uncertainty,
                        lower = trust data more. Default 1.0 is a uniform Dirichlet(1,1,1,1) prior.
        """
        self.pseudocount = pseudocount
        self.counts = None
        self.motif_length = None
    
    def fit(self, sequences):
        """
        Build the PWM from aligned sequences (all same length).
        
        Args:
            sequences: list of strings, all uppercase, same length
                      e.g., ['TATAAT', 'TATGAT', 'TAGAAT']
        """
        if not sequences:
            raise ValueError("Empty sequence list")
        
        # Validate all same length
        lengths = set(len(s) for s in sequences)
        if len(lengths) != 1:
            raise ValueError(f"All sequences must have same length. Got: {lengths}")
        
        self.motif_length = lengths.pop()
        
        # Initialize count matrix with pseudocounts
        self.counts = np.full((self.motif_length, 4), self.pseudocount)
        
        # Add observed counts
        for seq in sequences:
            for pos, nuc in enumerate(seq.upper()):
                if nuc not in NUC_TO_IDX:
                    raise ValueError(f"Invalid nucleotide '{nuc}' at position {pos}")
                self.counts[pos, NUC_TO_IDX[nuc]] += 1
        
        # Compute posterior probabilities
        # Each row sums to: n_sequences + pseudocount*4
        row_sums = self.counts.sum(axis=1, keepdims=True)
        self.probs = self.counts / row_sums
    
    def score_sequence(self, sequence):
        """
        Calculate probability of this sequence given the motif model.
        P(sequence | motif) = ∏ P(nucleotide_i | position_i)
        
        Returns the product of probabilities (not log, for intuition).
        For long sequences, we'll switch to log-space later.
        """
        if len(sequence) != self.motif_length:
            raise ValueError(
                f"Sequence length {len(sequence)} != motif length {self.motif_length}"
            )
        
        prob = 1.0
        for pos, nuc in enumerate(sequence.upper()):
            prob *= self.probs[pos, NUC_TO_IDX[nuc]]
        return prob
    
    def score_log(self, sequence):
        """
        Log-probability: sum of logs instead of product.
        Prevents numerical underflow for long motifs.
        """
        if len(sequence) != self.motif_length:
            raise ValueError(
                f"Sequence length {len(sequence)} != motif length {self.motif_length}"
            )
        
        log_prob = 0.0
        for pos, nuc in enumerate(sequence.upper()):
            log_prob += np.log(self.probs[pos, NUC_TO_IDX[nuc]])
        return log_prob
    
    def consensus_sequence(self):
        """Return the most likely nucleotide at each position."""
        consensus = []
        for pos in range(self.motif_length):
            best_idx = np.argmax(self.probs[pos])
            consensus.append(IDX_TO_NUC[best_idx])
        return ''.join(consensus)
    
    def information_content(self):
        """
        Calculate information content at each position.
        IC = 2 + Σ p × log2(p)  (for DNA, max is 2 bits)
        High IC = strong preference, low IC = flexible position.
        """
        ic = np.zeros(self.motif_length)
        for pos in range(self.motif_length):
            p = self.probs[pos]
            # Avoid log(0)
            p = p[p > 0]
            entropy = -np.sum(p * np.log2(p))
            ic[pos] = 2.0 - entropy
        return ic
    
    def get_probability_matrix(self):
        """Return the probability matrix as a 2D array."""
        return self.probs.copy()
    
    def __repr__(self):
        if self.counts is None:
            return "PositionWeightMatrix(not fitted)"
        return f"PositionWeightMatrix(length={self.motif_length}, sequences used for training awaiting attribute)"


def create_pwm_from_sequences(sequences, pseudocount=1.0):
    """Convenience function to create a PWM from a list of sequences."""
    pwm = PositionWeightMatrix(pseudocount=pseudocount)
    pwm.fit(sequences)
    return pwm
