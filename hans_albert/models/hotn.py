import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import sinkhorn

class HierarchicalOTNetworks(nn.Module):
    def __init__(self, word_hidden_size, sentence_hidden_size, embedding_size, num_classes, n_iters):
        super(HierarchicalOTNetworks, self).__init__()

        self.word_gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=word_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        # Replace standard WordAttention with the OT-based attention.
        self.word_attention = OTWordAttention(hidden_size=2 * word_hidden_size, n_iters=n_iters)
        
        self.sentence_gru = nn.GRU(
            input_size=2 * word_hidden_size,
            hidden_size=sentence_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        # Replace standard SentenceAttention with the OT-based attention.
        self.sentence_attention = OTSentenceAttention(hidden_size=2 * sentence_hidden_size, n_iters=n_iters)
        
        # Final classification layer.
        self.fc = nn.Linear(2 * sentence_hidden_size, num_classes)


    def forward(self, x):
        """
        x shape: (batch_size, num_sentences, num_words, embedding_size)
        """
        batch_size, num_sentences, num_words, embedding_size = x.size()
        
        # Process words in each sentence.
        # Reshape x to combine batch and sentence dimensions: (batch_size * num_sentences, num_words, embedding_size)
        x = x.reshape(batch_size * num_sentences, num_words, embedding_size)
        word_gru_out, _ = self.word_gru(x)
        
        # Apply word-level OT-based attention to get a sentence vector.
        sentence_vectors, word_attn = self.word_attention(word_gru_out)
        
        # Reshape back to separate sentences: (batch_size, num_sentences, 2*word_hidden_size)
        sentence_vectors = sentence_vectors.view(batch_size, num_sentences, -1)
        
        sent_gru_out, _ = self.sentence_gru(sentence_vectors)
        
        # Apply sentence-level OT-based attention to get a document representation.
        document_vector, sentence_attn = self.sentence_attention(sent_gru_out)
        
        # Compute the final output logits for classification.
        output = self.fc(document_vector)
        return output, word_attn, sentence_attn

class OTWordAttention(nn.Module):
    def __init__(self, hidden_size, num_prototypes=5, epsilon=1e-5, n_iters=5):
        """
        Args:
            hidden_size: Dimension of the hidden state coming from the GRU.
            num_prototypes: Number of learnable prototypes to align with.
            epsilon: Entropy regularization parameter.
            n_iters: Number of Sinkhorn iterations.
        """
        super(OTWordAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.epsilon = epsilon
        self.n_iters = n_iters

        # Transform the hidden states.
        self.attn_fc = nn.Linear(hidden_size, hidden_size)
        # Instead of a single context vector, use multiple prototypes.
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_size))
        init.xavier_uniform_(self.prototypes)  # or use init.xavier_normal_(self.prototypes)


    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size * num_sentences, num_words, hidden_size)
        
        Returns:
            s: Aggregated sentence vector of shape (batch_size * num_sentences, hidden_size)
            attn_weights: Attention weights over words of shape (batch_size * num_sentences, num_words)
        """
        # Transform the hidden states.
        u = torch.tanh(self.attn_fc(x))  # Shape: (B, n, hidden_size)
        B, n, d = u.shape
        
        # Compute the cost matrix between each word representation and each prototype.
        u_exp = u.unsqueeze(2)
        prototypes_exp = self.prototypes.unsqueeze(0).unsqueeze(0)
        cost_matrix = torch.norm(u_exp - prototypes_exp, dim=-1)  # Shape: (B, n, num_prototypes)
        
        # Compute the optimal transport plan.
        transport_plan = sinkhorn(cost_matrix, self.n_iters, self.epsilon)
        # Average over the prototypes to obtain attention weights for each word.
        attn_weights = transport_plan.mean(dim=-1)
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        
        # Compute the aggregated sentence vector as a weighted sum over the original word representations.
        s = torch.sum(x * attn_weights_expanded, dim=1)
        return s, attn_weights

class OTSentenceAttention(nn.Module):
    def __init__(self, hidden_size, num_prototypes=5, epsilon=1e-5, n_iters=5):
        """
        Args:
            hidden_size: Dimension of the hidden state coming from the sentence-level GRU.
            num_prototypes: Number of learnable prototypes.
            epsilon: Entropy regularization parameter.
            n_iters: Number of Sinkhorn iterations.
        """
        super(OTSentenceAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.epsilon = epsilon
        self.n_iters = n_iters

        self.attn_fc = nn.Linear(hidden_size, hidden_size)
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_size))
        init.xavier_uniform_(self.prototypes)  # or use init.xavier_normal_(self.prototypes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_sentences, hidden_size)
        
        Returns:
            s: Aggregated document vector of shape (batch_size, hidden_size)
            attn_weights: Attention weights over sentences of shape (batch_size, num_sentences)
        """
        u = torch.tanh(self.attn_fc(x))
        B, n, d = u.shape
        
        u_exp = u.unsqueeze(2)
        prototypes_exp = self.prototypes.unsqueeze(0).unsqueeze(0)
        cost_matrix = torch.norm(u_exp - prototypes_exp, dim=-1) # Shape: (B, n, num_prototypes)
        
        transport_plan = sinkhorn(cost_matrix, self.n_iters, self.epsilon) 
        attn_weights = transport_plan.mean(dim=-1)
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        
        s = torch.sum(x * attn_weights_expanded, dim=1)
        return s, attn_weights