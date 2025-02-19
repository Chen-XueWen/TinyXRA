import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalAttentionNetworks(nn.Module):
    def __init__(self, word_hidden_size, sentence_hidden_size, embedding_size, num_classes):
        super(HierarchicalAttentionNetworks, self).__init__()

        self.word_gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=word_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.word_attention = WordAttention(hidden_size=2 * word_hidden_size)
        
        self.sentence_gru = nn.GRU(
            input_size=2 * word_hidden_size,
            hidden_size=sentence_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.sentence_attention = SentenceAttention(hidden_size=2 * sentence_hidden_size)
        
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
        
        # Apply word-level attention to get a sentence vector.
        sentence_vectors, word_attn = self.word_attention(word_gru_out)
        
        # Reshape back to separate sentences: (batch_size, num_sentences, 2*word_hidden_size)
        sentence_vectors = sentence_vectors.view(batch_size, num_sentences, -1)
        
        sent_gru_out, _ = self.sentence_gru(sentence_vectors)
        
        # Apply sentence-level attention to get a document representation.
        document_vector, sentence_attn = self.sentence_attention(sent_gru_out)
        
        # Compute the final output logits for classification.
        output = self.fc(document_vector)
        return output, word_attn, sentence_attn

class WordAttention(nn.Module):
    def __init__(self, hidden_size):
        super(WordAttention, self).__init__()
        self.hidden_size = hidden_size
        # Project each hidden state into a new space.
        self.attn_fc = nn.Linear(hidden_size, hidden_size)
        # Learnable context vector for word-level attention.
        self.context_vector = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        """
        x shape: (batch_size * num_sentences, num_words, hidden_size)
        """
        u = torch.tanh(self.attn_fc(x))
        attn_scores = torch.matmul(u, self.context_vector)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        s = torch.sum(x * attn_weights_expanded, dim=1)
        return s, attn_weights

class SentenceAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SentenceAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_fc = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        """
        x shape: (batch_size, num_sentences, hidden_size)
        """
        u = torch.tanh(self.attn_fc(x))
        attn_scores = torch.matmul(u, self.context_vector)
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        s = torch.sum(x * attn_weights_expanded, dim=1)
        return s, attn_weights
