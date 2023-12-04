
import torch
import torch.nn as nn
import torch.nn.functional as F
class GNN(nn.Module):
    def __init__(self, hidden_size):
        super(GNN, self).__init__()
        # Replace convolutional layers with fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Forward pass using fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class MovieLensGNN(nn.Module):
    def __init__(self, num_users, num_movies, hidden_dim):
        super(MovieLensGNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.movie_embedding = nn.Embedding(num_movies, hidden_dim)
        self.gnn = GNN(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)  # Output layer for rating prediction

    
    def forward(self, user_ids):
        user_emb = self.user_embedding(user_ids)  # Shape: [batch_size, hidden_dim]
        user_emb = user_emb.unsqueeze(2)  # Shape: [batch_size, hidden_dim, 1]

        all_movies_emb = self.movie_embedding.weight  # Shape: [num_movies, hidden_dim]

        # Compute scores for all movies for each user
        scores = torch.matmul(all_movies_emb, user_emb).squeeze(2)  # Shape: [num_movies, batch_size]
        return torch.softmax(scores.transpose(0, 1), dim=1)

class SessionGraph(nn.Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden
