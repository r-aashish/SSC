import logging
import argparse
import torch
from torch.utils.data import DataLoader
from data.preprocess import *
from algorithms.srgnn import MovieLensGNN
from evaluation.evaluation import *
import torch

def create_relevant_movies_dict(file_path):
    all_relevant_movies = {}
    with open(file_path, 'r') as file:
        for line in file:
            user_id, movie_id = line.strip().split(' ')
            user_id, movie_id = int(user_id), int(movie_id)

            if user_id not in all_relevant_movies:
                all_relevant_movies[user_id] = set()
            all_relevant_movies[user_id].add(movie_id)

    return all_relevant_movies

def train(model, data_loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            user_ids = batch['user_id'].to(device)
            movie_ids = batch['movie_id'].to(device)
            targets = torch.zeros(len(user_ids), device=device)
            optimizer.zero_grad()
            predictions = model(user_ids)  # Shape should be [batch_size, num_movies]
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}')

def main():
    parser = argparse.ArgumentParser(description='SR-GNN for MovieLens 1M Dataset')
    parser.add_argument('--dataset_path', type=str, help='Path to the MovieLens dataset file')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load and preprocess the data
    dataset, test_dataset = load_movielens_data(args.dataset_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # Initialize the model
    num_users = max(dataset.user_ids) + 1
    num_movies = max(dataset.movie_ids) + 1
    model = MovieLensGNN(num_users, num_movies, args.hidden_dim).to(device)

    # Define optimizer and loss function (modify the loss function as per your task)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()  # or another appropriate loss function
  
    # Replace with your file path
    all_relevant_movies = create_relevant_movies_dict(args.dataset_path)

    # Train the model
    train(model, data_loader, optimizer, criterion, args.epochs, device)
    ndcg_score = evaluate_ndcg(model, test_data_loader, device, all_relevant_movies, top_k=10)
    print(f"NDCG Score : {ndcg_score}")
    print("Done")

if __name__ == '__main__':
    main()
