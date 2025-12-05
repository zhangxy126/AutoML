import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def load_and_preprocess_data(meta_path, label_path):
    meta_df = pd.read_excel(meta_path).rename(columns={'Dataset': 'Dataset_Name'})
    label_df = pd.read_excel(label_path).rename(columns={'Dataset': 'Dataset_Name'})

    expanded_data = []
    for _, row in label_df.iterrows():
        dataset = row['Dataset_Name']
        for i in range(1, 4):
            expanded_data.append([dataset, row[f'Top{i}_Category']])
    expanded_df = pd.DataFrame(expanded_data, columns=['Dataset_Name', 'Label'])

    merged_df = pd.merge(meta_df, expanded_df, on='Dataset_Name')

    X = merged_df.iloc[:, 1:-1].values.astype(np.float32)
    y = merged_df['Label'].values.astype(np.int32)

    X = np.nan_to_num(X, nan=0.0)
    X[X > 1e5] = 1e5
    X[X < -1e5] = -1e5

    return X, y

class DQN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

class FeatureSelector:
    def __init__(self, X, y, rewards_path, k=30):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X = X
        self.y = y
        self.n_features = X.shape[1]
        self.k = k

        self.rewards = torch.tensor(np.load(rewards_path),
                                    dtype=torch.float32).to(self.device)

        self.policy_net = DQN(self.n_features).to(self.device)
        self.target_net = DQN(self.n_features).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.target_update_freq = 5

        self.loss_history = []
        self.accuracy_history = []

        os.makedirs("results", exist_ok=True)

    def get_state(self, selected):
        state = torch.zeros(self.n_features, device=self.device)
        state[selected] = 1
        return state.unsqueeze(0)

    def select_action(self, state):
        if random.random() < self.epsilon:
            available = [i for i in range(self.n_features) if not state[0, i]]
            return random.choice(available) if available else None
        else:
            with torch.no_grad():
                q_values = self.policy_net(state).squeeze(0)
                q_values[state.bool().squeeze(0)] = -torch.inf
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((
            state.cpu().numpy().flatten(),
            action,
            reward,
            next_state.cpu().numpy().flatten()
        ))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.array([x[0] for x in batch]),
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor([x[1] for x in batch],
                               dtype=torch.long).to(self.device)
        rewards = torch.tensor([x[2] for x in batch],
                               dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([x[3] for x in batch]),
                                   dtype=torch.float32).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        return loss.item()

    def train(self, episodes=60):
        best_accuracy = 0.0
        best_features = []
        episode_losses = []

        progress_bar = tqdm(range(episodes), desc="Training DQN")
        for ep in progress_bar:
            selected = []
            state = self.get_state(selected)
            total_loss = 0.0
            valid_steps = 0

            for _ in range(self.k):
                action = self.select_action(state)
                if action is None:
                    break

                reward = self.rewards[action].item()
                next_state = self.get_state(selected + [action])

                self.store_transition(state, action, reward, next_state)

                loss = self.train_step()
                if loss > 0:
                    total_loss += loss
                    valid_steps += 1

                state = next_state
                selected.append(action)

            if ep % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.scheduler.step()

            if selected:
                current_acc = self._evaluate_selection(selected)
                self.accuracy_history.append(current_acc)

                if current_acc > best_accuracy:
                    best_accuracy = current_acc
                    best_features = selected.copy()
                    torch.save({
                        'policy_net': self.policy_net.state_dict(),
                        'target_net': self.target_net.state_dict(),
                        'features': best_features,
                        'accuracy': best_accuracy
                    }, "results/best_model.pth")

                avg_loss = total_loss / valid_steps if valid_steps > 0 else 0.0
                episode_losses.append(avg_loss)
                progress_bar.set_postfix({
                    "Acc": f"{current_acc:.4f}",
                    "Best": f"{best_accuracy:.4f}",
                    "Loss": f"{avg_loss:.4f}",
                    "Eps": f"{self.epsilon:.2f}"
                })

        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'features': best_features,
            'accuracy': best_accuracy
        }, "results/final_model.pth")
        self._plot_training_curves(episode_losses)

        return best_features

    def _evaluate_selection(self, selected):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []

        for train_idx, test_idx in skf.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train[:, selected], y_train)

            y_proba = model.predict_proba(X_test[:, selected])
            y_pred_top3 = np.argsort(y_proba, axis=1)[:, -3:]

            acc = self._compute_top3_accuracy(y_test, y_pred_top3)
            accuracies.append(acc)

        return np.mean(accuracies)

    def _compute_top3_accuracy(self, y_true, y_pred_top3):
        correct = 0
        total = len(y_true) // 3
        for i in range(total):
            true = {y_true[i * 3], y_true[i * 3 + 1], y_true[i * 3 + 2]}
            pred = set(y_pred_top3[i * 3])
            if true & pred:
                correct += 1
        return correct / total

    def _plot_training_curves(self, episode_losses):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(episode_losses, alpha=0.3, label='Raw Loss')
        window_size = 10
        weights = np.repeat(1.0, window_size) / window_size
        smoothed_loss = np.convolve(episode_losses, weights, 'valid')
        plt.plot(range(window_size - 1, len(episode_losses)), smoothed_loss,
                 linewidth=2, label=f'Smoothed (window={window_size})')
        plt.title("Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history)
        plt.title("Validation Accuracy")
        plt.xlabel("Episode")
        plt.ylabel("Top3 Accuracy")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("results/training_curves.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    META_PATH = '../meta-features/cleaned_file.xlsx'
    LABEL_PATH = '../dataset/top3_algorithms_avg.xlsx'
    REWARDS_PATH = 'reward/rewards.npy'

    X, y = load_and_preprocess_data(META_PATH, LABEL_PATH)

    meta_df = pd.read_excel(META_PATH)
    feature_names = meta_df.columns[1:].tolist()

    selector = FeatureSelector(
        X=X,
        y=y,
        rewards_path=REWARDS_PATH,
        k=30
    )

    best_features = selector.train(episodes=60)

    np.save("results/selected_features.npy", best_features)

    with open("results/selected_features.txt", "w") as f:
        f.write("Selected Feature Index\tFeature Name\n")
        for idx in best_features:
            f.write(f"{idx}\t{feature_names[idx]}\n")


