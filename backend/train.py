import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF
import sys
import os

sys.path.append(os.path.abspath("../backend"))
from preprocess import OASISFolderDataset
from model import CNNtoSNNConverter


class DeepSNNClassifier(nn.Module):
    def __init__(self, beta=0.95):
        super(DeepSNNClassifier, self).__init__()
        self.fc = nn.Linear(2048, 4)
        self.dropout = nn.Dropout(0.3)
        self.lif = snn.Leaky(beta=beta, threshold=0.5, init_hidden=True, output=True)

    def forward(self, incoming_spikes):
        current = self.fc(self.dropout(incoming_spikes))
        spk_out, mem_out = self.lif(current)
        return spk_out, mem_out


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing High-Speed SNN Training on: {device.type.upper()}")

    dataset_path = os.path.abspath("./Processed_Data")

    if not os.path.exists(dataset_path):
        print(f"❌ Error: Could not find Processed_Data at {dataset_path}")
        return

    print("📁 Loading Preprocessed OASIS Dataset...")
    train_dataset = OASISFolderDataset(root_dir=dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    print(f"✅ Loaded {len(train_dataset)} clean images.")

    retina = CNNtoSNNConverter().to(device)
    brain = DeepSNNClassifier().to(device)

    num_epochs = 30
    num_steps = 50
    learning_rate = 5e-4

    optimizer = torch.optim.Adam(
        list(retina.parameters()) + list(brain.parameters()), lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    class_weights = torch.tensor([3.0, 10.0, 1.0, 2.0]).to(device)
    loss_fn = SF.ce_count_loss(weight=class_weights)

    print("\n--- 🧠 COMMENCING NEUROMORPHIC TRAINING ---")

    for epoch in range(num_epochs):
        retina.train()
        brain.train()

        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)

            utils.reset(retina)
            utils.reset(brain)
            optimizer.zero_grad()

            spk_rec = []
            mem_rec = []

            for step in range(num_steps):
                spikes_2048, _ = retina(data)
                class_spikes, class_mem = brain(spikes_2048)
                spk_rec.append(class_spikes)
                mem_rec.append(class_mem)

            spk_rec = torch.stack(spk_rec)
            # just for debugging purposes
            total_spikes = spk_rec.sum(dim=0)
            print("Average spikes per batch:",total_spikes.mean().item())
            loss = loss_fn(spk_rec, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(retina.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(brain.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = spk_rec.sum(dim=0).max(1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            if batch_idx % 5 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )

        scheduler.step()

        epoch_acc = (correct_predictions / total_samples) * 100
        print(
            f"✅ EPOCH {epoch + 1} SUMMARY | Avg Loss: {total_loss / len(train_loader):.4f} | Accuracy: {epoch_acc:.2f}%\n"
        )

    print("💾 Training Complete. Saving Weights...")
    torch.save(retina.state_dict(), "snn_retina.pth")
    torch.save(brain.state_dict(), "snn_brain.pth")
    print("Done. Models saved successfully.")


if __name__ == "__main__":
    train_model()
