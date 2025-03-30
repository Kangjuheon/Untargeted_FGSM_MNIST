import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# GPU 연결
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 로딩
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# FGSM (untargeted)
def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    output = model(x_adv)
    loss = F.cross_entropy(output, label.to(device))
    model.zero_grad()
    loss.backward()
    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv + eps * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()

# 훈련
train_losses = []

def train(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# 테스트 및 Clean & Adversarial Accuracy 측정
def test(model, test_loader, eps=0.3):
    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # 원본 정확도
        output = model(data)
        pred = output.argmax(dim=1)
        correct_clean += pred.eq(target).sum().item()

        # FGSM 공격 후 정확도
        data_adv = fgsm_untargeted(model, data, target, eps)
        output_adv = model(data_adv)
        pred_adv = output_adv.argmax(dim=1)
        correct_adv += pred_adv.eq(target).sum().item()

        total += len(data)

    clean_acc = 100 * correct_clean / total
    adv_acc = 100 * correct_adv / total
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Adversarial Accuracy (eps={eps}): {adv_acc:.2f}%")

# ✅ 실행 시작
if __name__ == "__main__":
    model = SimpleCNN().to(device)
    train(model, train_loader, epochs=3)
    test(model, test_loader, eps=0.3)
