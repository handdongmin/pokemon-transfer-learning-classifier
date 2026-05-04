import os
import copy
import time
import random
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


SEED = 42
BATCH_SIZE = 32
EPOCHS = 8
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_root(path):
    for root, dirs, files in os.walk(path):
        image_dirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
        if len(image_dirs) >= 100:
            return root
    return path


def get_dataloaders():
    path = kagglehub.dataset_download("lantian773030/pokemonclassification")
    print("Dataset path:", path)

    data_root = find_image_root(path)
    print("ImageFolder root:", data_root)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_root, transform=train_tf)

    train_size = int(len(full_dataset) * 0.7)
    val_size = int(len(full_dataset) * 0.15)
    test_size = len(full_dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_set.dataset = copy.deepcopy(full_dataset)
    test_set.dataset = copy.deepcopy(full_dataset)
    val_set.dataset.transform = test_tf
    test_set.dataset.transform = test_tf

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, full_dataset.classes


def build_model(exp_name, num_classes):
    if exp_name == "resnet18_frozen":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif exp_name == "resnet18_finetune":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif exp_name == "mobilenet_frozen":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif exp_name == "efficientnet_finetune":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError("Unknown experiment name")

    return model.to(DEVICE)


def train_one_experiment(exp_name, train_loader, val_loader, test_loader, num_classes, class_names):
    print(f"\n===== Experiment: {exp_name} =====")

    model = build_model(exp_name, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    best_acc = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    train_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        val_acc = evaluate(model, val_loader)["accuracy"]
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    test_metrics = evaluate(model, test_loader)
    print("Test:", test_metrics)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "exp_name": exp_name
    }, f"models/{exp_name}.pth")

    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_accs, label="val accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.title(exp_name)
    plt.savefig(f"results/{exp_name}_curve.png")
    plt.close()

    return {
        "experiment": exp_name,
        **test_metrics
    }


def evaluate(model, loader):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    set_seed(SEED)

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    experiments = [
        "resnet18_frozen",
        "resnet18_finetune",
        "mobilenet_frozen",
        "efficientnet_finetune"
    ]

    results = []

    for exp in experiments:
        result = train_one_experiment(
            exp,
            train_loader,
            val_loader,
            test_loader,
            num_classes,
            class_names
        )
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("results/experiment_results.csv", index=False)

    print("\nFinal Results")
    print(df)


if __name__ == "__main__":
    main()