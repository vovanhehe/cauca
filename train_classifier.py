import os, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as D
from classifier import KeyClassifier

def main(
    data_root="datasets/cls",
    epochs=30,
    batch_size=64,
    lr=3e-4,
    out_path="runs/cls/best_cls.pt"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf_train = T.Compose([
        T.Resize((96,96)),
        T.RandomAffine(degrees=6, translate=(0.05,0.05), scale=(0.95,1.05)),
        T.ColorJitter(0.1,0.1,0.1,0.05),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tf_val = T.Compose([
        T.Resize((96,96)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_set = D.ImageFolder(os.path.join(data_root,"train"), transform=tf_train)
    val_set   = D.ImageFolder(os.path.join(data_root,"val"), transform=tf_val)
    classes = train_set.classes
    print("Classes:", classes)

    model = KeyClassifier(num_classes=len(classes)).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    best = 0.0
    for ep in range(1, epochs+1):
        model.train(); t0 = time.time()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()

        # validate
        model.eval(); correct=0; total=0
        with torch.inference_mode():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.numel()
        acc = correct/total if total else 0.0
        print(f"Epoch {ep}/{epochs} val_acc={acc:.4f} time={time.time()-t0:.1f}s")

        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": classes}, out_path)
            print("Saved:", out_path)

if __name__=="__main__":
    main()