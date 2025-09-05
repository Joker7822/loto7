
# trainers.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from early_stopping import EarlyStopping, EarlyStopConfig

def train_lstm_with_early_stopping(model, X, y, *, max_epochs=200, batch_size=64, val_ratio=0.2,
                                   lr=1e-3, patience=12, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = TensorDataset(X, y)
    n_val = int(len(ds) * val_ratio)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    es = EarlyStopping(EarlyStopConfig(patience=patience, min_delta=1e-4, mode="min", restore_best=True))

    for epoch in range(1, max_epochs+1):
        model.train()
        total = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                out = model(bx)
                loss = criterion(out, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        train_loss = total / max(1, len(train_loader))

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                vloss = criterion(out, by)
                vtotal += vloss.item()
        val_loss = vtotal / max(1, len(val_loader))
        print(f"[LSTM][Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f}")
        if es.step(val_loss, model):
            print(f"[LSTM] Early stopped at epoch {epoch}. Best val={es.best:.4f}")
            break
    return model

def train_gan_with_early_stopping(gan, real_data_tensor, *, max_epochs=3000, batch_size=32,
                                  lr=1e-3, patience=10, device=None, print_every=200):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan.generator.to(device)
    gan.discriminator.to(device)

    optimizer_G = optim.Adam(gan.generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()
    es = EarlyStopping(EarlyStopConfig(patience=patience, min_delta=1e-4, mode="min", restore_best=True))

    n = real_data_tensor.size(0)
    for epoch in range(1, max_epochs+1):
        gan.generator.train()
        gan.discriminator.train()

        idx = torch.randint(0, n, (batch_size,))
        real_batch = real_data_tensor[idx].to(device)

        noise = torch.randn(batch_size, gan.noise_dim, device=device)
        fake_batch = gan.generator(noise).detach()

        optimizer_D.zero_grad(set_to_none=True)
        d_real = gan.discriminator(real_batch)
        d_fake = gan.discriminator(fake_batch)
        d_loss = (criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))) * 0.5
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad(set_to_none=True)
        noise = torch.randn(batch_size, gan.noise_dim, device=device)
        fake = gan.generator(noise)
        g_loss = criterion(gan.discriminator(fake), torch.ones_like(d_fake))
        g_loss.backward()
        optimizer_G.step()

        if epoch % print_every == 0:
            print(f"[GAN][Epoch {epoch}] D={d_loss.item():.4f} G={g_loss.item():.4f}")

        if es.step(float(g_loss.item()), gan.generator):
            print(f"[GAN] Early stopped at epoch {epoch}. Best G loss={es.best:.4f}")
            break
    return gan
