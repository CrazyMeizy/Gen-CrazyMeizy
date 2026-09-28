from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from time import perf_counter

import matplotlib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 37
CLASSES = (0, 1, 2)
IMAGE_SIZE = 28
N_PIXELS = IMAGE_SIZE ** 2
LATENT_SIZE = 16
VAE_LATENT_SIZE = 16
BATCH_SIZE = 256
AE_EPOCHS = 50
VAE_EPOCHS = 50
LEARNING_RATE = 0.001
N_SAMPLES = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


@dataclass
class Dataset:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset

    def loader(self, split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(getattr(self, split), batch_size=BATCH_SIZE, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(SEED))


@dataclass
class Metrics:
    reconstruction: float
    kl: float

    @property
    def loss(self) -> float:
        return self.reconstruction + self.kl


@dataclass
class EpochResult:
    model: str
    epoch: int
    train_loss: float
    validation_loss: float
    train_reconstruction: float
    validation_reconstruction: float
    train_kl: float
    validation_kl: float


def load_dataset() -> Dataset:
    datasets = []
    for train in (True, False):
        source = MNIST(DATA_DIR, train=train, download=True)
        mask = torch.isin(source.targets, torch.tensor(CLASSES))
        x = source.data[mask].float().flatten(1) / 255
        datasets.append(TensorDataset(x, source.targets[mask]))
    x, y = datasets[0].tensors
    train, validation = train_test_split(
        np.arange(len(y)), test_size=0.1, stratify=y.numpy(), random_state=SEED,
    )
    return Dataset(TensorDataset(x[train], y[train]),
                   TensorDataset(x[validation], y[validation]), datasets[1])


class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_PIXELS, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, LATENT_SIZE),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_SIZE, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, N_PIXELS),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))

    def loss(self, x: Tensor, _: Tensor) -> tuple[Tensor, Tensor]:
        reconstruction = F.binary_cross_entropy_with_logits(self(x), x, reduction="sum") / len(x)
        return reconstruction, x.new_zeros(())


class ConditionalVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_PIXELS + len(CLASSES), 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
        )
        self.mean = nn.Linear(64, VAE_LATENT_SIZE)
        self.log_variance = nn.Linear(64, VAE_LATENT_SIZE)
        self.decoder = nn.Sequential(
            nn.Linear(VAE_LATENT_SIZE + len(CLASSES), 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, N_PIXELS),
        )

    def decode(self, z: Tensor, condition: Tensor) -> Tensor:
        return self.decoder(torch.cat((z, condition), dim=1))

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.encoder(torch.cat((x, condition), dim=1))
        mean, log_variance = self.mean(hidden), self.log_variance(hidden)
        z = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return self.decode(z, condition), mean, log_variance

    def loss(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        condition = F.one_hot(y, num_classes=len(CLASSES)).float()
        logits, mean, log_variance = self(x, condition)
        reconstruction = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
        kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(1).mean()
        return reconstruction, kl


class Trainer:
    def __init__(self, model: Autoencoder | ConditionalVAE, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.history: list[EpochResult] = []

    def run_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer | None = None) -> Metrics:
        is_training = optimizer is not None
        self.model.train(is_training)
        reconstruction_sum, kl_sum = 0.0, 0.0
        with torch.set_grad_enabled(is_training):
            for x, y in loader:
                reconstruction, kl = self.model.loss(x.to(self.device), y.to(self.device))
                loss = reconstruction + kl
                if is_training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                reconstruction_sum += reconstruction.item() * len(x)
                kl_sum += kl.item() * len(x)
        return Metrics(reconstruction_sum / len(loader.dataset), kl_sum / len(loader.dataset))

    def fit(self, name: str, data: Dataset, epochs: int) -> dict:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        train_loader = data.loader("train", shuffle=True)
        validation_loader = data.loader("validation")
        best_loss, best_epoch = float("inf"), 0
        checkpoint = os.path.join(OUTPUT_DIR, f"{name}.pt")
        for epoch in range(1, epochs + 1):
            train = self.run_epoch(train_loader, optimizer)
            validation = self.run_epoch(validation_loader)
            self.history.append(EpochResult(name, epoch, train.loss, validation.loss,
                                            train.reconstruction, validation.reconstruction, train.kl, validation.kl))
            if validation.loss < best_loss:
                best_loss, best_epoch = validation.loss, epoch
                torch.save(self.model.state_dict(), checkpoint)
            print(f"{name} {epoch:2d}/{epochs}: train={train.loss:.2f}, "
                  f"val={validation.loss:.2f}", flush=True)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        test = self.run_epoch(data.loader("test"))
        print(f"{name}: лучшая эпоха {best_epoch}, test loss={test.loss:.2f}", flush=True)
        return {"model": name, "best_epoch": best_epoch, "validation_loss": best_loss,
                "test_loss": test.loss, "test_reconstruction": test.reconstruction, "test_kl": test.kl}


class GaussianGenerator:
    def __init__(self, x: np.ndarray, diagonal: bool) -> None:
        self.mean = x.mean(axis=0)
        self.diagonal = diagonal
        self.spread = x.std(axis=0) if diagonal else np.cov(x, rowvar=False)

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if self.diagonal:
            return rng.normal(self.mean, self.spread, size=(count, len(self.mean)))
        return rng.multivariate_normal(self.mean, self.spread, size=count)


class SMOTEGenerator:
    def __init__(self, x: np.ndarray) -> None:
        self.x = x
        self.neighbors = NearestNeighbors(n_neighbors=6).fit(x)

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.integers(len(self.x), size=count)
        # Первый сосед — сама исходная точка, выбираем из пяти остальных.
        neighbors = self.neighbors.kneighbors(self.x[indices], return_distance=False)[:, 1:]
        chosen = neighbors[np.arange(count), rng.integers(5, size=count)]
        weight = rng.random((count, 1))
        return self.x[indices] + weight * (self.x[chosen] - self.x[indices])


def save_csv(filename: str, rows: list[dict]) -> None:
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(history: list[EpochResult]) -> None:
    plots = {
        "training_curves": [("ae", "loss", "BCE"), ("cvae", "loss", "Total: BCE + KL")],
        "cvae_loss_components": [("cvae", "reconstruction", "Reconstruction: BCE"),
                                 ("cvae", "kl", "KL"), ("cvae", "loss", "Total: BCE + KL")],
    }
    for filename, panels in plots.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4))
        for ax, (name, metric, title) in zip(axes, panels):
            rows = [row for row in history if row.model == name]
            for split in ("train", "validation"):
                ax.plot([row.epoch for row in rows], [getattr(row, f"{split}_{metric}") for row in rows],
                        label=split)
            ax.set(title=f"{name.upper()} — {title}", xlabel="Эпоха", ylabel="Потеря / изображение")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=160)
        plt.close(fig)


def plot_images(
        filename: str, images: np.ndarray, labels: list[str], title: str,
        columns: list[str] | None = None,
) -> None:
    fig, axes = plt.subplots(len(images), len(images[0]), figsize=(13, 1.6 * len(images)), squeeze=False)
    for row, label, row_axes in zip(images, labels, axes):
        for image, ax in zip(row, row_axes):
            ax.imshow(image.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        row_axes[0].set_ylabel(label)
    if columns is not None:
        for ax, label in zip(axes[0], columns):
            ax.set_title(label, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=160)
    plt.close(fig)


@torch.no_grad()
def plot_reconstructions(model: Autoencoder, data: Dataset, device: torch.device) -> None:
    model.eval()
    x, y = data.test.tensors
    rows, labels = [], []
    for label in CLASSES:
        original = x[y == label][:N_SAMPLES]
        restored = model(original.to(device)).sigmoid().cpu()
        rows.extend((original.numpy(), restored.numpy()))
        labels.extend((f"{label}: оригинал", f"{label}: AE"))
    plot_images("reconstructions.png", np.stack(rows), labels, "AE: восстановление тестовых изображений")


@torch.no_grad()
def plot_generators(model: Autoencoder, data: Dataset, device: torch.device) -> None:
    model.eval()
    x, y = data.train.tensors
    latent = torch.cat([model.encoder(batch.to(device)).cpu() for batch, _ in data.loader("train")]).numpy()
    rng = np.random.default_rng(SEED)
    for space, features in (("pixels", x.numpy()), ("latent", latent)):
        gaussian_images, smote_images = [], []
        for label in CLASSES:
            points = features[y.numpy() == label]
            gaussian = GaussianGenerator(points, diagonal=space == "pixels")
            smote = SMOTEGenerator(points)
            for generator, rows in ((gaussian, gaussian_images), (smote, smote_images)):
                generated = generator.sample(N_SAMPLES, rng)
                if space == "latent":
                    z = torch.tensor(generated, dtype=torch.float32, device=device)
                    generated = model.decoder(z).sigmoid().cpu().numpy()
                else:
                    generated = generated.clip(0, 1)
                rows.append(generated)
        description = "пиксели" if space == "pixels" else "скрытое пространство AE → декодер"
        for name, rows in (("Gaussian", gaussian_images), ("SMOTE", smote_images)):
            plot_images(f"{name.lower()}_{space}.png", np.stack(rows), [str(c) for c in CLASSES],
                        f"{name}: {description}")


@torch.no_grad()
def plot_conditional(model: ConditionalVAE, device: torch.device) -> None:
    model.eval()
    torch.manual_seed(SEED)
    conditions = torch.eye(len(CLASSES), device=device)
    conditions = torch.cat((conditions, torch.full((1, len(CLASSES)), 1 / len(CLASSES), device=device)))
    z = torch.randn(N_SAMPLES, VAE_LATENT_SIZE, device=device)
    rows = [model.decode(z, condition.expand(N_SAMPLES, -1)).sigmoid().cpu().numpy()
            for condition in conditions]
    plot_images("conditional_generation.png", np.stack(rows), ["0", "1", "2", "⅓·0 + ⅓·1 + ⅓·2"],
                "CVAE: условная генерация; в каждом столбце одинаковый z")

    alpha = torch.linspace(0, 1, 11, device=device).unsqueeze(1)
    fixed_z = z[:1].expand(len(alpha), -1)
    rows, labels = [], []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        condition = (1 - alpha) * conditions[first] + alpha * conditions[second]
        rows.append(model.decode(fixed_z, condition).sigmoid().cpu().numpy())
        labels.append(f"{first} → {second}")
    plot_images("class_mixtures.png", np.stack(rows), labels,
                "CVAE: (1 − α) · первый класс + α · второй класс; z фиксирован",
                [f"α={value:.1f}" for value in alpha.flatten().tolist()])


def main() -> None:
    started = perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Устройство: {device}", flush=True)
    data = load_dataset()
    for split in ("train", "validation", "test"):
        labels = getattr(data, split).tensors[1]
        print(f"{split}: {len(labels)} изображений; по классам {torch.bincount(labels).tolist()}", flush=True)

    ae = Autoencoder()
    ae_trainer = Trainer(ae, device)
    ae_result = ae_trainer.fit("ae", data, AE_EPOCHS)
    plot_reconstructions(ae, data, device)
    plot_generators(ae, data, device)

    cvae = ConditionalVAE()
    cvae_trainer = Trainer(cvae, device)
    cvae_result = cvae_trainer.fit("cvae", data, VAE_EPOCHS)
    plot_conditional(cvae, device)
    history = ae_trainer.history + cvae_trainer.history
    plot_curves(history)
    save_csv("history.csv", [asdict(row) for row in history])
    save_csv("metrics.csv", [ae_result, cvae_result])
    print(f"Результаты: {OUTPUT_DIR}\nОбщее время: {perf_counter() - started:.1f} с", flush=True)


if __name__ == "__main__":
    main()
