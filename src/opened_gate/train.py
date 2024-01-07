import lightning
import torch
import torchmetrics
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImagesDataModule(lightning.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # RGB
                    std=[0.229, 0.224, 0.225],  # RGB
                ),
            ]
        )

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # do not assign state (self.x = y)
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # do not assign state (self.x = y)
        dataset = ImageFolder(self.data_dir, transform=self.transform)

        # Split dataset into train (80%), val (10%), test (10%)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class ClassificationModule(lightning.LightningModule):
    def __init__(self, num_classes: int, backbone: str, learning_rate: float):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Define model
        if backbone == "resnet18":
            self.backbone = torchvision.models.resnet18(pretrained=True)
            self.backbone.fc = torch.nn.Linear(512, num_classes)
        else:
            raise NotImplementedError

        self.model = self.backbone

        # Define loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # Define metrics
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # Log metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss


def main():
    lightning.seed_everything(1410)

    # Create data module
    data = ImagesDataModule(data_dir="data/01_raw/camera-images", batch_size=16)

    # Create model
    model = ClassificationModule(
        num_classes=2,
        backbone="resnet18",
        learning_rate=0.001,
    )

    # Create trainer
    trainer = lightning.Trainer(max_epochs=10, deterministic=True)

    # Train the model
    trainer.fit(model, data)

    # Validate the model
    trainer.validate(model, data)

    # Test the model
    trainer.test(model, data)

    # Save the model
    trainer.save_checkpoint("data/06_models/model.ckpt")
    # trainer.save_checkpoint("s3://opened-gate/models/model.ckpt")


if __name__ == "__main__":
    main()
