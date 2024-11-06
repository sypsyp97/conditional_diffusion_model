import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration parameters for the diffusion model."""
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 1e-4
    image_size: int = 32
    channels: int = 1
    lr_warmup_steps: int = 700
    num_train_timesteps: int = 1000
    num_inference_steps: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class NoisyMNISTDataset(Dataset):
    """
    Custom dataset for paired clean and noisy MNIST images.
    Adds artifacts to create noisy versions of the images.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[transforms.Compose] = None):
        """
        Args:
            root: Directory for storing the MNIST dataset
            train: If True, creates dataset from training data
            transform: Optional transform to be applied on images
        """
        self.mnist = datasets.MNIST(root=root, train=train, download=True, transform=transform)

    def add_artifacts(self, img: torch.Tensor) -> torch.Tensor:
        """Add random artificial artifacts to the image."""
        noisy_img = img.clone()
        # Random number of patches (3 to 5)
        num_patches = torch.randint(3, 6, (1,)).item()

        for _ in range(num_patches):
            # Random patch size between 5 and 8 pixels
            patch_h = torch.randint(5, 11, (1,)).item()
            patch_w = torch.randint(5, 11, (1,)).item()

            # Random position ensuring patch fits within image
            max_h = img.shape[1] - patch_h
            max_w = img.shape[2] - patch_w
            h_start = torch.randint(0, max_h + 1, (1,)).item()
            w_start = torch.randint(0, max_w + 1, (1,)).item()

            # Add white patch
            noisy_img[:, h_start:h_start+patch_h, w_start:w_start+patch_w] = 1.0

        return torch.clamp(noisy_img, 0, 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_img, _ = self.mnist[idx]
        noisy_img = self.add_artifacts(clean_img)
        return clean_img, noisy_img

    def __len__(self) -> int:
        return len(self.mnist)

class DiffusionCleaner:
    """
    Implements the conditional diffusion model for image restoration.
    Handles model initialization, training, and inference.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the diffusion model with given configuration.

        Args:
            config: ModelConfig instance containing model parameters
        """
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model components
        self.model = self._setup_model()
        self.noise_scheduler = self._setup_scheduler()
        self.transform = self._setup_transforms()
        self._setup_data()
        self.optimizer, self.lr_scheduler = self._setup_optimization()
        self.best_val_loss = float('inf')
        self.checkpoint_dir = './checkpoints'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        logger.info(f"Model initialized on device: {self.device}")

    def _setup_model(self):
        """Initialize and configure the UNet model."""
        model = UNet2DModel(
            sample_size=self.config.image_size,
            in_channels=self.config.channels * 2,
            out_channels=self.config.channels,
            layers_per_block=2,
            block_out_channels=(32, 64, 128, 256),
            down_block_types=("DownBlock2D",) * 4,
            up_block_types=("UpBlock2D",) * 4,
        ).to(self.device)
        return model

    def _setup_scheduler(self):
        """Initialize the noise scheduler."""
        return DDIMScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

    def _setup_transforms(self) -> transforms.Compose:
        """Set up image transformations."""
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
        ])

    def _setup_data(self) -> None:
        """Initialize dataset and dataloader."""
        # Define the transform
        transform = self._setup_transforms()

        # Load the MNIST dataset
        full_dataset = NoisyMNISTDataset("./data", train=True, transform=transform)

        # Split into train and validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # Test dataset
        test_dataset = NoisyMNISTDataset("./data", train=False, transform=transform)

        # Dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )

    def _setup_optimization(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Set up optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.config.num_epochs),
        )
        return optimizer, lr_scheduler

    def train(self):
        """Execute the training loop."""
        self.model.train()

        for epoch in range(self.config.num_epochs):
            self._train_epoch(epoch)
            self.validate(epoch)

    def _train_epoch(self, epoch: int):
        """Train the model for one epoch."""
        for batch_idx, (clean_images, noisy_images) in enumerate(self.train_dataloader):
            loss = self._train_step(clean_images, noisy_images)

            if batch_idx % 100 == 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss:.4f} | LR: {lr:.6f}")

    def _train_step(self, clean_images: torch.Tensor, condition_images: torch.Tensor) -> float:
        """Execute a single training step."""
        clean_images = clean_images.to(self.device)
        condition_images = condition_images.to(self.device)

        # Sample noise and timesteps
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],), device=self.device
        )

        # Add noise to clean images
        noisy_samples = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        model_input = torch.cat([noisy_samples, condition_images], dim=1)

        # Predict and optimize
        noise_pred = self.model(model_input, timesteps).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return loss.item()

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (clean_images, noisy_images) in enumerate(self.val_dataloader):
                loss = self._validation_step(clean_images, noisy_images)
                val_loss += loss
        val_loss /= len(self.val_dataloader)

        logger.info(f"Epoch {epoch} | Validation Loss: {val_loss:.4f}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_model(epoch)
            logger.info(f"New best model saved with validation loss {val_loss:.4f}")
        self.model.train()

    def _validation_step(self, clean_images: torch.Tensor, condition_images: torch.Tensor) -> float:
        """Execute a single validation step."""
        clean_images = clean_images.to(self.device)
        condition_images = condition_images.to(self.device)

        # Sample noise and timesteps
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],), device=self.device
        )

        # Add noise to clean images
        noisy_samples = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        model_input = torch.cat([noisy_samples, condition_images], dim=1)

        # Predict
        noise_pred = self.model(model_input, timesteps).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return loss.item()

    def save_model(self, epoch):
        """Save the model checkpoint."""
        save_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, save_path)

    def load_model(self):
        """Load the best model from checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {self.best_val_loss:.4f}")
        else:
            logger.warning("No checkpoint found at '{}'".format(checkpoint_path))

    def test(self, num_samples: int = 4):
        """
        Run inference and visualize results for multiple samples.

        Args:
            num_samples: Number of samples to generate and visualize
        """
        self.load_model()
        self.model.eval()
        self.noise_scheduler.set_timesteps(self.config.num_inference_steps)

        with torch.no_grad():
            # Get sample images from the test dataloader
            clean_images, noisy_images = next(iter(self.test_dataloader))
            clean_images = clean_images[:num_samples].to(self.device)
            conditions = noisy_images[:num_samples].to(self.device)

            # Generate denoised images
            results = []
            for condition in conditions:
                # Add batch dimension
                condition = condition.unsqueeze(0)
                result = self._denoise_image(condition)
                results.append(result)

            # Concatenate all results
            results = torch.cat(results, dim=0)

            # Visualize results
            self._plot_results(
                clean_images,
                conditions,
                results,
                num_samples=num_samples,
                save_path=f'restoration_results_{num_samples}samples.png'
            )

    def _denoise_image(self, condition: torch.Tensor) -> torch.Tensor:
        """Denoise a single image using the diffusion model."""
        noisy_image = torch.randn(
            1, self.config.channels,
            self.config.image_size,
            self.config.image_size,
            device=self.device
        )

        for t in self.noise_scheduler.timesteps:
            model_input = torch.cat([noisy_image, condition], dim=1)
            noise_pred = self.model(model_input, t).sample
            noisy_image = self.noise_scheduler.step(noise_pred, t, noisy_image).prev_sample

        return noisy_image

    def _plot_results(self, clean_images: torch.Tensor, conditions: torch.Tensor, results: torch.Tensor,
                      num_samples: int = 8, save_path: str = 'restoration_results.png'):
        """
        Plot multiple results in a grid layout and save as a single image.

        Args:
            clean_images: Original clean images
            conditions: Noisy condition images
            results: Generated restoration results
            num_samples: Number of samples to plot
            save_path: Path to save the result image
        """
        # Ensure we don't try to plot more than we have
        num_samples = min(num_samples, clean_images.shape[0])

        # Create a grid of plots: num_samples rows x 3 columns
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

        # Set column titles
        titles = ['Ground Truth (Clean)', 'Condition (Noisy)', 'Generated (Clean)']
        for ax, title in zip(axes[0], titles):
            ax.set_title(title, fontsize=12, pad=10)

        # Plot each set of images
        for idx in range(num_samples):
            images = [
                clean_images[idx:idx+1],
                conditions[idx:idx+1],
                results[idx:idx+1]
            ]

            for col, img in enumerate(images):
                axes[idx, col].imshow(img[0, 0].cpu(), cmap='gray')
                axes[idx, col].axis('off')

                # Add PSNR and SSIM metrics for generated images
                if col == 2:  # Generated image column
                    psnr = self._calculate_psnr(clean_images[idx:idx+1], img)
                    ssim = self._calculate_ssim(clean_images[idx:idx+1], img)
                    metrics_text = f'PSNR: {psnr:.2f}\nSSIM: {ssim:.3f}'
                    axes[idx, col].text(1.05, 0.5, metrics_text,
                                      transform=axes[idx, col].transAxes,
                                      verticalalignment='center')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.show()

    def _calculate_psnr(self, clean_img: torch.Tensor, generated_img: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio between two images."""
        mse = torch.mean((clean_img.cpu() - generated_img.cpu()) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    def _calculate_ssim(self, clean_img: torch.Tensor, generated_img: torch.Tensor) -> float:
        """
        Calculate Structural Similarity Index between two images.
        This is a simplified version of SSIM.
        """
        clean_img = clean_img.cpu()
        generated_img = generated_img.cpu()

        # Constants for stability
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2

        # Calculate means
        mu1 = torch.mean(clean_img)
        mu2 = torch.mean(generated_img)

        # Calculate variances and covariance
        sigma1_sq = torch.var(clean_img)
        sigma2_sq = torch.var(generated_img)
        sigma12 = torch.mean((clean_img - mu1) * (generated_img - mu2))

        # Calculate SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim.item()

def main():
    """Main execution function."""
    torch.manual_seed(1234)    # Set random seed for reproducibility
    try:
        torchvision.disable_beta_transforms_warning()
        config = ModelConfig()
        model = DiffusionCleaner(config)
        model.train()
        model.test(num_samples=4)
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
