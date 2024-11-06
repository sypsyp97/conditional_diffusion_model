import torch
from torchvision import datasets, transforms 
from diffusers import UNet2DModel, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Config
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
IMAGE_SIZE = 32
CHANNELS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR_WARMUP_STEPS = 500

# Set up model for image conditioning
model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=CHANNELS * 2,  # Double channels for concatenated condition
    out_channels=CHANNELS,
    layers_per_block=2,
    block_out_channels=(32, 64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(DEVICE)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",
    prediction_type="epsilon"
)

# Custom dataset for paired clean and noisy images
class NoisyMNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.mnist = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        
    def add_artifacts(self, img):
        patch = torch.zeros_like(img)
        patch[:, 10:20, 10:20] = 1  # Adding a white patch in the center
        return torch.clamp(img + patch, 0, 1)
        
    def __getitem__(self, idx):
        clean_img, label = self.mnist[idx]
        noisy_img = self.add_artifacts(clean_img)
        return clean_img, noisy_img
        
    def __len__(self):
        return len(self.mnist)

# Data transforms
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# Create dataset and dataloader
dataset = NoisyMNISTDataset("./data", train=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Optimization
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=LR_WARMUP_STEPS,
    num_training_steps=(len(dataloader) * NUM_EPOCHS),
)

def train_loop():
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (clean_images, noisy_images) in enumerate(dataloader):
            clean_images = clean_images.to(DEVICE)
            noisy_images = noisy_images.to(DEVICE)
            
            # Sample noise and timesteps
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                    (clean_images.shape[0],), device=DEVICE)
            
            # Add noise to clean images
            noisy = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Concatenate condition with noisy input
            model_input = torch.cat([noisy, noisy_images], dim=1)
            
            # Predict noise
            noise_pred = model(model_input, timesteps).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}")

def test_pipeline():
    model.eval()
    
    # Set number of inference steps
    num_inference_steps = 50
    noise_scheduler.set_timesteps(num_inference_steps)
    
    with torch.no_grad():
        # Get sample condition image and ground truth
        clean_images, noisy_images = next(iter(dataloader))
        clean_image = clean_images[0:1].to(DEVICE)
        condition = noisy_images[0:1].to(DEVICE)
        
        # Start from noise
        noisy_image = torch.randn(1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
        
        # Gradual denoising
        for t in noise_scheduler.timesteps:
            model_input = torch.cat([noisy_image, condition], dim=1)
            noise_pred = model(model_input, t).sample
            noisy_image = noise_scheduler.step(noise_pred, t, noisy_image).prev_sample
        
        # Display and save results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        ax1.imshow(clean_image[0,0].cpu(), cmap='gray')
        ax1.set_title('Ground Truth (Clean)')
        ax1.axis('off')
        
        # Noisy condition
        ax2.imshow(condition[0,0].cpu(), cmap='gray')
        ax2.set_title('Condition (Noisy)')
        ax2.axis('off')
        
        # Generated result
        ax3.imshow(noisy_image[0,0].cpu(), cmap='gray')
        ax3.set_title('Generated (Clean)')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig('restoration_comparison.png')
        plt.show()

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    try:
        train_loop()
        test_pipeline()
    except RuntimeError as e:
        print(f"Error during training: {e}")
    except ValueError as e:
        print(f"Value error: {e}")