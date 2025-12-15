import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as utils
import torchvision.utils as vutils
import matplotlib.pyplot as plt

home_dir = os.getcwd() # %userprofile%\Documents\Github\csen140-terrain-generation
data_dir = os.path.join(home_dir, 'Data')
data_path = os.path.join(data_dir, 'data_norm.npy')
model_dir = os.path.join(home_dir, 'Model')


# Class to provide a view to the data without loading the entire thing at once
class TerrainDataset(Dataset):
    def __init__(self, tile_size, stride, num_classes, mode='train', val_split=0.2, pull=0.01, noise=0.01):
        self.tile_size = tile_size # each section of terrain that we look at
        self.stride = stride # how much to shift the view each time
        self.num_classes = num_classes # number of classes in land cover
        self.data = None
        self.pull = pull
        self.noise = noise

        if not os.path.exists(data_path):
            print('data_path does not exist: ', data_path)
            exit()

        # I've learned my lesson, I promise not to keep loading the entire dataset into memory
        data = np.load(data_path, mmap_mode='r')

        # Get indices for each of the tiles
        indices = []
        for x in range(0, data.shape[0] - self.tile_size + 1, self.stride):
            for y in range(0, data.shape[1] - self.tile_size + 1, self.stride):
                indices.append((x, y))
        
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - val_split))

        if mode == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        print(f'Dataset ({mode}) initialized with {len(self.indices)} tiles')

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        x, y = self.indices[idx]

        # Load just one tile from memory
        if self.data is None:
            self.data = np.load(data_path, mmap_mode='r')
        tile = np.array(self.data[x : x + self.tile_size, y : y + self.tile_size, :]) # np.array() force loads the tile into memory so we can slice faster
        
        height_tensor = torch.from_numpy(tile[:, :, 0]).unsqueeze(0)
        veg_tensor = torch.from_numpy(tile[:, :, 1]).unsqueeze(0)
        cover = torch.from_numpy(tile[:, :, 2]).long()
        cover_onehot = F.one_hot(cover, num_classes=self.num_classes).permute(2, 0, 1).float()
        # change the one-hot probabilities to be ~0.1 for not and ~0.9 for is, then small noise? might help discrimin on this layer

        # Soften the one-hot probabilities and apply some noise, so that the discriminator doesn't have the easy out of 1.0 or 0.0 = real, anything else = fake.
        self.pull = 0.01 # average that probabilities are from 1.0/0.0
        self.noise = 0.01 # max noise offset, should be <= PULL
        cover_soft = cover_onehot * (1 - 2 * self.pull) + self.pull + torch.rand_like(cover_onehot) * 2 * self.noise - self.noise

        final_tensor = torch.cat([height_tensor, veg_tensor, cover_soft], dim=0)
        return final_tensor.half() # keep in FP16 here because data is FP16. also my RAM is crying and buying more is too expensive


# Much of the below is adapted from https://www.cnblogs.com/linzzz98/articles/13656162.html
# It has been significantly modified from the source, but it was a valuable reference while creating the initial structure.
class Generator(nn.Module):
    def __init__(self, z_dim, img_size, num_classes, kernel_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Dimensions for the shared trunk
        self.init_size = img_size // 16 # 64 // 16 = 4
        
        # Shared trunk
        self.l1 = nn.Sequential(nn.Linear(z_dim, 512 * self.init_size ** 2))
        
        # Use a series of upsamples to avoid needing a massive series of fully-connected layers
        # my computer is only so good
        CONV_KERNEL_SIZE = kernel_size
        PADDING = (CONV_KERNEL_SIZE - 1) // 2

        def generator_block(in_filters, out_filters, batch=True):
            block = [nn.Upsample(scale_factor=2, mode='bilinear'),
                     nn.Conv2d(in_filters, out_filters, kernel_size=CONV_KERNEL_SIZE, stride=1, padding=PADDING, padding_mode='reflect')]
            if batch:
                block.append(nn.BatchNorm2d(out_filters))
            else:
                block.append(nn.InstanceNorm2d(out_filters, affine=True))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),

            *generator_block(512, 256), # 4x4  ->  8x8
            *generator_block(256, 128), # 8x8  -> 16x16
            *generator_block(128, 64), # 16x16 -> 32x32
            *generator_block(64, 32),  # 32x32 -> 64x64
        )
        
        # Continuous head for height and vegetation
        self.head_continuous = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # Discrete head for probabilities
        self.head_discrete = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        
        features = self.conv_blocks(out)
        continuous = self.head_continuous(features) # continuous features
        discrete_logits = self.head_discrete(features) # 'discrete' features
        
        # Use Gumbel-Softmax to produce a result close to one-hot, but not exactly, so we can still learn from our prediciton beyond simple right/wrong
        discrete = F.gumbel_softmax(discrete_logits, tau=1, hard=False, dim=1)
        
        img = torch.cat([continuous, discrete], dim=1)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)), 
                     nn.LeakyReLU(0.2, inplace=True), 
                     nn.Dropout2d(0.05)]
            if bn == True:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        # Basically the reverse of how we do the generator
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The Output Layer
        # Downsample to 1 value
        self.adv_layer = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() # needed for BCE/MSE, don't use for Hinge
        )

    def forward(self, img):
        out = self.model(img)
        out = self.adv_layer(out)
        return out.view(out.shape[0], -1)


# To set small random starting weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# A noise-based regularization offset
def get_noise_annealed(epoch, max_epochs, start_noise=0.1):
    # Linearly decay noise from start_noise to 0.0 over the course of training
    # Applied to the data passed to the discriminator to reduce overfitting and improve generalization
    if epoch < max_epochs/2:
        return start_noise * (1 - 2 * epoch / max_epochs)
    else: # no noise for the second half, so the discriminator will challenge the generator better
        return 0


# Fix random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
np.random.seed(42)


def train(config):
    # Parameters n' stuff
    Z_DIM = 100
    IMG_SIZE = 64
    STRIDE = 64
    NUM_CLASSES = 21
    TOTAL_CHANNELS = 1 + 1 + NUM_CLASSES # 23
    
    RUN_NAME = config.get('name', 'default_run')
    BATCH_SIZE = config.get('batch_size', 256)
    EPOCHS = config.get('epochs', 8)
    BATCHES_PER_EPOCH = config.get('batches_per_epoch', 1024) # i am impatient
    LR_G = config.get('lr_g', 0.0002)
    LR_D = config.get('lr_d', 0.00015)
    BETAS = config.get('betas', (0.5, 0.999))
    GEN_CONV_KERNEL_SIZE = config.get('kernel_size', 3)
    PULL = config.get('pull', 0.01)
    NOISE = config.get('noise', 0.01)
    START_NOISE = config.get('start_noise', 0.1)

    current_model_dir = os.path.join(model_dir, RUN_NAME)
    os.makedirs(current_model_dir, exist_ok=True)

    print(f"Starting Run: {RUN_NAME}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize data stuff
    dataset = TerrainDataset(tile_size=IMG_SIZE, stride=STRIDE, num_classes=NUM_CLASSES, mode='train', pull=PULL, noise=NOISE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    val_dataset = TerrainDataset(tile_size=IMG_SIZE, stride=STRIDE, num_classes=NUM_CLASSES, mode='validate', pull=PULL, noise=NOISE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    # Initialize the models
    generator = Generator(Z_DIM, IMG_SIZE, NUM_CLASSES, GEN_CONV_KERNEL_SIZE).to(device)
    discriminator = Discriminator(TOTAL_CHANNELS).to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Initialize the optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=BETAS)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=BETAS)

    # |  ||
    # || |_
    criterion = nn.MSELoss()

    # Random noise
    fixed_z = torch.randn(BATCH_SIZE, Z_DIM).to(device) # used for evaluation - consistent across epochs

    # Track losses across epochs
    losses = [[], []]

    # Train!
    for epoch in range(EPOCHS):
        current_noise = get_noise_annealed(epoch, EPOCHS, START_NOISE)
        d_loss_epoch = 0
        g_loss_epoch = 0

        for i, data in enumerate(dataloader):
            # Clear previous gradient
            discriminator.zero_grad()

            # Train with a batch of real data
            real_data = data.to(device).float() # cast to FP32! model is in FP32 for better stability hopefully, but data is FP16
            noise = torch.randn_like(real_data) * current_noise
            real_data += noise - noise/2 # noise can be + or -
            b_size = real_data.size(0) # in case we don't get a full batch
            label = torch.full((b_size,), 1.0, dtype=torch.float, device=device) # smooth label so overfitting less likely

            output = discriminator(real_data).view(-1)
            d_loss_real = criterion(output, label)
            # d_loss_real = torch.mean(F.relu(1.0 - output))
            d_loss_real.backward()

            # Train with a batch of fake data
            noise = torch.randn(b_size, Z_DIM, device=device)
            fake_data = generator(noise)
            noise = torch.randn_like(fake_data) * current_noise
            noisy_fake_data = fake_data + noise - noise/2 # need to preserve original fake_data for generator training
            label.fill_(0.0) # fake labels are all 0

            output = discriminator(noisy_fake_data.detach()).view(-1)
            d_loss_fake = criterion(output, label)
            # d_loss_fake = torch.mean(F.relu(1.0 + output))
            d_loss_fake.backward()

            # Iterate the discriminator optimizer
            d_loss = d_loss_real + d_loss_fake
            optimizer_D.step()

            # Clear previous gradient
            generator.zero_grad()

            # Try to trick the discriminator
            label.fill_(1.0)
            
            output = discriminator(fake_data).view(-1)
            g_loss = criterion(output, label)
            # g_loss = -torch.mean(output)
            g_loss.backward()

            # Iterate the generator optimizer
            optimizer_G.step()

            # Update our tracking
            d_loss_epoch += d_loss.item()
            g_loss_epoch += g_loss.item()

            # Post stats to the console every so often. 256 seems like a fine number
            if i % 256 == 0:
                print(f"[{epoch}/{EPOCHS}][{i}/{BATCHES_PER_EPOCH}] "
                    f"Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}")
            
            # Enforce BATCHES_PER_EPOCH
            if i >= BATCHES_PER_EPOCH:
                losses[0].append(d_loss_epoch / BATCHES_PER_EPOCH)
                losses[1].append(g_loss_epoch / BATCHES_PER_EPOCH)
                break
        
        # Perform validation
        with torch.no_grad():
            # Run the discriminator on a piece of validation data
            val_data = next(iter(val_dataloader)).to(device).float()
            val_score = discriminator(val_data).mean().item()

            print(f"Validation Loss: {val_score:.4f}")

        # Save the model from the latest epoch
        torch.save(generator.state_dict(), os.path.join(current_model_dir, f'generator{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(current_model_dir, f'discriminator{epoch}.pth'))
        
        # Save an image for this epoch to visualize progress
        with torch.no_grad():
            fake = generator(fixed_z).detach().cpu()
            
            # Extract the height channel for quick visualization
            # Normalize it to [0, 1] (from [-1, 1]) for image saving
            height_map = fake[:, 0, :, :].unsqueeze(1)
            height_map = (height_map + 1) / 2 
            
            vutils.save_image(height_map, os.path.join(current_model_dir, f"{epoch}.png"), normalize=False)
    
    plt.figure(); plt.plot(losses[0], label='discriminator'); plt.plot(losses[1], label='generator'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Generator and Discriminator Losses'); plt.legend()
    plt.savefig(os.path.join(current_model_dir, 'losses.png'))
    plt.close()

if __name__ == "__main__":
    default_config = {
        'name': 'manual_run',
    }
    train(default_config)
