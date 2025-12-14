import os
import numpy as np
import torch

from train import Generator # Generator class from train.py


home_dir = os.getcwd() # %userprofile%\Documents\Github\csen140-terrain-generation
data_dir = os.path.join(home_dir, 'Data')
model_dir = os.path.join(home_dir, 'attempt11') # Model would be where the output from train.py goes, but I'm saving outside of it during iteration
generator_path = os.path.join(model_dir, 'generator.pth')

if __name__ == "__main__":
    # Parameters n' stuff - sync with train.py
    Z_DIM = 100
    IMG_SIZE = 64
    NUM_CLASSES = 21

    # For de-normalization of data - sync with data_preparation.py
    MIN_HEIGHT = -500.0 # ~Death Valley
    MAX_HEIGHT = 9000.0 # ~Mt. Everest
    MAX_VEG = 100.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load the generator model
    generator = Generator(Z_DIM, IMG_SIZE, NUM_CLASSES).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    # Generate some samples
    for i in range(4):
        # Run inference
        z = torch.randn(1, Z_DIM).to(device)
        with torch.inference_mode():
            out_tensor = generator(z)
        
        # Convert the output tensor to numpy and denormalize
        out_data = out_tensor.squeeze(0).detach().cpu().numpy()

        height_denorm = (((out_data[0, :, :] + 1) / 2) * (MAX_HEIGHT - MIN_HEIGHT) + MIN_HEIGHT).astype(np.int16)
        vegetation_denorm = (((out_data[1, :, :] + 1) / 2) * MAX_VEG).astype(np.int16)
        cover_classed = np.argmax(out_data[2:, :, :], axis=0).astype(np.int16)

        output = np.dstack((height_denorm, cover_classed, vegetation_denorm))

        # Save product to disk
        out_path = os.path.join(model_dir, f'generated_{i}.npy')
        np.save(out_path, output)
