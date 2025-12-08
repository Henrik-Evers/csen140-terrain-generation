import zipfile
import os
import tifffile
import numpy as np


home_dir = os.getcwd() # %userprofile%\Documents\Github\csen140-terrain-generation
data_dir = os.path.join(home_dir, 'Data')
zips_dir = os.path.join(data_dir, 'zips')
tifs_dir = os.path.join(data_dir, 'tifs')
mega_dir = os.path.join(data_dir, 'mega')

# For normalization and de-normalization of data
MIN_HEIGHT = -500.0 # ~Death Valley
MAX_HEIGHT = 9000.0 # ~Mt. Everest
MAX_VEG = 100.0


# Extract zips
def extract_zips():
    # Create output folder if necessary
    if not os.path.exists(tifs_dir):
        os.mkdir(tifs_dir)

    # Grab all source data layers from /Data/zips
    for layer in [fpath for fpath in os.listdir(zips_dir) if os.path.isdir(os.path.join(zips_dir, fpath))]:
        # /Data/zips contains a separate layer for each zip
        # For each layer, create a corresponding file in the /Data/tifs
        layer_in_dir = os.path.join(zips_dir, layer)
        layer_out_dir = os.path.join(tifs_dir, layer)

        # If there is an empty output dir for this layer, remove it so we can make it
        # If it's not empty, os.rmdir will error and we will choose to not modify the folder
        if os.path.exists(layer_out_dir):
            try:
                os.rmdir(layer_out_dir)
            except:
                print('Existing dir was not modified: ', layer_out_dir)
                continue

        # Make an output folder in /Data/tifs
        os.mkdir(layer_out_dir)

        # Unzip each tile of the layer and put the tif in the output dir
        for tile in os.listdir(layer_in_dir):
            if tile.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(layer_in_dir, tile), 'r') as zf:
                    zf.extractall(layer_out_dir)
    
    return


# Create a mega-TIF
def combine_images():
    # Create output folder if necessary
    if not os.path.exists(mega_dir):
        os.mkdir(mega_dir)
    
    megas = []
    
    # Grab all unzipped data layers from /Data/tifs
    for layer in [fpath for fpath in os.listdir(tifs_dir) if os.path.isdir(os.path.join(tifs_dir, fpath))]:
        layer_in_dir = os.path.join(tifs_dir, layer)

        imgs = []
        stride = -1

        # Get all images from this layer
        for img in os.listdir(layer_in_dir):
            # All tif names end in '_{x-coord}_{y-coord}.tif', starting from (1, 1) and increasing to the South and East
            # They have the same prefix within a layer, so listdir will helpfully return the names pre-sorted in ascending order by x-coord then y-coord
            # Find where _ is used
            underscores = [i for i, char in enumerate(img) if char == '_']
            x = int(img[underscores[-2]+1:underscores[-1]])
            y = int(img[underscores[-1]+1:-4])
            #print('(', x, ',', y, ')')
            
            if y > stride:
                stride = y
            
            imgs.append(tifffile.imread(os.path.join(layer_in_dir, img)))
            #print(imgs[-1].shape)

        # Make the mega array
        rows = []
        for i in range(len(imgs)//stride):
            rows.append(np.hstack(imgs[i*stride:(i+1)*stride]))
        mega = np.vstack(rows)

        megas.append(mega)

        print(mega.shape)
        print(imgs[0][:5])
        print(mega[:5])

        mega_path = os.path.join(mega_dir, layer + '_mega.tif')
        tifffile.imwrite(mega_path, mega, photometric='minisblack')

    return megas


# Create a numpy array with a tuple with each layer for every pixel
def export_data(megas=None):
    imgs = []

    # Used passed layers if available
    if not megas is None:
        imgs = megas
    else:
        # Read layers
        for img in [fpath for fpath in os.listdir(mega_dir) if os.path.join(mega_dir, fpath).endswith('.tif')]:
            imgs.append(tifffile.imread(os.path.join(mega_dir, img)))
            # Print layer names
            print(img)
    
    # Combine the layers - each entry is now a tuple consisting of the data of each layer, in the order the layers are printed
    data = np.dstack(imgs)

    data_path = os.path.join(data_dir, 'data.npy')

    # Delete existing data if there is any
    if os.path.exists(data_path):
        os.remove(data_path)

    # Save the data to the disk
    np.save(data_path, data)

    return data


# Remove NaNs, normalize, etc.
def preprocessing(data=None):
    # Look for stored data if none passed
    if data is None:
        data_path = os.path.join(data_dir, 'data.npy')
        print('Loading data')
        data = np.load(data_path)

    # Split the channels
    print('Splitting channels')
    height_map = data[:, :, 0].astype(np.float16)
    veg_map = data[:, :, 2].astype(np.float16)
    cover_map = data[:, :, 1].astype(np.uint8)
    del data # my computer is begging for mercy, so i'll be reducing the memory load slightly

    # Process height
    height_map[height_map == -9999] = 0 # Set no data / sea level to be 0m
    height_map[height_map == 9998] = 0

    # Normalize to [-1, 1] via simple min-max
    # stddev norm. is probably not ideal since this is a heightmap? unsure.
    print('Norm height')
    height_map = 2 * ((height_map - MIN_HEIGHT) / (MAX_HEIGHT - MIN_HEIGHT)) - 1
    height_map = np.clip(height_map, -1.0, 1.0)

    # Process vegetation
    # Input is 0-100. Scale to [-1, 1]
    print('Norm vegetation')
    veg_map[veg_map >= 254] = 0 # Set no data / water to be 0%
    veg_map = (veg_map * 2) / MAX_VEG - 1
    veg_map = np.clip(veg_map, -1.0, 1.0)

    # Recombine data
    data_norm = np.dstack((height_map, veg_map, cover_map))
    del height_map, veg_map, cover_map

    data_norm_path = os.path.join(data_dir, 'data_norm.npy')

    # Delete existing data if there is any
    if os.path.exists(data_norm_path):
        os.remove(data_norm_path)

    # Save the data to the disk
    np.save(data_norm_path, data_norm)
    
    return data_norm


def prepare():
    # maybe include some logic for how to select each stage
    # maybe either (default) check what is needed, and skip stages that are already done, or (by flag) redo from scratch
    extract_zips()
    megas = combine_images()
    data = export_data(megas)
    data_norm = preprocessing(data)

    return 0

preprocessing()
