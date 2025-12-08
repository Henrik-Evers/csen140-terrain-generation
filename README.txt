Packages: numpy, tifffile, zipfile, pytorch, torchvision

Installed the NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-downloads) for GPU computation.

PyTorch packages were installed with this command to get the CUDA-enabled versions:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Used Godot 4.5-latest. Prior versions will not work.

To run the project:
You will need at least 60 GBs of storage space, and it is best to have at least 32 GBs of system memory, especially if computing on the CPU (not recommended).
1. Download the project, dependencies, and dataset. Within the project directory, place the dataset in Data/zips, with the dataset folders inside (where they have a folder for each layer, containing that layer's zips). Also create an empty Model directory in the project directory.
2. Run prepare() in data_preparation.py. You can instead call the functions one at a time in the script body, but prepare() will do everything from scratch.
3. Run train.py. If you have not installed both the NVIDIA CUDA toolkit and the CUDA-enabled PyTorch packages, it will give a warning and run on the CPU. If you have, you should see "cuda" in the console output.
4. Run inference.py. You should see some generated_number.npy files in the Model directory.
5. Open the Godot project and change TERRAIN_PATH in terrain.gd to point to the desired generated_number.npy. With the uploaded file structure, res://../ points to the project directory. Press F5 to run the project.
5b. If you're curious about seeing the visualization on some actual data, there are some commented lines to do that. Make sure you include the lines that enforces a maximum size for dim1 and dim2, otherwise it might try to load the entire dataset. Press F8 to close the running project, or Alt+F4 to close Godot if necessary.

Alternatively, if there is any generator.pth in the submission only a few MBs of storage will be required, and you can use these simplified instructions:
You will just need to install numpy, pytorch and torchvision (CUDA-enabled or regular), and Godot. You will need the Model directory, the godot-terrain-viz directory, and inference.py.
Skip to step 4 of the above.
