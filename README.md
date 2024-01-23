# cycleCDR

To setup the environment, install conda and run (Must run on servers with multiple GPUs):

```bash
conda create --name <your_env_name> --file requirements.txt

torchrun --nproc_per_node=2 --master-port=29501  cycleCDR/linc_main.py

mpirun -n 2 python cycleCDR/sciplex_train.py

mpirun -n 2 python cycleCDR/sciplex_test.py
```

+ `cycleCDR`: contains the code for the model, the data, and the training loop.

+ `preprocessing`: Scripts for processing the data.

+ `configs`: Configuration file for hyperparameters.

+ `plot_script`: Code for creating images in the paper.

+ `datasets`: Directory where data is stored.

+ `results`: This directory needs to be created by oneself. It is used to store the results of the experiment.

    + `modules`: Model storage directory.

    + `plot`: Training loss curve.

    + `plot_data`: Data of the training process.

