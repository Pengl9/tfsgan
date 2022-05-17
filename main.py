# Local Modules
import os
import time
from matplotlib.pyplot import get

# 3rd-Party Modules
import numpy as np
import torch

# Self-Written Modules
from Utils.io_utils import parse_args
#from Pre_dfs import dfs_preprocess
from Utils.backbone import Penglbackbone
from Utils.utils import gan_generator,gan_trainer

def main(args):

    ## Data Directory
    data_dir = os.path.abspath(f"./Data/{args.type}")
    if not os.path.exists(data_dir):
        raise ValueError(f"Data file not found at {data_dir}.")

    ## Output Directory
    out_dir = os.path.abspath(f"./Output/{args.type}/{args.exp}/")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)

    print(f"\nData {args.type} directory:\t\t{data_dir}")
    print(f"Output directory:\t{out_dir}\n")

    # Initialize random
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Initialize CUDA
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Choose Device | {device} \n")
    
    ## Load and Preprocess data for model
    print(f"  Processed {args.type} data: b_size x {args.semsor_num} x {args.n_seg} (Batech_size x Features x MaxSeqLen) \n")

    # Initialize and Run model
    start = time.perf_counter()

    # Model Loading
    model = Penglbackbone(args)
    
    # Start training or testing
    if args.is_train == True:
        gan_trainer(
            model = model,
            args = args)
    else:
        gan_generator(
            model = model,
            num = 300,
            args = args)
    
    end = time.perf_counter()

    #print(f"Generated data preview:\n{generated_data[:2, -10:, :2]}\n")
    #print(f"Generated data: {generated_data.shape} (Idx x MaxSeqLen x Features)\n")
    print(f" Model Runtime: {end - start:.4f} s\n")


if __name__ == '__main__':

    args = parse_args('rfid')
    main(args)