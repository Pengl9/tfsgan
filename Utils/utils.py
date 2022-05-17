# Local modules
from ast import arg
from operator import mod
import os
import pickle
from tkinter.ttk import Label
from typing import Type

# 3rd party modules
import numpy as np
from tqdm import tqdm,trange
import torch,gc
import torch.utils.data as D
from torch.utils.tensorboard import SummaryWriter

# Self-written modules
from Utils.dataset import PenglDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autoencoder_trainer(model,dataloader,a_opt,s_opt,args,writer,Dlen):
    logger = trange(args.epochs,desc=f" Epoch: 0 | Loss: 0 ")
    max_a, max_d, max_f = torch.zeros(args.n_seg), torch.zeros(args.n_seg), 0

    for epoch in logger:

        running_loss = 0
        #num_loss = Dlen / args.bsize # 10 / 5
        model.train()

        for data, _ in dataloader:

            data = data.to(device)
            #print(data.shape)
            model.zero_grad()

            # Forward pass
            loss1, a, d = model(
                Idata = data,
                Label = "0",
                Type = args.type,
                Obj = "autoencoder"
            )
            loss2, feature_data = model(
                Idata = data,
                Label = "0",
                Type = args.type,
                Obj = "joint"
            )

            loss = loss1+loss2
            # Backward Pass
            loss.backward()
            running_loss += loss.item()

            # Update model parameters
            a_opt.step()
            s_opt.step()

            if torch.mean(a) + torch.mean(d) > torch.mean(max_a) + torch.mean(max_d):
                max_a, max_d = a,d
            if torch.mean(feature_data) > max_f:
                max_feature_data = feature_data
                max_f = torch.mean(feature_data)

        model.eval()
        loss = running_loss / Dlen
        logger.set_description(f" Epoch: {epoch} | Loss: {loss:.4f} ")
        if writer:
            writer.add_scalar(
                "AutoencoderNetwork/loss:",
                loss,
                epoch
            )
            writer.flush()

    dwt_data = torch.zeros((2,args.n_seg))
    dwt_data[0], dwt_data[1] = max_a, max_d
    np.save(f'./Data/{args.type}/dwt.npy',(dwt_data.unsqueeze(1)).unsqueeze(1))
    np.save(f'./Data/{args.type}/feature.npy',torch.mean(max_feature_data,0).cpu().detach().numpy())

def supervisor_trainer(model,dataloader,s_opt,g_opt,args,writer,Dlen):
    logger = trange(args.epochs,desc=f" Epoch: 0 | Loss: 0 ")
    for epoch in logger:

        running_loss = 0.0
        #num_loss = Dlen / args.bsize # 10 / 5
        model.train()

        for dfs,label in dataloader:
            dfs = np.transpose(dfs,(2,0,1)) # 1 x 484 x 200 -> 200 x 1 x 484
            dfs , label = dfs.to(device) , label.to(device)
            model.zero_grad()

            # Forward pass
            loss = model(
                Idata = dfs,
                Gdata = None,
                Label = label,
                Obj = "supervisor"
            )

            # Backward Pass
            loss.backward()
            running_loss += loss.item()

            # Update model parameters
            s_opt.step()
            g_opt.step()

        loss = running_loss / Dlen
        logger.set_description(f" Epoch: {epoch} | Loss: {loss:.4f} ")
        if writer:
            writer.add_scalar(
                "Supervisor/loss:",
                loss,
                epoch
            )
            writer.flush()

def joint_trainer(model,dataloader,a_opt,s_opt,g_opt,d_opt,args,writer,Dlen):
    # logger = trange(
    #     args.epochs,
    #     desc=f" Epoch: 0 | Eloss: 0 , Gloss: 0 , Dloss0 : 0 , Dloss1: 0 "
    # )
    logger = trange(
        args.epochs,
        desc=f" Epoch: 0 | Dloss-T : 0 , Dloss-F: 0 "
    )

    for epoch in logger:

        running_loss1 = 0.0
        running_loss2 = 0.0
        #num_loss = Dlen / args.bsize # 10 / 5
        model.train()

        for data, _ in dataloader:
            # 64 x 120 x 200
            data, fake_data = data.to(device), torch.rand((data.shape)).to(device)

            """ Real Data Training """
            model.zero_grad()
            Loss1, _ = model(
                Idata = data,
                Label = "0",
                Type = args.type,
                Obj = "joint"
            )
            Loss1.backward()

            """ Generator Training """
            model.zero_grad()
            Loss2 = model(
                Idata = fake_data,
                Label = "1",
                Type = args.type,
                Obj = "joint"
            )
            Loss2.backward()

            # Update model parameters
            a_opt.step()
            s_opt.step()
            g_opt.step()
            d_opt.step()

            running_loss1 += Loss1.item()
            running_loss2 += Loss2.item()
        
        lossT = running_loss1 / Dlen
        lossF = running_loss2 / Dlen
        logger.set_description(
            f" Epoch: {epoch} | DlossT: {lossT:.4f} , DlossF: {lossF:.4f} "
        )
        if writer:
            writer.add_scalar(
                'Joint/Discriminator_LossT:', 
                lossT, 
                epoch
            )
            writer.add_scalar(
                'Joint/Discriminator_LossF:', 
                lossF, 
                epoch
            )
            writer.flush()

def gan_trainer(model,args):

    # dataset = PenglGanDataset(args)
    # dataset_len = dataset.len # 10
    # dataloader = D.DataLoader(
    #     dataset=dataset,
    #     batch_size=args.bsize, #5
    #     shuffle=True
    # )

    dataset = PenglDataset(args)
    dataset_len = dataset.len
    dataloader = D.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model.to(args.device)

    # Initialize Optimizers
    a_opt = torch.optim.Adam(model.autoencoder.parameters(), lr=args.learning_rate)
    s_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
    g_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(f"./Output/{args.exp}"))

    print(" \nStart Autoencoder Network Training ")
    autoencoder_trainer(
        model = model,
        dataloader = dataloader,
        a_opt = a_opt,
        s_opt = s_opt,
        args = args,
        writer = writer,
        Dlen = dataset_len
    )
    print("Autoencoder training is completed and wavelet transform A-D features are saved \n")

    print(" \nStart Joint Network Training ")
    joint_trainer(
        model = model,
        dataloader = dataloader,
        a_opt = a_opt,
        s_opt = s_opt,
        g_opt = g_opt,
        d_opt = d_opt,
        args = args,
        writer = writer,
        Dlen = dataset_len
    )

    # Save model, args, and hyperparameters
    torch.save(args, f"{args.args_path}/args.pickle")
    torch.save(model.state_dict(), f"{args.model_path}/0405_model.pt")
    print(f"\nSaved at path: {args.model_path}")



def gan_generator(model,num,args):
    
    import scipy.io as scio

    if not os.path.exists(args.model_path):
        raise ValueError(" Model directory not found ")

    print("\n Generating data... ")
    model.load_state_dict(torch.load(f"{args.model_path}/0405_model.pt"))
    model.to(device)

    model.eval()
    with torch.no_grad():

        if args.type == 'wifi' or args.type == 'uwb':
            data = torch.rand(num,args.semsor_num,args.n_seg).to(device)
        elif args.type == 'rfid':
            data = torch.rand(num,args.n_seg).to(device)
        elif args.type == 'mmwave':
            data = torch.rand(num,args.n_seg,32,32).to(device)

        gdata = model(
            Idata = data,
            Label = "1",
            Type = args.type,
            Obj = "inference"
        )
    print(" Reliable timing data has been generated \n")


    if args.type == 'wifi':
        for i in range(num):
            latent_data = gdata[i]
            latent_data = latent_data.permute(1,0).cpu().numpy()
            scio.savemat(f'./Data/{args.type}/Gan_data/gan_{i+1}.mat', {'csi_latent_seg':latent_data})

    elif args.type == 'uwb':
        latent_data = gdata.cpu().numpy()
        np.save(f'./Data/{args.type}/Gan_data.npy',latent_data)

    elif args.type == 'rfid':
        latent_data = gdata.squeeze(1).cpu().numpy()
        np.save(f'./Data/{args.type}/Gan_data.npy',latent_data)

    elif args.type == 'mmwave':
        latent_data = gdata.permute(0,2,1).reshape(num,args.n_seg,32,32).cpu().numpy()
        np.save(f'./Data/{args.type}/Gan_data.npy',latent_data)

    return 
