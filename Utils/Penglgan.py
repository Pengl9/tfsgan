import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class EmbeddingNetwork(nn.Module):
    def __init__(self,args):
        super(EmbeddingNetwork,self).__init__()
        self.n_feature = args.n_feature # 121*4
        self.new_feature = args.new_feature # 128
        self.n_head = args.n_head # 4

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_feature,nhead=self.n_head)
        encode_norm = nn.LayerNorm(self.n_feature)
        self.emb_transformer = nn.TransformerEncoder(encoder_layer,6,encode_norm)
        self.conv = nn.Conv1d(self.n_feature,self.n_feature,1)
        self.init()

        self.emb_linear = nn.Linear(self.n_feature,self.new_feature) # 200 x bsize x 484 -> 200 x bsize x 128

    def init(self):
        '''
            Fourier weights initialization
        '''
        basis = torch.tensor([math.pi * 2 * j / self.window_size for j in range(self.window_size)])
        weight = torch.zeros((self.k * 2, self.window_size))
        for i in range(self.k * 2):
            f = int(i / 2) + 1
            if i % 2 == 0:
                weight[i] = torch.cos(f * basis)
            else:
                weight[i] = torch.sin(-f * basis)
        self.conv.weight = torch.nn.Parameter( weight.unsqueeze(1), requires_grad=True)


    def forward(self,dfs):
        """ Forward pass for embedding features from original space into latent space
        Args:
            - dfs: Input initial data of Doppler spectrum [200 x bsize x 121*4] N_seg x Amoun x Feature
        Returns:
            - Dfs: Latent space embeddings [200 x bsize x 128] N_seg x Amoun x New_feature
        """
        Edfs = self.emb_transformer(dfs)
        Enewdfs = self.emb_linear(Edfs)

        return Enewdfs

class RecoveryNetwork(nn.Module):
    def __init__(self,args):
        super(RecoveryNetwork,self).__init__()
        self.n_feature = args.new_feature # 128
        self.new_feature = args.n_feature # 121*4
        self.n_head = args.n_head # 4

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_feature,nhead=self.n_head)
        encode_norm = nn.LayerNorm(self.n_feature)
        self.rec_transformer = nn.TransformerEncoder(encoder_layer,6,encode_norm)

        self.rec_linear = nn.Linear(self.n_feature,self.new_feature) # 200 x bsize x 128 -> 200 x bsize x 484

    def forward(self,Edfs):
        """ Forward pass for the recovering features from latent space to original space
        Args:
            - Edfs: Input embedding data of Doppler spectrum [200 x bsize x 128] N_seg x Amoun x Feature
        Returns:
            - Rdfs: Latent space embeddings [200 x bsize x 121*4] N_seg x Amoun x New_feature
        """
        Rdfs = self.rec_transformer(Edfs)
        Rnewdfs = self.rec_linear(Rdfs)

        return Rnewdfs

class SupervisorNetwork(nn.Module):
    def __init__(self,args):
        super(SupervisorNetwork,self).__init__()
        self.n_feature = args.new_feature
        self.n_head = args.n_head

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_feature,nhead=self.n_head)
        encode_norm = nn.LayerNorm(self.n_feature)
        self.sup_transformer = nn.TransformerEncoder(encoder_layer,6,encode_norm)

        self.sup_linear = nn.Linear(self.n_feature,self.n_feature)


    def forward(self,dfs):
        """Forward pass for the supervisor for predicting next step
        Args:
            - Dfs: Latent representation (200 x bsize x 128)
        Returns:
            - Sdfs: Adjusted data (200 x bsize x 128)
        """
        Stdfs = self.sup_transformer(dfs)
        Sdfs = self.sup_linear(Stdfs)

        return Sdfs

class GeneratorNetwork(nn.Module):
    def __init__(self,args):
        super(GeneratorNetwork,self).__init__()
        self.n_feature = args.n_feature
        self.new_feature = args.new_feature
        self.n_head = args.n_head

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_feature,nhead=self.n_head)
        encode_norm = nn.LayerNorm(self.n_feature)
        self.gen_transformer = nn.TransformerEncoder(encoder_layer,6,encode_norm)

        self.gen_linear = nn.Linear(self.n_feature,self.new_feature) # 200 x bsize x 484 -> 200 x bsize x 128

    def forward(self,dfs):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - Edfs: latent representation [200 x bsize x 484] N_seg x Amoun x New_feature
        Returns:
            - Gdfs: predicted logits [200 x bsize x 128] N_seg x Amoun x New_feature
        """
        Gtdfs = self.gen_transformer(dfs)
        Gdfs = self.gen_linear(Gtdfs)

        return Gdfs

class DiscriminatorNetwork(nn.Module):
    def __init__(self,args):
        super(DiscriminatorNetwork,self).__init__()
        self.n_feature = args.new_feature # 128
        self.new_feature = 2 # 1
        self.n_head = args.n_head # 4

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.n_feature,nhead=self.n_head)
        encode_norm = nn.LayerNorm(self.n_feature)
        self.dis_transformer = nn.TransformerEncoder(encoder_layer,6,encode_norm)

        self.dis_linear = nn.Linear(self.n_feature,self.new_feature) # 200 x bsize x 128 -> [-1,1]
        self.drop = nn.Dropout(0.2) # Prevent overfitting

    def forward(self,Gdfs):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - Edfs: latent representation [200 x bsize x 128] N_seg x Amoun x New_feature
        Returns:
            - logits: predicted logits [200 x bsize x 2] N_seg x Amoun 【Predict】
        """
        Ddfs = self.dis_transformer(Gdfs)
        Plogits = self.dis_linear(self.drop(Ddfs))#.squeeze(-1)

        return Plogits

class PenglGan(nn.Module):
    def __init__(self,args):
        super(PenglGan,self).__init__()
        self.device = args.device

        self.embedder = EmbeddingNetwork(args)
        self.recovery = RecoveryNetwork(args)
        self.generator = GeneratorNetwork(args)
        self.supervisor = SupervisorNetwork(args)
        self.discriminator = DiscriminatorNetwork(args)

    def _autoencoder_forward(self,dfs):
        """The embedding network forward pass and the embedder network loss
        Args:
            - Dfs: The original input features [200 x seg x 484]
        Returns:
            - Eloss: The reconstruction loss
            - Rdfs: The reconstructed features
        """
        # Forward Pass
        Edfs = self.embedder(dfs)
        Rdfs = self.recovery(Edfs)

        # Supervisor loss
        Cosine = nn.CosineSimilarity(dim=2,eps=1e-6)

        Sdfs = self.supervisor(Edfs) # 200 x seg x 484

        Sloss1 = F.pairwise_distance(Edfs.permute(1,0,2),Sdfs.permute(1,0,2),3) # bsize x 200 x 128 -> bsize x 200
        Sloss2 = Cosine(Edfs.permute(1,0,2),Sdfs.permute(1,0,2))

        if torch.mean(Sloss1) > 2:
            Sloss = torch.mean(Sloss1) - torch.mean(Sloss2) # Average vector distance
        else:
            Sloss = torch.mean(Sloss1) + torch.mean(Sloss2) # Average vector distance

        # Reconstruction loss 
        Rloss1 = F.pairwise_distance(Rdfs.permute(1,0,2),dfs.permute(1,0,2),3) # bsize x 200 x 128 -> bsize x 200
        Rloss2 = Cosine(Rdfs.permute(1,0,2),dfs.permute(1,0,2))

        if torch.mean(Rloss1) > 2:
            Rloss = torch.mean(Rloss1) - torch.mean(Rloss2) # Average vector distance
        else:
            Rloss = torch.mean(Rloss1) + torch.mean(Rloss2) # Average vector distance

        Aloss = Rloss + Sloss

        return Aloss
    
    def _supervisor_forward(self,dfs):
        """The supervisor training forward pass
        Args:
            - Dfs: The original input features
        Returns:
            - Sloss: The supervisor loss
        """
        # Supervisor Forward Pass
        Edfs = self.embedder(dfs)
        Sdfs = self.supervisor(Edfs)
        Rdfs = self.recovery(Sdfs)

        """
        Logic of loss needs to be optimized [20220225]
        """

        # Supervised loss
        Sloss1 = F.mse_loss(Sdfs,Edfs)
        Sloss2 = F.mse_loss(Rdfs,dfs)
        Sloss = Sloss1*50 + Sloss2*50

        return Sloss

    def _discriminator_forward(self,Idfs,label):
        """The discriminator training forward pass
        Args:
            - Idfs: Data to be Identified 200 x 1 x 484
            - label: Data Category [T/F]
        Returns:
            - Dloss: The supervisor loss
        """
        # Real & fake
        Gdfs = self.generator(Idfs)
        Sdfs = self.supervisor(Gdfs)

        # Forward Pass
        Didfs= self.discriminator(Gdfs)
        Dsdfs = self.discriminator(Sdfs)

        if label == "0":
            Diloss = F.binary_cross_entropy_with_logits(Didfs, torch.zeros_like(Didfs))
            Dsloss = F.binary_cross_entropy_with_logits(Dsdfs, torch.zeros_like(Dsdfs))
        elif label == "1":
            Diloss = F.binary_cross_entropy_with_logits(Didfs, torch.ones_like(Didfs))
            Dsloss = F.binary_cross_entropy_with_logits(Dsdfs, torch.ones_like(Dsdfs))
        else: raise ValueError(" 'label' should be either '0' or '1' ")

        Dloss = Diloss + Dsloss

        return Dloss

    def _generator_forward(self,dfs):
        """The supervisor training forward pass
        Args:
            - Dfs: The input features
            - Mapping: Identification basis
        Returns:
            - Dloss: The supervisor loss
        """
        # Generater forward pass
        Gdfs = self.generator(dfs) # 200 x bsize x 484 -> 200 x bsize x 128
        Sgdfs = self.supervisor(Gdfs) # 200 x bsize x 128 -> 200 x bsize x 128

        # Synthetic data generated
        Rdfs = self.recovery(Sgdfs) # 200 x bsize x 128 -> 200 x bsize x 484

        ## Generator loss
        Dfake = self.discriminator(Gdfs) # 200 x bsize x 128 -> 200 x bsize x 2
        Dsfake = self.discriminator(Sgdfs) # 200 x bsize x 128 -> 200 x bsize x 2

        Gfloss = F.binary_cross_entropy_with_logits(Dfake, torch.ones_like(Dfake))
        Gfsloss = F.binary_cross_entropy_with_logits(Dsfake, torch.ones_like(Dsfake))
        Sloss = F.mse_loss(Rdfs,dfs)

        Gloss = Gfloss + Gfsloss + Sloss*50

        return Gloss

    def _inference(self,dfs):
        """Inference for generating synthetic data
        Args:
            - Edfs: Enter the data from Embedder [200 x 1 x 484]
        Returns:
            - Gdfs: Complete generation of data  [200 x 1 x 484]
        """
        # Generator forward pass
        gdfs = self.generator(dfs) # 200 x bsize x 484 -> 200 x bsize x 128
        sdfs = self.supervisor(gdfs) # 200 x bsize x 128 -> 200 x bsize x 128

        # Synthetic data generated
        Idfs = self.recovery(sdfs) # 200 x bsize x 128 -> 200 x bsize x 484
        return Idfs

    def forward(self,Idata,Gdata,Label,Obj):
        """
        Args:
            - Idata: The input features (B, H, F)
            - Gdata: Random generator
            - Label: Labels for identification
            - obj: The network to be trained
            ('autoencoder', 'supervisor', 'generator', 'discriminator')
        Returns:
            - loss: The loss for the forward pass
            - new_dfs: The generated data
        """
        Idata = Idata.to(torch.float32)
        #Label = Label.long()
        if Obj == "autoencoder":
            # Embedder & Recovery
            loss = self._autoencoder_forward(Idata)

        elif Obj == "supervisor":
            # Supervisor
            loss = self._supervisor_forward(Idata)
        
        elif Obj == "generator":
            # Generator
            loss = self._generator_forward(Idata)

        elif Obj == "discriminator":
            # Discriminator
            loss = self._discriminator_forward(Gdata.to(torch.float32),Label)

        elif Obj == "inference":
            new_dfs = self._inference(Gdata.to(torch.float32))
            #new_dfs = new_dfs.cpu().detach()
            return new_dfs
        
        else: raise ValueError(" 'obj' should be either 'autoencoder','supervisor','generator','discriminator','inference' ")

        return loss