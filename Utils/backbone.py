from ast import arg
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pywt

class ECA(nn.Module):
    """
        Capturing time-domain correlations using attention mechanisms
    """
    def __init__(self,feature):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            in_channels = 1,
            out_channels = 1,
            kernel_size = 3,
            padding = 1,
            bias = False
        )

        self.sigmoid = nn.Sigmoid()
        self.feature = feature

    def forward(self, x):

        y = x.permute(0,2,1).unsqueeze(-1) # 64 x 120 x 200 -> 64 x 200 x 120 x 1
        y = self.avg_pool(y)  # 64 x 200 x 120 x 1 -> 64 x 200 x 1 x 1

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).permute(0,2,1)) # 64 x 1 x 200

        # Multi-scale information fusion
        y = self.sigmoid(y)
        y = y.repeat(1,self.feature,1)

        return y

class FIC(nn.Module):
    """
        Capturing frequency domain features using 1D convolution
    """
    def __init__(self, window_size, kernel_size, stride):
        super(FIC, self).__init__()
        self.window_size = window_size
        self.k_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels = 1,
            out_channels = window_size,
            kernel_size = kernel_size,
            stride = stride,
            padding = 0,
            bias = False)
        # [batch_size, out_channels, (n-f)/s+1] n：max_len, p：padding, f： filter_size, s： stride
        self.init()

    def forward(self, x):
        # x: 64 x 180 x 200 [batch_size x dim_feature x nseg]
        B, C = x.size(0), x.size(1)

        x = torch.reshape(x, (B * C, -1)).unsqueeze(1) # 升维度 11520 x 1 x 200
        x = self.conv(x) # 11520 x out_channels x int(3d-K_size)/stride
        x = torch.reshape(x, (B, C, -1, x.size(-1))) # batch_size x 180 x 
        return x # B * C * fc * L
    

    def init(self):
        # Fourier weights initialization
        basis = torch.tensor([math.pi * 2 * j / self.k_size for j in range(self.k_size)])
        weight = torch.zeros((self.window_size,self.k_size))
        for i in range(self.window_size):
            f = int(i / 2) + 1
            if i % 2 == 0:
                weight[i] = torch.cos(f * basis)
            else:
                weight[i] = torch.sin(-f * basis)
        self.conv.weight = torch.nn.Parameter( weight.unsqueeze(1), requires_grad=True)

class TSE(nn.Module):
    """
        Integrating the frequency domain traits in FIC for feature extraction
    """
    def __init__(self, window_size, k_size, stride, n_max, type): # 1 1 1 60 / 2 1 2 40 ....
        super(TSE, self).__init__()

        self.window_size = window_size
        self.k_size = k_size
        self.n_max = n_max
        self.datatype = type

        # FIC Layer
        self.FIC = FIC(
            window_size = window_size,
            kernel_size = k_size,
            stride = stride)

    def forward(self, x):


        if self.datatype == 'wifi':
            h_f = self.FIC(x) # 64 x 180 x out_channels x 卷积后的尺寸
            h_f_pos, _ = (h_f).topk(self.n_max, dim = -1, largest = True, sorted = True)
        
        elif self.datatype == 'rfid':
            h_f = self.FIC(x) # 64 x 180 x out_channels x 卷积后的尺寸
            h_f_pos, _ = (h_f).topk(self.n_max, dim = -1, largest = True, sorted = True)
        
        elif self.datatype == 'uwb':
            h_f = self.FIC(x) # 64 x 180 x out_channels x 卷积后的尺寸
            h_f_pos, _ = (h_f).topk(self.n_max, dim = -1, largest = True, sorted = True)

        elif self.datatype == 'mmwave':
            h_f = self.FIC(x) # 64 x 180 x out_channels x 卷积后的尺寸
            h_f_pos, _ = (h_f).topk(self.n_max, dim = -1, largest = True, sorted = True)
        
        return h_f_pos

# ECA, FIC, TSE are all serving AutoencoderNetwork 
class AutoencoderNetwork(nn.Module):
    def __init__(self, args):
        super(AutoencoderNetwork, self).__init__()
        
        self.input_size = args.n_seg # 200
        self.sensor_num = args.semsor_num # 120
        self.hidden_channel = args.hidden_channel # 40
        self.datatype = args.type
        self.window_list = [1, 2, 4, 8, 16, 32, 64, 128]
        self.kernel_list = [1, 1, 2, 4, 8, 16, 32, 64]
        self.stride_list = [2, 2, 4, 4, 16, 16, 36, 36]


        if args.type == 'wifi':
            self.n_list = [60,60,27,27,10,10,3,3]
        elif args.type == 'rfid':
            self.n_list = [22,22,14,14,5,5,1,1]
        elif args.type == 'uwb':
            self.n_list = [52,52,25,25,10,10,3,3]
        elif args.type == 'mmwave':
            self.window_list = [1,1,2,2,4,4,8,8]
            self.kernel_list = [1,1,2,2,3,3,4,4]
            self.stride_list = [1,1,2,2,4,4,8,8]
            self.n_list = [6,5,4,4,2,2,1,1]

        n_len = len(self.n_list)

        self.current_size = [1 + int((self.input_size - self.kernel_list[i]) / self.stride_list[i])  for i in range(n_len)]
        self.ts_encoders = nn.ModuleList([
           TSE(self.window_list[i], self.kernel_list[i], self.stride_list[i], self.n_list[i], self.datatype) for i in range(n_len)
            ]) # One-dimensional convolution of eight layers
   
        # o.size(): B * C * num_frequency_channel * current_size
        #self.multi_channel_fusion = nn.ModuleList([nn.ModuleList() for _ in range(n_len)])
        self.conv_branches = nn.ModuleList([nn.ModuleList() for _ in range(n_len)])
        self.bns =  nn.ModuleList([nn.BatchNorm1d(self.hidden_channel) for _ in range(n_len)])

        # 九层的2d卷积
        self.multi_channel_fusion = nn.ModuleList([nn.Conv2d(in_channels = self.sensor_num, out_channels = self.hidden_channel,
              kernel_size = (self.window_list[i], 1)) for i in range(n_len)])

        self.feature_max = args.hidden_channel

        # ECA : Capturing time-domain correlations using attention mechanisms
        self.ECA1 = ECA(self.feature_max)
        self.ECA2 = ECA(self.sensor_num)

        # Compressed full connection
        self.auto_linear = nn.Linear(self.feature_max,self.sensor_num)


    def forward(self, x):
        """
            Forward pass for the supervisor for predicting next step (batch_size x 120 x 200)
        Args:
            - Data: Latent representation
        Returns: 
            - X: Data with time-frequency domain features after convolution
            - X1: Raw data for full connection restoration
        """
        
        multi_scale_x = []
        eca_x, _ = (x).topk(self.feature_max, dim = -2, largest = True, sorted = True)
        eca_x = self.ECA1(eca_x)
        eca_x1 = self.ECA2(x)

        for i in range(len(self.current_size)): # len = 8
            tmp = self.ts_encoders[i](x) # 
            tmp = F.relu(self.bns[i](self.multi_channel_fusion[i](tmp).squeeze(2)))
            multi_scale_x.append(tmp)

        x = torch.cat(multi_scale_x, -1)
        x = x * eca_x
        x1 = self.auto_linear(x.permute(0,2,1)).permute(0,2,1) * eca_x1

        return x,x1

# SupervisorNetwork
class SupervisorNetwork(nn.Module):
    def __init__(self,args):
        super(SupervisorNetwork,self).__init__()
        self.n_feature = args.semsor_num
        self.nseg = args.n_seg
        self.batch_size = args.batch_size
        self.datatype = args.type
        self.rpnum = args.semsor_num // args.hidden_channel

        self.dwt_root = np.load(args.dwt_root)
        self.dwt_high = torch.from_numpy(self.dwt_root[0]) # 1 x 1 x 200 -> 3 x 120 x 200 
        self.dwt_low = torch.from_numpy(self.dwt_root[1]) # 1 x 1 x 200 -> 3 x 120 x 200 

        self.sup_linear = nn.Linear(self.n_feature,self.n_feature)

    def forward(self,data,feature_data,label):
        """Forward pass for the supervisor for predicting next step
        Args:
            - data: Base data requiring regulatory adjustment (bsize x 120 x 200)
            - feature_data: Learned / extracted feature data (args.hidden_channel x 200)
            - label: Categories of truth and falsity of data (0/1)
        Returns:
            - Sdata: Adjusted data (bsize x 120 x 200)
            - A: Low frequency features extracted from discrete wavelet variations (200)
            - D: High frequency features extracted from discrete wavelet variations (200)
        """
        feature_data = feature_data.repeat(1,self.rpnum,1) # 64 x 40 x 200 -> 64 x 120 x 200
        
        if label == "0": # Find trusted dwt data
            Sdata = feature_data
            max_data = torch.max((torch.max(data,0).values),0).values.cpu().detach().numpy()
            cA , cD = pywt.dwt(max_data,'db5')
            A,D = pywt.upcoef('a',cA,'db5',1,self.nseg),pywt.upcoef('d',cD,'db5',1,self.nseg)
            A,D = torch.from_numpy(A),torch.from_numpy(D)

        elif label == "1": # Adaptation of the generated data using the effective features learned in pre-training
            dwt_low, dwt_high = self.dwt_low.repeat(data.shape[0],self.n_feature,1).cuda(), self.dwt_high.repeat(data.shape[0],self.n_feature,1).cuda()
            Sdata = (data+feature_data+dwt_low)/3 + 0.19*dwt_high
            dwt_data, _ = (torch.mean(Sdata,0)).topk(1, dim = 0, largest = True, sorted = True)
            cA , cD = pywt.dwt(dwt_data.squeeze(0).cpu().detach().numpy(),'db5')
            A,D = pywt.upcoef('a',cA,'db5',1,self.nseg),pywt.upcoef('d',cD,'db5',1,self.nseg)
            A,D = torch.from_numpy(A),torch.from_numpy(D)
            
        else: raise ValueError(" Enter the wrong mode selection ")

        Sdata = self.sup_linear(Sdata.permute(0,2,1)).permute(0,2,1) # 64 x 120 x 200 -> 64 x 120 x 200
            
        return Sdata,A,D

# GeneratorNetwork
class GeneratorNetwork(nn.Module):
    def __init__(self,args):
        super(GeneratorNetwork,self).__init__()
        self.feature_adjustment = SupervisorNetwork(args)
        self.linear = torch.nn.Linear(args.semsor_num, args.semsor_num)
        self.datatype = args.type

    def forward(self,data,feature_data,label="1"):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - data: Base data requiring regulatory adjustment (bsize x 120 x 200)
            - feature_data: Learned / extracted feature data (args.hidden_channel x 200)
            - label: Categories of truth and falsity of data (1)
        Returns:
            - gdata: The data after feature extraction and adjustment
        """
        adjustment_data, _, _ = self.feature_adjustment(data,feature_data,label)
        gdata = self.linear(adjustment_data.permute(0,2,1)).permute(0,2,1)

        return gdata

# DiscriminatorNetwork
class DiscriminatorNetwork(nn.Module):
    def __init__(self,args):
        super(DiscriminatorNetwork,self).__init__()

        self.int_feature = args.hidden_channel
        if args.type == 'rfid':
            self.out_feature = 1
        else:
            self.out_feature = int(args.hidden_channel/2)
        self.out_channels = int((args.n_seg - 16)/4) + 1

        self.blocks = nn.Sequential(
            nn.Conv1d(self.int_feature,self.out_feature,16,4),
            nn.BatchNorm1d(self.out_feature),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(self.out_feature*self.out_channels, self.int_feature),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.int_feature, 2)
        )

    def forward(self,data):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - data: Data to be judged for authenticity
        Returns:
            - logits: Collection of true and false data
        """
        out = self.blocks(data)
        logits = self.classifier(self.dropout(out.view(len(out),-1)))

        return logits

# Network backbone
class Penglbackbone(nn.Module):
    def __init__(self,args):
        super(Penglbackbone,self).__init__()

        self.autoencoder = AutoencoderNetwork(args)
        self.generator = GeneratorNetwork(args)
        self.supervisor = SupervisorNetwork(args)
        self.discriminator = DiscriminatorNetwork(args)
        self.rfidconv = nn.Conv1d(1,1,1)
        self.datatype = args.type
        self.mloss = args.mloss

        self.Cosine = nn.CosineSimilarity(dim=-1,eps=1e-6)

    def _autoencoder_supervisor_forward(self,data):
        """
            Joint training module for autoencoder and supervisor
        Args:
            - data: Initial data: for feature learning
        Returns:
            - Aloss: The loss of autoencoder and supervisor training process
            - A: Discrete wavelet transform of A
            - D: Discrete wavelet transform of D
        """
        # Forward Pass
        size = data.shape[0]

        xdata, x1data = self.autoencoder(data)
        sdata, A, D = self.supervisor(x1data,xdata,'0')

        if self.datatype == 'wifi' or self.datatype == 'uwb':
            aloss1 = F.mse_loss(x1data,data)*self.mloss  - torch.mean(self.Cosine(x1data.reshape(size,-1),data.reshape(size,-1))) 
            aloss2 = F.mse_loss(sdata,data)*self.mloss  - torch.mean(self.Cosine(sdata.reshape(size,-1),data.reshape(size,-1)))
        elif self.datatype == 'rfid':
            aloss1 = F.l1_loss(x1data.squeeze(),data.squeeze())*self.mloss  - torch.mean(self.Cosine(x1data.squeeze(),data.squeeze())) 
            aloss2 = F.l1_loss(sdata.squeeze(),data.squeeze())*self.mloss  - torch.mean(self.Cosine(sdata.squeeze(),data.squeeze()))
        elif self.datatype == 'mmwave':
            aloss1 = F.mse_loss(x1data,data)*self.mloss  - torch.mean(self.Cosine(x1data.reshape(size,-1),data.reshape(size,-1))) 
            aloss2 = F.mse_loss(sdata,data)*self.mloss  - torch.mean(self.Cosine(sdata.reshape(size,-1),data.reshape(size,-1)))

        Aloss = aloss1 + aloss2

        return Aloss,A,D

    def _generator_discriminator_forward(self,data,label):
        """
            Joint training module for generator and discriminator
        Args:
            - data: True data / False data: for joint training
            - label: Data Category [T/F]
        Returns:
            - loss: The loss of generator and discriminator training process
        """
        # Real & fake

        if label == "0":
            feature_data, auto_data = self.autoencoder(data)
            generator_data = self.generator(auto_data,feature_data,label)
            feature_gener, _ = self.autoencoder(generator_data)

            identify_data_1 = self.discriminator(feature_data)
            identify_data_2 = self.discriminator(feature_gener)

            identify_loss1 = F.binary_cross_entropy_with_logits(identify_data_1, torch.zeros_like(identify_data_1)) + F.mse_loss(auto_data,data)*self.mloss
            identify_loss2 = F.binary_cross_entropy_with_logits(identify_data_2, torch.zeros_like(identify_data_2)) + F.mse_loss(generator_data,data)*self.mloss

            Loss = identify_loss1 + identify_loss2

            return Loss,feature_data

        elif label =="1":

            dwt = torch.from_numpy(np.load(f'./Data/{self.datatype}/dwt.npy')).squeeze(1).squeeze(1)
            feature = torch.from_numpy(np.load(f'./Data/{self.datatype}/feature.npy')).cuda()
            feature = feature.unsqueeze(0).repeat(data.shape[0],1,1)

            gdata = self.generator(data,feature,label)
            feature_data,generator_data = self.autoencoder(gdata)
            sdata_1, A1, D1 = self.supervisor(generator_data,feature_data,label)
            sdata_2, A2, D2 = self.supervisor(gdata,feature_data,label)
            sdata_1, _ = self.autoencoder(sdata_1)
            sdata_2, _ = self.autoencoder(sdata_2)

            Sloss1 = F.mse_loss(A1,dwt[0])*10 + F.mse_loss(D1,dwt[1])*10
            Sloss2 = F.mse_loss(A2,dwt[0])*10 + F.mse_loss(D2,dwt[1])*10

            # Forward Pass
            dis_data_1= self.discriminator(sdata_1)
            dis_data_2 = self.discriminator(sdata_2)

            
            Diloss = F.binary_cross_entropy_with_logits(dis_data_1, torch.ones_like(dis_data_1))
            Dsloss = F.binary_cross_entropy_with_logits(dis_data_2, torch.ones_like(dis_data_2))

            Loss = Diloss + Dsloss + Sloss1 + Sloss2

            return Loss

        else: raise ValueError(" 'label' should be either '0' or '1' ")


    def _inference(self,data):
        """Inference for generating synthetic data
        Args:
            - data: Randomly generated timing data [batsh_size x 120 x 200]
        Returns:
            - data: Generated reliable timing data  [batsh_size x 120 x 200]
        """
        # Generator forward pass
        feature = torch.from_numpy(np.load(f'./Data/{self.datatype}/feature.npy')).cuda()
        feature = feature.unsqueeze(0).repeat(data.shape[0],1,1)

        gdata = self.generator(data,feature) # 200 x bsize x 484 -> 200 x bsize x 128
        
        return gdata

    def forward(self,Idata,Label,Type,Obj):
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

        if Type == 'wifi' or Type == 'uwb':
            Idata = Idata
        elif Type == 'rfid':
            Idata = self.rfidconv(Idata.unsqueeze(1)) # 32 x 1 x 84 -> 32 x 2 x 84
        elif Type == 'mmwave':
            a, b, c, _ = Idata.shape
            Idata = Idata.reshape(a,b,c*c).permute(0,2,1) # # 32 x 1024 x 25

        if Obj == "autoencoder":
            # Embedder & Recovery
            return self._autoencoder_supervisor_forward(Idata)

        elif Obj == "joint":
            # Generator x Discriminator
            return self._generator_discriminator_forward(Idata,Label)

        elif Obj == "inference":
            return self._inference(Idata)

        else: raise ValueError(" 'obj' should be either 'autoencoder','supervisor','generator','discriminator','inference' ")

# eval()
if __name__ == '__main__':

    stft_m = AutoencoderNetwork()
    x = torch.zeros(3, 120, 200)
    output = stft_m(x)
    print(output.size())