import argparse

def parse_args(data_type):##pattern ：test or train
    parser = argparse.ArgumentParser(
        description = 'Adversarial generation of Time Series Data in Wi-Fi'
    )

    # Experiment Arguments
    parser.add_argument("--is_train", default=False)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--exp', default=f'./{data_type}/20220426', type=str)
    parser.add_argument('--seed', default = 999, type = int)
    parser.add_argument('--type', default = data_type, type = str)
    
    # Data Arguments
    if data_type == 'wifi': # 120 x 200

        parser.add_argument('--mloss', default=1, type=int)
        parser.add_argument('--n_seg', default=200, type=int)
        parser.add_argument('--semsor_num', default=120, type=int)
        parser.add_argument('--hidden_channel', default=40, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--dwt_root',default=f'./data/{data_type}/dwt.npy',type=str)

    elif data_type == 'rfid': # 84

        parser.add_argument('--mloss', default=1, type=int)
        parser.add_argument('--n_seg', default=84, type=int)
        parser.add_argument('--semsor_num', default=1, type=int)
        parser.add_argument('--hidden_channel', default=1, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--dwt_root',default=f'./data/{data_type}/dwt.npy',type=str)

    elif data_type == 'uwb': # 160 x 180

        parser.add_argument('--mloss', default=50, type=int)
        parser.add_argument('--n_seg', default=180, type=int)
        parser.add_argument('--semsor_num', default=160, type=int)
        parser.add_argument('--hidden_channel', default=40, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--dwt_root',default=f'./data/{data_type}/dwt.npy',type=str)

    elif data_type == 'mmwave': # 25 x 32 x 32

        parser.add_argument('--mloss', default=1, type=int)
        parser.add_argument('--n_seg', default=25, type=int)
        parser.add_argument('--semsor_num', default=1024, type=int)
        parser.add_argument('--hidden_channel', default=64, type=int)
        parser.add_argument('--learning_rate', default=1e-3, type=float)
        parser.add_argument('--dwt_root',default=f'./data/{data_type}/dwt.npy',type=str)

    else: raise ValueError(" Type : [WiFi / Rfid / Uwb / mmWave] ")

    # Model Arguments
    parser.add_argument('--args_path', default=f"./Output/{data_type}/", type=str)
    parser.add_argument('--model_path', default=f"./Output/{data_type}/", type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=4, type=int)

    return parser.parse_args()

if __name__ == '__main__':

    opt = parse_args('wifi')
    print(opt)
    