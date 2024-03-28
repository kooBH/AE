import torch
import argparse
import os
import glob
from utils.hparams import HParam
from tqdm.auto import tqdm
from common import get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,required=True)
    parser.add_argument('--default', type=str, default=None)
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('-i','--dir_in',type=str,required=True)
    parser.add_argument('-o','--dir_out',type=str,required=True)
    args = parser.parse_args()

    ## Parameters 
    hp = HParam(args.config,args.default)
    print('NOTE::Loading configuration :: ' + args.config)

    device = args.device
    torch.cuda.set_device(device)

    num_epochs = 1
    batch_size = 1
 
    ## dirs 
    dir_in = args.dir_in
    dir_out = args.dir_out
    os.makedirs(dir_out,exist_ok=True)

    list_data = glob.glob(os.path.join(dir_in,"**","*.pt"), recursive=True)

    ## Model
    model = get_model(hp).to(device)
    model.load_state_dict(torch.load(args.chkpt, map_location=device))
    model.eval()
    print('NOTE::Loading pre-trained model : ' + args.chkpt)

    with torch.no_grad():
        for path in tqdm(list_data) : 
            dir_item = path.split("/")[-2]
            name_item = (path.split("/")[-1]).split(".")[0]

            lip = torch.load(path)().to(device)
            lip = torch.unsqueeze(lip,1)
            feat = model.encode(lip)[0]
            import pdb;pdb.set_trace()
            
            ## Save
            os.makedirs(os.path.join(dir_out,dir_item),exist_ok=True)
            torch.save(feat.to("cpu") ,os.path.join(dir_out,dir_item,name_item+".pt"))








   
