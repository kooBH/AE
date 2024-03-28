import torch
import torch.nn as nn
from utils.metric import run_metric
from models.ALVIT import FrameAutoEncoder


def get_model(hp):

    if hp.model.type == "ALVIT" :
        model = FrameAutoEncoder(**hp.model.architecture)
    else :
        raise ValueError("Model type not found : {}".format(hp.model.type))

    return model

def run(data,model,criterion,hp,device="cuda:0",ret_output=False): 
    crop = data.to(device)

    output = model(crop)

    if hp.loss.type == "MSELoss" : 
        loss = criterion(output,crop).to(device)
        loss /= crop.shape[0]
    if ret_output :
        return output, loss
    else : 
        return loss


def evaluate(hp, model,list_data,device="cuda:0"):
    #### EVAL ####
    model.eval()
    with torch.no_grad():
        ## Metric
        metric = {}
        for m in hp.log.eval : 
            metric["{}".format(m)] = 0.0

        for pair_data in list_data : 
            path_noisy = pair_data[0]
            path_clean = pair_data[1]
            noisy = rs.load(path_noisy,sr=hp.data.sr)[0]
            noisy = torch.unsqueeze(torch.from_numpy(noisy),0).to(device)
            estim = model(noisy).cpu().detach().numpy()[0]
            clean = rs.load(path_clean,sr=hp.data.sr)[0]

            if len(clean) > len(estim) :
                clean = clean[:len(estim)]
            else :
                estim = estim[:len(clean)]
            for m in hp.log.eval : 
                val= run_metric(estim,clean,m) 
                metric["{}".format(m)] += val
            
        for m in hp.log.eval : 
            key = "{}".format(m)
            metric[key] /= len(list_data)
    return metric