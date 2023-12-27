import numpy as np
import torch
import pickle 
from utils.utils import *
import os
from collections import OrderedDict
import h5py

from argparse import Namespace
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored


def train_loop_survival_coattn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    model.test_mode=False
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        hazards, S, Y_hat, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size:'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk)))
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival_coattn(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.test_mode=True
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            
            hazards, S, Y_hat, A = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, results_dict

        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg


    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    print("val c index: ", c_index)
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_survival_coattn(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):
        
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            hazards, survival, Y_hat, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, results_dict

        risk = (-torch.sum(survival, dim=1).cpu().numpy()).item()
        event_time = event_time.item()
        c = c.item()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index



def summary_survival_coattn_importance(model, loader):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.test_mode=True
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    c_indeces=[]
    all_attentions=[]
    for i in range(8):
        print(i, " of ", 7)
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):
            data_WSI = data_WSI.to(device)

            data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
            data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
            data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
            data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
            data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
            data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)
            c = c.type(torch.FloatTensor).to(device)
            slide_id = slide_ids.iloc[batch_idx]
            
            if i==0: data_WSI=None
            if i==1: data_omic1=None
            if i==2: data_omic2=None
            if i==3: data_omic3=None
            if i==4: data_omic4=None
            if i==5: data_omic5=None
            if i==6: data_omic6=None

            with torch.no_grad():
                hazards, survival, Y_hat, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, results_dict
                if i==7:
                    bottom_attention,top_attention=model.attention_rollout(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                    all_attentions.append((slide_id,bottom_attention,top_attention))
                    
            risk = (-torch.sum(survival, dim=1).cpu().numpy()).item()
            event_time = event_time.item()
            c = c.item()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c
            all_event_times[batch_idx] = event_time
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        c_indeces.append(c_index)


    for i in range(7):
        print(i, " of ", 7)
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):

            if i==0: data_WSI,data_omic1,data_omic2,data_omic3,data_omic4,data_omic5,data_omic6=data_WSI.to(device), None,None,None,None,None,None
            if i==1: data_WSI,data_omic1,data_omic2,data_omic3,data_omic4,data_omic5,data_omic6= None,data_omic1.type(torch.FloatTensor).to(device),None,None,None,None,None
            if i==2: data_WSI,data_omic1,data_omic2,data_omic3,data_omic4,data_omic5,data_omic6= None,None,data_omic2.type(torch.FloatTensor).to(device),None,None,None,None
            if i==3: data_WSI,data_omic1,data_omic2,data_omic3,data_omic4,data_omic5,data_omic6= None,None,None,data_omic3.type(torch.FloatTensor).to(device),None,None,None
            if i==4: data_WSI,data_omic1,data_omic2,data_omic3,data_omic4,data_omic5,data_omic6= None,None,None,None,data_omic4.type(torch.FloatTensor).to(device),None,None
            if i==5: data_WSI,data_omic1,data_omic2,data_omic3,data_omic4,data_omic5,data_omic6= None,None,None,None,None,data_omic5.type(torch.FloatTensor).to(device),None
            if i==6: data_WSI,data_omic1,data_omic2,data_omic3,data_omic4,data_omic5,data_omic6= None,None,None,None,None,None,data_omic6.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)

            c = c.type(torch.FloatTensor).to(device)
            slide_id = slide_ids.iloc[batch_idx]

            with torch.no_grad():
                hazards, survival, Y_hat, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6) # return hazards, S, Y_hat, A_raw, results_dict

            risk = (-torch.sum(survival, dim=1).cpu().numpy()).item()
            event_time = event_time.item()
            c = c.item()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c
            all_event_times[batch_idx] = event_time
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        c_indeces.append(c_index)


    patch_risks=[]
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):
        patch_risk=[]
        for patch_feature in data_WSI:
            with torch.no_grad():
                hazards, survival, Y_hat, A  = model(x_path=patch_feature.unsqueeze(0).to(device), x_omic1=None, x_omic2=None, x_omic3=None, x_omic4=None, x_omic5=None, x_omic6=None)
                risk = (-torch.sum(survival, dim=1).cpu().numpy()).item()
                patch_risk.append(risk)
        patch_risks.append((slide_ids.iloc[batch_idx],patch_risk))

    return patient_results, c_indeces,all_attentions,patch_risks