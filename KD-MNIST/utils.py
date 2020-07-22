import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.cm
from matplotlib.cm import ScalarMappable
from model import TeacherNet, StudentNet, KidNet
import os
from copy import deepcopy

def getLossAccuracyOnDataset(network, dataset_loader, fast_device, criterion=None):
	"""
	Returns (loss, accuracy) of network on given dataset
	"""
	network.eval()
	accuracy = 0.0
	loss = 0.0
	dataset_size = 0
	for j, D in enumerate(dataset_loader, 0):
		X, y = D
		X = X.to(fast_device)
		y = y.to(fast_device)
		with torch.no_grad():
			pred = network(X)
			if criterion is not None:
				loss += criterion(pred, y) * y.shape[0]
			accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
		dataset_size += y.shape[0]
	loss, accuracy = loss / dataset_size, accuracy / dataset_size
	return loss, accuracy

def TeacherIterate(path, dataset_loader, device, criterion=None):
    """
    Returns (loss, accuracy) of network on given dataset
    """
    
    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break
    n = len(f)
    best_acc = 0
    name = ''
    for i in range(n):

        p_inp = float(f[i].split(',')[1].split('=')[1])
        p_hid = float(f[i].split(',')[0].split('=')[1])
        
        TNet = TeacherNet(p_inp,p_hid)
        model_file = os.path.join(path,f[i])
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['model_state_dict']
        TNet.load_state_dict(state_dict)
        TNet = TNet.to(device)
        TNet.eval()
        
        accuracy = 0.0
        loss = 0.0
        dataset_size = 0
        with torch.no_grad():
            for j, D in enumerate(dataset_loader):
                X, y = D
                X = X.to(device)
                y = y.to(device)

                pred = TNet(X)
                loss += criterion(pred, y).item() * y.shape[0]
                accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
                dataset_size += y.shape[0]
        
        loss, accuracy = loss / dataset_size, 100.0*accuracy / dataset_size
        print('For %s, Testing loss= %.4f acc= %.4f'%(f[i],loss,accuracy))
        
        if best_acc < accuracy:
            best_acc = accuracy
            name = f[i]
    return name
        
def StudentIterate(path, dataset_loader, device, criterion=None):
    """
    Returns (loss, accuracy) of network on given dataset
    """
    
    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break
    n = len(f)
    for i in range(n):

        p_inp = float(f[i].split(',')[3].split('=')[1])
        p_hid = float(f[i].split(',')[2].split('=')[1])
        
        SNet = StudentNet(p_inp,p_hid)
        model_file = os.path.join(path,f[i])
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['best_state_dict']
        SNet.load_state_dict(state_dict)
        SNet = SNet.to(device)
        SNet.eval()
        
        accuracy = 0.0
        loss = 0.0
        dataset_size = 0
        with torch.no_grad():
            for j, D in enumerate(dataset_loader):
                X, y = D
                X = X.to(device)
                y = y.to(device)

                pred = SNet(X)
                loss += criterion(pred, y).item() * y.shape[0]
                accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
                dataset_size += y.shape[0]
        
        loss, accuracy = loss / dataset_size, 100.0*accuracy / dataset_size
        print('For %s, Testing loss= %.4f acc= %.4f'%(f[i],loss,accuracy))
        
        #LAST STEP ACC
        state_dict = checkpoint['model_state_dict']
        SNet.load_state_dict(state_dict)
        SNet = SNet.to(device)
        SNet.eval()

        accuracy = 0.0
        loss = 0.0
        dataset_size = 0
        with torch.no_grad():
            for j, D in enumerate(dataset_loader):
                X, y = D
                X = X.to(device)
                y = y.to(device)

                pred = SNet(X)
                loss += criterion(pred, y).item() * y.shape[0]
                accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
                dataset_size += y.shape[0]

        loss, accuracy = loss / dataset_size, 100.0*accuracy / dataset_size
        print('For %s, Testing loss= %.4f acc= %.4f'%(f[i],loss,accuracy))
        
        print("_________________________")

def RAPIterate(path, dataset_loader, device, criterion=None):
    """
    Returns (loss, accuracy) of network on given dataset
    """
    
    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break
    n = len(f)
    for i in range(n):

        SNet = StudentNet(0,0)
        model_file = os.path.join(path,f[i])
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['best_state_dict']
        SNet.load_state_dict(state_dict)
        SNet = SNet.to(device)
        SNet.eval()
        
        accuracy = 0.0
        loss = 0.0
        dataset_size = 0
        with torch.no_grad():
            for j, D in enumerate(dataset_loader):
                X, y = D
                X = X.to(device)
                y = y.to(device)

                pred = SNet(X)
                loss += criterion(pred, y).item() * y.shape[0]
                accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
                dataset_size += y.shape[0]
        
        loss, accuracy = loss / dataset_size, 100.0*accuracy / dataset_size
        print('For %s, Testing loss= %.4f acc= %.4f'%(f[i],loss,accuracy))

        print('Last State')
        state_dict = checkpoint['model_state_dict']
        SNet.load_state_dict(state_dict)
        SNet = SNet.to(device)
        SNet.eval()

        accuracy = 0.0
        loss = 0.0
        dataset_size = 0
        with torch.no_grad():
            for j, D in enumerate(dataset_loader):
                X, y = D
                X = X.to(device)
                y = y.to(device)

                pred = SNet(X)
                loss += criterion(pred, y).item() * y.shape[0]
                accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
                dataset_size += y.shape[0]

        loss, accuracy = loss / dataset_size, 100.0*accuracy / dataset_size
        print('For %s, Testing loss= %.4f acc= %.4f'%(f[i],loss,accuracy))
        print("______________________________")

def trainTeacher(hparam, num_epochs, 
                    train_loader, val_loader,
                    device=torch.device('cpu')):
    """
    Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch 
    Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
    """
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    
    TNet = TeacherNet(hparam['dropout_input'],hparam['dropout_hidden'])
    TNet = TNet.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(TNet.parameters(), lr=hparam['lr'], weight_decay=hparam['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])
    max_val_acc = float('-inf')
    for epoch in range(num_epochs):
        TNet.train()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(train_loader, 0):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = TNet(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            run_acc += torch.sum(torch.argmax(outputs, dim=1) == y).item()
            run_loss += loss.item()*y.size(0)
            total += y.size(0) 
        
        lr_scheduler.step()
        
        train_loss = run_loss/total
        train_acc = 100.0*float(run_acc)/total
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        TNet.eval()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(val_loader):
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                
                outputs = TNet(X)
                loss = criterion(outputs, y)
                
                run_acc += torch.sum(torch.argmax(outputs, dim=1) == y).item()
                run_loss += loss.item()*y.size(0)
                total += y.size(0) 

        val_loss = run_loss/total
        val_acc = 100.0*float(run_acc)/total
        
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            #temp = 
            print('[%d/%d] train loss: %.3f train accuracy: %.3f || val loss: %.3f val accuracy: %.3f' %
              (epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

        
    return ({'train_loss': train_loss_list, 
            'train_acc': train_acc_list, 
            'val_loss': val_loss_list, 
            'val_acc': val_acc_list},TNet.state_dict())


def trainStudent(TNet,hparam, num_epochs, 
                        train_loader, val_loader, 
                        device):
    """
    Trains teacher on given hyperparameters for given number of epochs; Pass val_loader=None when not required to validate for every epoch
    Return: List of training loss, accuracy for each update calculated only on the batch; List of validation loss, accuracy for each epoch
    """
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    T = hparam['T']
    alpha = hparam['alpha']
    
    SNet = StudentNet(hparam['dropout_input'],hparam['dropout_hidden'])
    SNet = SNet.to(device)
    
    TNet.eval()
    for param in TNet.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(SNet.parameters(), lr=hparam['lr'], weight_decay=hparam['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])
    
    def LossFn(teacher_pred, student_pred, y, T, alpha):
        
        if (alpha > 0):
            loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), 
        F.softmax(teacher_pred / T, dim=1), reduction='batchmean') * (T ** 2) * alpha + F.cross_entropy(student_pred, y) * (1 - alpha)
        else:
            loss = F.cross_entropy(student_pred, y)
        return loss

    max_val_acc = float('-inf')
    for epoch in range(num_epochs):
        SNet.train()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(train_loader, 0):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            output_s = SNet(X)
            with torch.no_grad():
                output_t = TNet(X)
            
            loss = LossFn(output_t,output_s, y, T, alpha)
            loss.backward()
            optimizer.step()
            
            run_acc += torch.sum(torch.argmax(output_s, dim=1) == y).item()
            run_loss += loss.item()*y.size(0)
            total += y.size(0) 
        
        lr_scheduler.step()
        
        train_loss = run_loss/total
        train_acc = 100.0*float(run_acc)/total
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        SNet.eval()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(val_loader):
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                
                output_s = SNet(X)
                output_t = TNet(X)
            
                loss = LossFn(output_t,output_s, y, T, alpha)
                
                run_acc += torch.sum(torch.argmax(output_s, dim=1) == y).item()
                run_loss += loss.item()*y.size(0)
                total += y.size(0) 

        val_loss = run_loss/total
        val_acc = 100.0*float(run_acc)/total
        
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            temp = SNet.state_dict()
            print('[%d/%d] train loss: %.3f train accuracy: %.3f || val loss: %.3f val accuracy: %.3f' %
              (epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))
            
    
    return ({'train_loss': train_loss_list, 
            'train_acc': train_acc_list, 
            'val_loss': val_loss_list, 
            'val_acc': val_acc_list},SNet.state_dict(),temp)


def trainRAP(TNet,hparam, num_epochs, 
                        train_loader, val_loader, 
                        device):
    train_loss_list, train_acc_list, val_acc_list = [], [], []
    T = hparam['T']
    alpha_st = hparam['alpha_st']
    alpha_rap = hparam['alpha_rap']
    
    SNet = StudentNet(0,0)
    SNet = SNet.to(device)
    
    TNet.eval()
    for param in TNet.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(SNet.parameters(), lr=hparam['lr'], weight_decay=hparam['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])
    
    def LossFn(teacher_pred, student_pred, y, T, rap_s, rap_t, num, alpha_st, alpha_rap):
        
        loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1), reduction='batchmean') * (T ** 2) * alpha_st + F.cross_entropy(student_pred, y) * (1 - alpha_st - alpha_rap) + F.mse_loss(rap_s,rap_t,reduction='sum')*alpha_rap/num
        #print(F.kl_div(F.log_softmax(student_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1), reduction='batchmean') * (T ** 2) * alpha_st)
        #print(F.mse_loss(rap_s,rap_t,reduction='sum')*alpha_rap/num)
        #print('LLLL')
        return loss

    max_val_acc = float('-inf')
    for epoch in range(num_epochs):
        SNet.train()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(train_loader, 0):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            output_s = SNet(X)
            output_t = TNet(X)
            Ts = compute_pred(output_s) #One Hot Encoded
            
            teacher_corr = (torch.argmax(output_t, dim=1) == y).type(torch.FloatTensor)
            num = torch.sum(teacher_corr).item()
            #Tt = compute_pred(output_t)
            
            RAP = SNet.RAP_relprop(R=Ts)
            RAP = RAP.data.cpu().numpy()
            #print(RAP.shape)
            norm = np.max(np.abs(RAP),axis=(1,2),keepdims=True)
            RAP = RAP/norm
            rap_s = torch.Tensor((1.0+RAP)/2.0)
            
            RAP = TNet.RAP_relprop(R=Ts)
            RAP = RAP.data.cpu().numpy()
            #RAPy = RAP
            norm = np.max(np.abs(RAP),axis=(1,2),keepdims=True)
            RAP = RAP/norm
            rap_t = torch.Tensor((1.0+RAP)/2.0)
            
            rap_s = rap_s*teacher_corr[:,None,None]
            rap_t = rap_t*teacher_corr[:,None,None]
            
            loss = LossFn(output_t,output_s, y, T, rap_s, rap_t, num, alpha_st, alpha_rap)
            loss.backward()
            optimizer.step()
            
            run_acc += torch.sum(torch.argmax(output_s, dim=1) == y).item()
            run_loss += loss.item()*y.size(0)
            total += y.size(0) 
        
        lr_scheduler.step()
        
        train_loss = run_loss/total
        train_acc = 100.0*float(run_acc)/total
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        SNet.eval()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            
            output_s = SNet(X)
            output_t = TNet(X)
            Ts = compute_pred(output_s) #One Hot Encoded
            
            teacher_corr = (torch.argmax(output_t, dim=1) == y).type(torch.FloatTensor)
            num = torch.sum(teacher_corr).item()
            #Tt = compute_pred(output_t)
            
            RAP = SNet.RAP_relprop(R=Ts)
            RAP = RAP.data.cpu().numpy()
            #print(RAP.shape)
            norm = np.max(np.abs(RAP),axis=(1,2),keepdims=True)
            RAP = RAP/norm
            rap_s = torch.Tensor((1.0+RAP)/2.0)
            
            RAP = TNet.RAP_relprop(R=Ts)
            RAP = RAP.data.cpu().numpy()
            #RAPy = RAP
            norm = np.max(np.abs(RAP),axis=(1,2),keepdims=True)
            RAP = RAP/norm
            rap_t = torch.Tensor((1.0+RAP)/2.0)
            
            rap_s = rap_s*teacher_corr[:,None,None]
            rap_t = rap_t*teacher_corr[:,None,None]
            
            loss = LossFn(output_t,output_s, y, T, rap_s, rap_t, num, alpha_st, alpha_rap)
            run_acc += torch.sum(torch.argmax(output_s, dim=1) == y).item()
            run_loss += loss.item()*y.size(0)
            total += y.size(0) 

        val_acc = 100.0*float(run_acc)/total
        val_loss = run_loss/total
        val_acc_list.append(val_acc)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            temp = SNet.state_dict()
        
        print('[%d/%d] train loss: %.3f train accuracy: %.3f || val loss %.3f val accuracy: %.3f' %
          (epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

    
    return ({'train_loss': train_loss_list, 
            'train_acc': train_acc_list, 
            'val_acc': val_acc_list},SNet.state_dict(),temp)


def ToString(hparam):
    """
    Convert hparam dictionary to string with deterministic order of attribute of hparam in output string
    """
    hparam_str = ''
    for k, v in sorted(hparam.items()):
        hparam_str += k + '=' + str(v) + ', '
    return hparam_str[:-2]

def DictToTuple(hparam):
    """
    Convert hparam dictionary to tuple with deterministic order of attribute of hparam in output tuple
    """
    hparam_tuple = [v for k, v in sorted(hparam.items())]
    return tuple(hparam_tuple)

def getTrainMetricPerEpoch(train_metric, updates_per_epoch):
    """
    Smooth the training metric calculated for each batch of training set by averaging over batches in an epoch
    Input: List of training metric calculated for each batch
    Output: List of training matric averaged over each epoch
    """
    train_metric_per_epoch = []
    temp_sum = 0.0
    for i in range(len(train_metric)):
        temp_sum += train_metric[i]
        if (i % updates_per_epoch == updates_per_epoch - 1):
            train_metric_per_epoch.append(temp_sum / updates_per_epoch)
            temp_sum = 0.0

    return train_metric_per_epoch




def hm_to_rgb(R, scaling = 3, cmap = 'bwr', normalize = True):
    cmap = eval('matplotlib.cm.{}'.format(cmap))
    if normalize:
        R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping
    R = R
    R = enlarge_image(R, scaling)
    rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    return rgb
def visualize(relevances, img_name):
    # visualize the relevance
    n = len(relevances)
    heatmap = np.sum(relevances.reshape([n, 224, 224, 1]), axis=3)
    heatmaps = []
    for h, heat in enumerate(heatmap):
        maps = hm_to_rgb(heat, scaling=3, cmap = 'seismic')
        heatmaps.append(maps)
        imageio.imsave('./results/'+ method + '/' + data_name + img_name, maps,vmax=1,vmin=-1)
        
def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    #print('Pred cls : '+str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 1)
    T = (T == np.arange(10)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    T.requires_grad = True
    Tt = T.cuda()
    return Tt





################NOT WORKING YET
def trainRAP_All(TNet,hparam, num_epochs, 
                        train_loader, val_loader, 
                        device):
    train_loss_list, train_acc_list, val_acc_list = [], [], []
    
    SNet = StudentNet(0,0)
    SNet = SNet.to(device)
    
    TNet.eval()
    for param in TNet.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(SNet.parameters(), lr=hparam['lr'], weight_decay=hparam['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])
    
    def LossFn(rap_s, rap_t, num):
        loss = F.mse_loss(rap_s,rap_t,reduction='sum')/float(num)
        return loss

    max_val_acc = float('-inf')
    for epoch in range(num_epochs):
        SNet.train()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(train_loader, 0):
            X, y = X.to(device), y.to(device)
            
            loss_list = torch.zeros(10,requires_grad=False)
            #loss = torch.empty(1,requires_grad=False)
            optimizer.zero_grad()
            
            output_s = SNet(X)
            output_t = TNet(X)
            
            teacher_corr = (torch.argmax(output_t, dim=1) == y).type(torch.FloatTensor)
            num = torch.sum(teacher_corr).item()
            
            Ts = torch.zeros([10,y.size(0),10])
            Ts.requires_grad = True
            Ts = Ts.to(device)
            RAP = torch.zeros([10,28,28])        
            for it in range(10):
                Ts[it] = torch.zeros([y.size(0),10]) #One Hot Encoded
                Ts[it,:,it] =torch.ones([y.size(0)])
                
                RAP[it] = SNet.RAP_relprop(R=Ts[it])
                RAP = RAP.data.cpu().numpy()
                norm = np.max(np.abs(RAP[it]),axis=(1,2),keepdims=True)
                RAP = RAP/norm
                rap_s = torch.Tensor((1.0+RAP)/2.0)

                RAP = TNet.RAP_relprop(R=Ts)
                RAP = RAP.data.cpu().numpy()
                norm = np.max(np.abs(RAP),axis=(1,2),keepdims=True)
                RAP = RAP/norm
                rap_t = torch.Tensor((1.0+RAP)/2.0)

                rap_s = rap_s*teacher_corr[:,None,None]
                rap_t = rap_t*teacher_corr[:,None,None]

                loss_list[it] = LossFn(rap_s, rap_t, num)/10.0
            
            
            #print(real_loss.requires_grad)
            loss = torch.sum(loss_list)
            print(loss)
            loss.backward()
            optimizer.step()
            
            run_acc += torch.sum(torch.argmax(output_s, dim=1) == y).item()
            run_loss += loss.item()*y.size(0)
            total += y.size(0) 
        
        lr_scheduler.step()
        
        train_loss = run_loss/total
        train_acc = 100.0*float(run_acc)/total
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        SNet.eval()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(val_loader):
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                
                output_s = SNet(X)
            
                run_acc += torch.sum(torch.argmax(output_s, dim=1) == y).item()
                #run_loss += loss.item()*y.size(0)
                total += y.size(0) 

        val_acc = 100.0*float(run_acc)/total
        
        val_acc_list.append(val_acc)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            sn = deepcopy(SNet)
            temp = sn.state_dict()
            print('[%d/%d] train loss: %.3f train accuracy: %.3f || val accuracy: %.3f' %
              (epoch + 1, num_epochs, train_loss, train_acc, val_acc))
            
    
    return ({'train_loss': train_loss_list, 
            'train_acc': train_acc_list, 
            'val_acc': val_acc_list},SNet.state_dict(),temp)


def trainRAP2(TNet,hparam, num_epochs, 
                        train_loader, val_loader, 
                        device):
    train_loss_list, train_acc_list, val_acc_list = [], [], []

    SNet = StudentNet(0,0)
    SNet = SNet.to(device)
    
    TNet.eval()
    for param in TNet.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(SNet.parameters(), lr=hparam['lr'], weight_decay=hparam['weight_decay'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])
    
    def LossFn(student_pred, y, rap_s, rap_t, num, epoch):
        alpha_ce = (0.7)*pow(0.96,epoch)
        loss = F.cross_entropy(student_pred, y)*alpha_ce + F.mse_loss(rap_s,rap_t,reduction='sum')*(1-alpha_ce)/num
        return loss

    max_val_acc = float('-inf')
    for epoch in range(num_epochs):
        SNet.train()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(train_loader, 0):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            output_s = SNet(X)
            output_t = TNet(X)
            Ts = compute_pred(output_s) #One Hot Encoded
            
            teacher_corr = (torch.argmax(output_t, dim=1) == y).type(torch.FloatTensor)
            num = torch.sum(teacher_corr).item()
            #Tt = compute_pred(output_t)
            
            RAP = SNet.RAP_relprop(R=Ts)
            RAP = RAP.data.cpu().numpy()
            #print(RAP.shape)
            norm = np.max(np.abs(RAP),axis=(1,2),keepdims=True)
            RAP = RAP/norm
            rap_s = torch.Tensor((1.0+RAP)/2.0)
            
            RAP = TNet.RAP_relprop(R=Ts)
            RAP = RAP.data.cpu().numpy()
            #RAPy = RAP
            norm = np.max(np.abs(RAP),axis=(1,2),keepdims=True)
            RAP = RAP/norm
            rap_t = torch.Tensor((1.0+RAP)/2.0)
            
            rap_s = rap_s*teacher_corr[:,None,None]
            rap_t = rap_t*teacher_corr[:,None,None]
            
            loss = LossFn(output_s, y,rap_s, rap_t, num, epoch)
            loss.backward()
            optimizer.step()
            
            run_acc += torch.sum(torch.argmax(output_s, dim=1) == y).item()
            run_loss += loss.item()*y.size(0)
            total += y.size(0) 
        
        lr_scheduler.step()
        
        train_loss = run_loss/total
        train_acc = 100.0*float(run_acc)/total
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        SNet.eval()
        run_acc = 0
        run_loss = 0.0
        total = 0
        for i, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            
            output_s = SNet(X)
            output_t = TNet(X)
            Ts = compute_pred(output_s) #One Hot Encoded
            
            teacher_corr = (torch.argmax(output_t, dim=1) == y).type(torch.FloatTensor)
            num = torch.sum(teacher_corr).item()
            #Tt = compute_pred(output_t)
            
            RAP = SNet.RAP_relprop(R=Ts)
            RAP = RAP.data.cpu().numpy()
            #print(RAP.shape)
            norm = np.max(np.abs(RAP),axis=(1,2),keepdims=True)
            RAP = RAP/norm
            rap_s = torch.Tensor((1.0+RAP)/2.0)
            
            RAP = TNet.RAP_relprop(R=Ts)
            RAP = RAP.data.cpu().numpy()
            #RAPy = RAP
            norm = np.max(np.abs(RAP),axis=(1,2),keepdims=True)
            RAP = RAP/norm
            rap_t = torch.Tensor((1.0+RAP)/2.0)
            
            rap_s = rap_s*teacher_corr[:,None,None]
            rap_t = rap_t*teacher_corr[:,None,None]
            
            loss = LossFn(output_s, y, rap_s, rap_t, num, epoch)
            run_acc += torch.sum(torch.argmax(output_s, dim=1) == y).item()
            run_loss += loss.item()*y.size(0)
            total += y.size(0) 

        val_acc = 100.0*float(run_acc)/total
        val_loss = run_loss/total
        val_acc_list.append(val_acc)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            temp = SNet.state_dict()
        
        print('[%d/%d] train loss: %.3f train accuracy: %.3f || val loss %.3f val accuracy: %.3f' %
          (epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc))

    
    return ({'train_loss': train_loss_list, 
            'train_acc': train_acc_list, 
            'val_acc': val_acc_list},SNet.state_dict(),temp)
