import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import time

from Model import ConvNet


import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


BATCHSIZE = 128
NUMEPOCHS = 50
LEARNINGRATE = 5e-4
NUM_DATA_POINTS = 8000
NUM_VAL_POINTS = 1000


train_loader = torch.utils.data.DataLoader(
										datasets.MNIST('data', train=True, download=True,
															transform=transforms.Compose([
																transforms.ToTensor(),
																transforms.Normalize((0.1307,), (0.3081,))])),
										batch_size=BATCHSIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
										datasets.MNIST('data', train=False, download=True,
															transform=transforms.Compose([
																transforms.ToTensor(),
																transforms.Normalize((0.1307,), (0.3081,))])),
										batch_size=BATCHSIZE, shuffle=True) 

# initializing everything
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model = ConvNet()
model.to(device)
print('Model Built.')

print('Initializing optimizer..')
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNINGRATE)
print('Optimizer initialized.')

# using this variable for best validation saving
min_val_loss = float('inf')
for epoch in range(1,NUMEPOCHS+1):
	start_time = time.time()
	model.train()
	# training loop, pretty self explanatory
	for i, (data_input, data_output) in enumerate(train_loader):
		
		data_input = data_input.to(device)
		data_output = data_output.to(device)
		
		output = model(data_input)
		
		loss = criterion(output, data_output)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# validating on all validation samples (slower than just one batch, but its a small model)
	val_loss = 0
	model.eval()
	corr = 0
	with torch.no_grad():
		for i,(data_input,data_output) in enumerate(test_loader):
			
			
			data_input = data_input.to(device)
			data_output = data_output.to(device)

			output = model(data_input)

			val_loss += criterion(output, data_output).data.item()
			pred = output.argmax(dim=1,keepdim=True)
			corr += pred.eq(data_output.view_as(pred)).sum().item()

	# saving model if validation loss is lowest
	val_loss /= len(test_loader.dataset)

	if val_loss <= min_val_loss:
		min_val_loss = val_loss
		torch.save(model.state_dict(), 'Model_best_val_quicksave.pt')

	# printing some stuff
	stop_time = time.time()
	time_el = int(stop_time-start_time)
	print('epoch [{}/{}], loss:{:.7f}, val loss:{:.7f}, Acc {:.5f} in {}h {}m {}s'.format(epoch, NUMEPOCHS,
																			  loss.data.item(), val_loss,
																			  100.*corr/len(test_loader.dataset),
																			  time_el//3600,
																			  (time_el%3600)//60,
																			  time_el%60))
	# quicksaving model every epoch if something goes wrong and I want to continue training later
	torch.save(model.state_dict(), 'Model_quicksave.pt')
