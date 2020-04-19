import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from PIL import Image
from torchvision import datasets, transforms
from Model import ConvNet



# some constants
N = 28
BATCHSIZE = 64
NUM_EXAMPLES = 10


test_data = datasets.MNIST('data', train=False, download=True,
															transform=transforms.Compose([
																transforms.ToTensor(),
																transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = torch.utils.data.DataLoader(test_data,batch_size=BATCHSIZE, shuffle=True) 


# loading model onto cuda device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on', device)
print('Building model..')
model = ConvNet()
model.train(False)
model.to(device)
model.load_state_dict(torch.load('Model_best_val_quicksave.pt'))
print('Model Built.')

def build_relevance_map(x, num_iter=10000, lr=1e-3, lamb=750):
	"""
	:param x: ndarray(1, N, N) of float32; image as given by batch-loader
	:param num_iter: int; number of optimization iterations
	:param lr: float; step size for optimization (lr is for learning rate as I just pass it to Adam)
	:param lamb: float; optimization parameter in minimization E[(Phi(W**(-1)(s*Wx+(1-s)*Wn))-Phi(x))**2]+lamb*||s||_1
	:return: ndarray(1, N, N) of float32; relevances of transformation components
	"""
	# if we dont have the transformation matrizes given, just do the normal optimization

	# initializing s and torch optimizer for optimization of s
	s = 0.5*np.ones((1,N,N))
	s = torch.as_tensor(s.astype(np.float32)).to(device)

	# I think this can be omitted in newer versions of pytorch but I left it in just to be sure
	s = torch.autograd.Variable(s, requires_grad=True)
	optimizer = torch.optim.Adam([s], lr=lr)

	# In the RDF-paper, optimization is done with respect to the output of the network in the highest component
	# (, which coincides with the label class if classified correctly) before SoftMax application (the reason
	# that SoftMax is not applied in the Network itself)
	x_input = x.to(device)
	x_out = model(x_input.clone()).detach()
	highest_dim = int(np.argmax(x_out.cpu().numpy(), axis=1))

	# optimization loop
	for i in range(num_iter//BATCHSIZE):
		# n is random gaussian noise
		n = torch.as_tensor(np.random.normal(size=(BATCHSIZE, *x[0].shape)).astype(np.float32)).to(device)
		data_input = (x_input-n)*s+n
		#print(x_input.shape, data_input.shape, n.shape)
		# for the l1-regularization, I use the mean, adjust lamb if you want to use sum
		out = model(data_input)
		loss = 0.5*torch.mean((out[:, highest_dim]-x_out[:, highest_dim])**2)+lamb*torch.mean(torch.abs(s))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# ensuring the [0,1]-constraint
		# this += thing was a weird hack to not lose Variable status of s for pytorch. s.data().clamp_ would probably
		# also work... but ya know... never touch a running system
		with torch.no_grad():
			s += s.clamp_(0,1)-s

		# I print my loss to see whats happening.. not very sophisticated but whatever
		if (i+1)%100==0:
			print(loss.data.item())
			#print(s.data)
			#print(0.5*torch.mean((out[:, highest_dim]-x_out[:, highest_dim])**2))
			
	return s.detach().cpu().numpy()


# some seed so I get the same random examples.. I change the seed when I want to see different ones
np.random.seed(100)

# I have to redo images which are not classified correctly, so I use this counter to keep track of iterations
counter = 0
while counter < NUM_EXAMPLES:
	k = np.random.randint(10000)
	image, target = test_data[k]
	#print(image.shape)
	#print(image[None,:,:,:].shape)	
	#print(type(target))
	#print(image)
	image = image[None,:,:,:]

	x_input = image.to(device)
	x_out = model(x_input.clone()).detach().cpu()
	
	if np.argmax(x_out) != target: continue

	print(x_out)
	rel_map_normal = build_relevance_map(image, num_iter=200000, lr=1e-3, lamb=0.1)
	#print(rel_map_normal)
	# save the results in a dict to analyze later
	sample_dict =  {'img': image[0, :, :, :],
					'relevance_map_normal': rel_map_normal,
					'model_prediction': x_out}
	np.save('SampleDicts/sample_dict_'+str(counter), sample_dict)
	counter += 1