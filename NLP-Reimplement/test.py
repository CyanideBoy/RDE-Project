import torch
from data_loader import test_data
from model import ConvNet

BATCHSIZE = 256

test_x, test_y = test_data()

model = ConvNet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model.to(device)
print('Model Built.')

model.load_state_dict(torch.load('Model_best_val_quicksave.pt'))
model.eval()

NUM_TEST_POINTS = len(test_x)

tloss = 0
corr = 0
with torch.no_grad():
    for i in range(NUM_TEST_POINTS//BATCHSIZE):
        test_data_input, test_data_output = test_x[i*BATCHSIZE:(i+1)*BATCHSIZE], test_y[i*BATCHSIZE:(i+1)*BATCHSIZE]

        data_input = torch.as_tensor(test_data_input)
        data_output = torch.as_tensor(test_data_output)

        data_input = data_input.to(device, dtype=torch.float)
        data_output = data_output.to(device, dtype=torch.long)

        output = model(data_input)
        pred = output.argmax(dim=1,keepdim=True)
        corr += pred.eq(data_output.view_as(pred)).sum().item()

print('Test Accuracy: {:.2f}%'.format(100 * corr/(BATCHSIZE*(NUM_TEST_POINTS//BATCHSIZE))))