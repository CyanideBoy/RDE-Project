import torch
import torch.optim as optim
import numpy as np
from data_loader import train_data
from model import ConvNet
import matplotlib.pyplot as plt
import time


BATCHSIZE = 256
NUMEPOCHS = 45
LEARNINGRATE = 1e-2


train_x, train_y = train_data()

NUM_DATA_POINTS = int(0.8*len(train_x))
NUM_VAL_POINTS = len(train_x) - NUM_DATA_POINTS

val_x = train_x[NUM_DATA_POINTS:]
train_X = train_x[:NUM_DATA_POINTS]

val_y = train_y[NUM_DATA_POINTS:]
train_y = train_y[:NUM_DATA_POINTS]

model = ConvNet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on',device)
print('Building model..')	
model.to(device)
print('Model Built.')

print('Initializing optimizer and scheduler..')

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNINGRATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)

print('Optimizer and scheduler initialized.')

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

model.apply(weights_init)

print('Printing Parameters')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
print('---Printing Parameters Finished!---')



min_val_loss = float('inf')

loss_values_train = []
loss_values_val = []

for epoch in range(1,NUMEPOCHS+1):
    start_time = time.time()
    model.train()
    runloss = 0.0
    # training loop, pretty self explanatory
    for i in range(NUM_DATA_POINTS//BATCHSIZE):
        data_input, data_output = train_x[i*BATCHSIZE:(i+1)*BATCHSIZE], train_y[i*BATCHSIZE:(i+1)*BATCHSIZE]
        
        data_input = torch.as_tensor(data_input)
        data_output = torch.as_tensor(data_output)
        
        data_input = data_input.to(device, dtype=torch.float)
        data_output = data_output.to(device, dtype=torch.long)
        
        output = model(data_input) 
        #print(output, data_output)
        loss = criterion(output, data_output)
        runloss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    runloss /= (NUM_DATA_POINTS//BATCHSIZE)
    loss_values_train.append(runloss)
    
    model.eval()
    
    val_loss = 0
    corr = 0
    with torch.no_grad():
        for i in range(NUM_VAL_POINTS//BATCHSIZE):
            val_data_input, val_data_output = val_x[i*BATCHSIZE:(i+1)*BATCHSIZE], val_y[i*BATCHSIZE:(i+1)*BATCHSIZE]

            data_input = torch.as_tensor(val_data_input)
            data_output = torch.as_tensor(val_data_output)

            data_input = data_input.to(device, dtype=torch.float)
            data_output = data_output.to(device, dtype=torch.long)

            output = model(data_input)
            pred = output.argmax(dim=1,keepdim=True)
            corr += pred.eq(data_output.view_as(pred)).sum().item()

            loss = criterion(output, data_output)
            val_loss += loss.data.item()
            
    # saving model if validation loss is lowest
    val_loss /= NUM_VAL_POINTS//BATCHSIZE
    loss_values_val.append(val_loss)

    if val_loss <= min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'Model_best_val_quicksave.pt')

    # printing some stuff
    stop_time = time.time()
    time_el = int(stop_time-start_time)
    print('epoch [{}/{}], loss:{:.7f}, val loss:{:.7f}, val acc:{:.7f} in {}h {}m {}s'.format(epoch, NUMEPOCHS,
                                                                                runloss, val_loss, 100*corr/(BATCHSIZE*(NUM_VAL_POINTS//BATCHSIZE)),
                                                                                time_el//3600,
                                                                                (time_el%3600)//60,
                                                                                time_el%60))
    # quicksaving model every epoch if something goes wrong and I want to continue training later
    torch.save(model.state_dict(), 'Model_quicksave.pt')

    scheduler.step()

plt.plot(np.array(loss_values_train), 'b')
plt.plot(np.array(loss_values_val), 'r')
plt.legend(['Train','Val'])
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
