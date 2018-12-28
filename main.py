from helper import *

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307, ), (0.3081, ))])

train_loader = DataLoader(torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transforms), batch_size = 64, shuffle = True)

test_loader = DataLoader(torchvision.datasets.MNIST(root = './data', train = False, transform = transforms), batch_size = 64, shuffle = True)

model = STCNN().to(device)
epochs = 10
learning_rate = 1e-4

optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

# train the model
print("Training:")
t_start = time.time()
for epoch in range(epochs):
    print("epoch {}".format(epoch))
    t_start_epoch = time.time()
    for batch_ix, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optim.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)

        loss.backward()
        optim.step()

        if batch_ix % 500 == 0:
            print("Batch {}, loss = {}".format(batch_ix, loss))
    t_end_epoch = time.time()
    print("epoch {} took {} seconds".format(epoch, t_end_epoch - t_start_epoch))
t_end = time.time()
print("Training took {} seconds".format(t_end - t_start))

# test the model
print("Testing:")
with torch.no_grad():
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        test_loss += F.nll_loss(y_pred, y, size_average = False).item()
        pred = y_pred.max(1, keepdim = True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print("Average loss of {} with accuracy of {}".format(test_loss, correct / len(test_loader.dataset)))
