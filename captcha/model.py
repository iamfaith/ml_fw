import torch
import torch.nn as nn
import torch, time
from torchvision.models.resnet import ResNet, BasicBlock

class MnistResNet(ResNet):
    def __init__(self, num_classes=36):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
        return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)



train_loader, val_loader, train_samples, val_samples = train_val_split(r'/content/label/*png', batch_size=64)


model = MnistResNet(36).cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epoch = 5

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        # print(data[0].size(), data[1])  #, data[1].numpy())
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數


        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值
        
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
        
        # print(np.argmax(train_pred.cpu().data.numpy(), axis=1))
        

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
            train_acc/(len(train_samples) * 4), train_loss/(len(train_samples) * 4), val_acc/(len(val_samples) * 4), val_loss/(len(val_samples) * 4)))
        
        
        
test_dataset = IterableDataset(val_samples[:20], has_label=False)
# test_dataset = IterableDataset(test_samples[:10], has_label=False)
test_loader = DataLoader(test_dataset, batch_size=4)

model.eval()
print(test_samples[:10])
with torch.no_grad():
    for i, data in enumerate(test_loader):
        prediction = []
        # print(i, data.size()[0])
        for i in range(data.size()[0]):
          cv2_imshow(data[i].numpy().transpose(1,2,0))
        test_pred = model(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)
        
        print([ values.get(p)   for p in prediction] )