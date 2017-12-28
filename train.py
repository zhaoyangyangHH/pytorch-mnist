import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import model

iteration = 10
batch = 100

transform = transforms.Compose([transforms.ToTensor()

# 训练集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（50000张图片作为训练数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=False, transform=transform)
print(len(trainset))

# 将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。shffule=True在表示不同批次的数据遍历时，打乱顺序。num_workers=2表示使用两个子进程来加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=2)
print(len(trainloader))

# 测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=False, transform=transform)

# 将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False, num_workers=2)
net = model.LeNet5()

optimizer = optim.SGD(net.parameters(), lr=0.001)
# 使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9, momentum=0.9

criterion = nn.CrossEntropyLoss(size_average=False)
# 叉熵损失函数


for epoch in range(iteration):
    # 遍历数据集两次

    for step, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
    # data的结构是：[4x3x32x32的张量,长度4的张量]

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
    # 把input数据从tensor转为variable

        # zero the parameter gradients
        optimizer.zero_grad()
    # 将参数的grad值初始化为0

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
         # 将output和labels使用叉熵计算损失
        loss.backward()
        # 反向传播
        optimizer.step()
        # 用SGD更新参数
        if step % 200 == 199:
            correct = 0
            total = 0
            for data in testloader:
                images, labels_test = data
                output_test = net(Variable(images))
                # out = F.softmax(output_test)
                # print outputs.data
                _, predicted = torch.max(output_test.data,
                                         1)
                # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
                total += labels_test.size(0)
                correct += (predicted == labels_test).sum()
                # 两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
            accuracy = correct / total
            print('Epoch:', epoch+1, '|Step:', step+1,
                 '|train loss: %.4f' % loss.data[0], '|test accuracy: %.4f' % accuracy)

torch.save(net, 'model.pkl')

print('Finished')

