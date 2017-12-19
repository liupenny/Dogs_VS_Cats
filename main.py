#coding:utf8
from config import opt
import os
import torch as t
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
# from utils.visualize import Visualizer
from tqdm import tqdm
import torch
import ipdb;

# 设置GPU
torch.cuda.set_device(4)

def test(**kwargs):
    opt.parse(kwargs)
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # 加载测试集数据
    train_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []

    # 开始测试
    for ii,(data,path) in enumerate(test_dataloader):
        input = t.autograd.Variable(data,volatile = True)
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score)[:,0].data.tolist()
        batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability) ]
        results += batch_results
    write_csv(results,opt.result_file)
    return results

# 先将结果按照id排序，因为算出来的是分类的概率，再根据概率进行转换。
def write_csv(results,file_name):
    import csv
    results.sort()    
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow([row[0], 1 if row[1] > 0.50000 else 0])


def train(**kwargs):
    opt.parse(kwargs)
    bench = 60.00000
 
    # 第一步: 设置模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # 第二步: 加载数据
    train_data = DogCat(opt.train_data_root,train=True)
    val_data = DogCat(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    # 第三步: 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)
        
    # 第四步: 统计指标，平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 开始训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        # 因为一次读取128张图片，所以需要轮回69次，+1。tqdm是进度条显示
        for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data)/128+1):
            # 训练模型
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()
            
            # 更新统计指标
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)

        
        # 计算验证集上的指标，如果比上次的准确率高就保存当前模型和准确率
        val_cm,val_accuracy = val(model,val_dataloader)
        if val_accuracy >= bench:
            model.save()
            bench = val_accuracy
        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        previous_loss = loss_meter.value()[0]

def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(t.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    print "------------------------------------accuracy = " + str(accuracy) + "-------------------------"
    return confusion_matrix, accuracy

def help():
    '''
    打印帮助的信息： python file.py help
    '''
    
    print('''

    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()
