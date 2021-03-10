import jittor as jt 
import jittor.nn as nn 
from dataset import TsinghuaDog
from jittor import transform
from jittor.optim import Adam, SGD
from tqdm import tqdm
import numpy as np
from model import Net
import argparse 
from dataset import transform_train, transform_test
import yaml
from utils.utils import AverageMeter, initial_logger
import time
import os
import os.path as osp

jt.flags.use_cuda=1

def train(model, train_loader, optimizer, epoch, param):
    model.train()
    logger = param['logger']
    epoch_loss, iter_loss = AverageMeter(), AverageMeter()
    acc = AverageMeter()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    if epoch in [20, 25, 30, 35, 40, 45]:
        optimizer.lr *= 0.5
    for idx, (images, labels) in enumerate(pbar):
        output = model(images)
        loss = nn.cross_entropy_loss(output, labels)
        optimizer.step(loss) 
        pred = np.argmax(output.data, axis=1)
        acc.update(np.mean(pred == labels.data) * 100)
        epoch_loss.update(loss.data[0])
        iter_loss.update(loss.data[0])
        if idx % param['iter_inter'] == 0:
            logger.info('[train] epoch:{} iter:{}/{} lr:{:.6f} loss:{:.6f} acc:{:.4f}'.format(
                epoch, idx, train_loader.__len__(), optimizer.lr, iter_loss.avg, acc.avg
            ))
            iter_loss.reset()
        pbar.set_description(f'Epoch {epoch} [TRAIN] lr = {optimizer.lr:.6f}, loss = {epoch_loss.avg:.6f}, acc = {acc.avg:.4f}')

def evaluate(model, val_loader, param, epoch=0):
    model.eval()
    logger = param['logger']
    
    total_acc = 0
    total_num = 0

    for images, labels in val_loader:
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0] 
    acc = total_acc / total_num 
    if acc > param['best_acc']:
        param['best_acc'] = acc
        model.save(osp.join(param['save_ckpt_dir'], 'model_best.pkl'))
    if epoch > param['min_inter'] and epoch % param['save_inter'] == 0:
        model.save(osp.join(param['save_ckpt_dir'], f'model_epoch{epoch}.pkl'))
    logger.info(f'Test in epoch {epoch} Accuracy is {acc:.4f} Best accuracy is {param["best_acc"]:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=130)

    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--backbone', type=str, default='Resnet50')
    parser.add_argument('--eval', type=bool, default=False)

    parser.add_argument('--save_dir', type=str, default='../outputs', help='the dir to save logs and models')
    parser.add_argument('--data_root', type=str, default='/home/gmh/dataset/TsinghuaDog/')
    parser.add_argument('--model_path', type=str, default='./best_model.pkl')
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help='config file name')

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    param = config['train']
    
    root_dir = args.data_root
    train_loader = TsinghuaDog(root_dir, batch_size=param['batch_size'], train=True, part='train', shuffle=True, transform=transform_train)
    val_loader = TsinghuaDog(root_dir, batch_size=param['batch_size'], train=False, part='val', shuffle=False, transform=transform_test)
    
    model = Net(num_classes=args.num_classes)
    
    epochs = param['epochs']
    lr = param['lr']
    weight_decay = param['weight_decay']
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) 
    save_log_dir = osp.join(args.save_dir, '_'.join([args.backbone, f'bs{param["batch_size"]}', f'lr{param["lr"]}']))
    param['save_ckpt_dir'] = os.path.join(save_log_dir, 'ckpt')
    if not osp.exists(param['save_ckpt_dir']): os.makedirs(param['save_ckpt_dir'])
    logger = initial_logger(os.path.join(save_log_dir, time.strftime("%m-%d-%H:%M:%S", time.localtime()) + '.log'))
    param['logger'] = logger
    param['best_acc'] = -1
    if args.resume_path:
        model.load(args.resume_path)
    if args.eval:
        evaluate(model, val_loader, param)
        return 
    train_loader_size = len(train_loader)
    valid_loader_size = len(val_loader)
    logger.info(f'Total Epoch:{param["epochs"]} Training num:{train_loader_size} Valid num:{valid_loader_size}')
    evaluate(model, val_loader, param)
    for epoch in range(args.epoch_start, epochs):
        train(model, train_loader, optimizer, epoch, param)
        evaluate(model, val_loader, param, epoch)


if __name__ == '__main__':
    main()

