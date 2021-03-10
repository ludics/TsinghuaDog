import jittor as jt 
import jittor.nn as nn 
from dataset import TsinghuaDog, TsinghuaDogTest
from jittor import transform
from jittor.optim import Adam, SGD
from tqdm import tqdm
import numpy as np
from model import Net
import argparse 
import json
import os
import os.path as osp


jt.flags.use_cuda=1

def infer(model, test_loader, topK=5, save_path='./best_model.pkl'):
    model.eval()
    result_dict = {}
    for images, names in tqdm(test_loader):
        output = model(images)
        pred = output.data.argsort(axis=1)[:, ::-1] + 1
        for idx, name in enumerate(names):
            result_dict[name] = pred[idx, :topK].tolist()
    return result_dict

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=130)

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)

    parser.add_argument('--dataroot', type=str, default='/home/gmh/dataset/TsinghuaDog/')
    parser.add_argument('--model_path', type=str, default='./best_model.pkl')
    parser.add_argument('--result_path', type=str, default='../result/')

    args = parser.parse_args()
    
    transform_test = transform.Compose([
        transform.Resize((512, 512)),
        transform.CenterCrop(448),
        transform.ToTensor(),
        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_loader = TsinghuaDogTest(args.dataroot, batch_size=args.batch_size, shuffle=False, transform=transform_test)
    model = Net(num_classes=args.num_classes)
    model.load(args.model_path)
    result_dict = infer(model, test_loader)
    json_str = json.dumps(result_dict, indent=4)
    with open(osp.join(args.result_path, 'result.json'), 'w') as f:
        f.write(json_str)


if __name__ == '__main__':
    main()

