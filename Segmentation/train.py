import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
from utils.mypath import Path
from dataloaders.dataloader import make_dataset
from models.sync_batchnorm.replicate import patch_replication_callback
from models import build_model
from loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from torch.utils.data import DataLoader
from utils.saver import Saver

class Trainer(object):
    def __init__(self, args,model,train_loader,val_loader):
        self.args = args
        self.nclass=3

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        if args.cuda:
            self.device='cuda'
        else:
            self.device='cpu'

        self.dtype=torch.float32

        # Define Dataloader
        self.train_loader=train_loader
        self.val_loader=val_loader
        # self.train_loader, self.val_loader, self.test_loader= make_data_loader(args)

        # # Define network
        model=model

        # Define Optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
        #                             weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion1 = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='ce')
        self.criterion2 = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='iou')

        self.model, self.optimizer = model, optimizer
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))
        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        patch_replication_callback(self.model)
        self.model=self.model.to(device=self.device)

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint['model'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        if args.ft:
            args.start_epoch = 0      

    def training(self, epoch):
        train_loss=0.0
        self.model.train()
        self.evaluator.reset()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i,(slice_img,slice_label) in enumerate(tbar):
            slice_img, slice_label = slice_img.to(device=self.device,dtype=self.dtype), slice_label.to(device=self.device)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            scores=self.model(slice_img)
            loss1 = self.criterion1(scores, slice_label)
            loss2 = self.criterion2(scores, slice_label)
            loss = loss1+0.4*loss2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss+=loss.item()

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            N,_,h,w=scores.shape
            label_pred=scores.max(dim=1)[1].cpu().numpy()
            label_true=slice_label.cpu().numpy()

            self.evaluator.add_batch(label_true,label_pred)
        train_accuracy=self.evaluator.Pixel_Accuracy_Mean()
        train_miou=self.evaluator.Mean_Intersection_over_Union()

        str_train='Train_Loss:'+str(np.round(train_loss/len(self.train_loader),5))+' '\
                  +'Train_acc:'+str(np.round(train_accuracy,5))+' '\
                  +'Train_miou:'+str(np.round(train_miou,5))+' '

        print(str_train)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def val(self, epoch):
        val_loss=0.0
        self.model.eval()
        self.evaluator.reset()
        vbar = tqdm(self.val_loader, desc='\r')

        for i,(slice_img,slice_label) in enumerate(vbar):
            slice_img, slice_label = slice_img.to(device=self.device,dtype=self.dtype), slice_label.to(device=self.device)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            scores=self.model(slice_img)
            loss1 = self.criterion1(scores, slice_label)
            loss2 = self.criterion2(scores, slice_label)
            loss = loss1+0.4*loss2

            val_loss+=loss.item()

            N,_,h,w=scores.shape
            label_pred=scores.max(dim=1)[1].cpu().numpy()
            label_true=slice_label.cpu().numpy()
            vbar.set_description('Val loss: %.3f' % (val_loss / (i + 1)))

            self.evaluator.add_batch(label_true,label_pred)

        val_accuracy=self.evaluator.Pixel_Accuracy_Mean()
        val_miou=self.evaluator.Mean_Intersection_over_Union()

        str_val='Val_Loss:'+str(np.round(val_loss/len(self.val_loader),5))+' '\
                  +'Val_acc:'+str(np.round(val_accuracy,5))+' '\
                  +'Val_miou:'+str(np.round(val_miou,5))+' '

        print(str_val)

        new_pred = val_miou
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="Coral Training")
    parser.add_argument('--model', type=str, default='deeplabv3p',
                        choices=['deeplab'],
                        help='model name (default: deeplabv3p)')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet_rgbd','xception'],
                        help='model name (default: deeplabv3p)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['plot18_2019','plot18_2019_pocill'],
                        help='dataset name (default: plot18_2019)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--val_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                val (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, #牛顿动量法
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--kfoldval', type=bool, default=False,
                        help='k fold validation')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'plot18_2019': 200,
            'plot18_2019_pocill': 200
        }
        args.epochs = epoches[args.dataset.lower()] #lower转换成小写

    if args.train_batch_size is None:
        args.train_batch_size = 8 * len(args.gpu_ids)

    if args.val_batch_size is None:
        args.val_batch_size = args.train_batch_size/4

    if args.test_batch_size is None:
        args.test_batch_size = args.gpu_ids

    if args.lr is None:
        lrs = {
            'deeplabv3p': 0.001
        }
        args.lr = lrs[args.model.lower()]


    if args.checkname is None:
        args.checkname = str(args.model)

    if args.kfoldval:
        k=5
    
    print(args)
    torch.manual_seed(args.seed) #设置CPU生成随机数的种子，方便下次复现实验结果。
    for ki in range(k):
        print("{} fold".format(ki))
        # Define network
        model = build_model(args)

        train_set = make_dataset(txt=os.path.join(Path.db_root_dir(args.dataset),'train'+str(ki)+'.txt'))
        val_set = make_dataset(txt=os.path.join(Path.db_root_dir(args.dataset),'val'+str(ki)+'.txt'))
        train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=True,drop_last=True)
        trainer = Trainer(args,model,train_loader,val_loader)
        print('Starting Epoch:', trainer.args.start_epoch)
        print('Total Epoches:', trainer.args.epochs)
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
                trainer.val(epoch)
        state = {'model':model.state_dict(), 'epoch':epoch}
        torch.save(state, os.path.join('./run', args.dataset, args.backbone, args.checkname,'kcv'+str(ki)+'.pth'))    

    trainer.writer.close()

if __name__ == "__main__":
   main()