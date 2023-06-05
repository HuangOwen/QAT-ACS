from .. import nets
from .earlytrain import EarlyTrain
from .coresetmethod import CoresetMethod
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel
import torch.nn.functional as F
from datetime import datetime
import os
import math
from copy import deepcopy

class ACS(CoresetMethod):
    def __init__(self, dst_train, args, current_epoch, current_model, fraction=0.5, random_seed=None, balance=True, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.model = current_model
        self.current_epoch = current_epoch
        self.dst_train = deepcopy(dst_train)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.balance = balance
        self.torchvision_pretrain = True if self.args.dataset == "ImageNet" else False
        self.specific_model = None

        if self.args.model == 'QResNet18':
            self.full_precision_model = 'ResNet18'
        elif self.args.model == 'QMobilenetv2':
            self.full_precision_model = 'Mobilenetv2'

        self.save_path = '/home/xhuangbs/DeepCore/score/score_check_resnet_cifar100'
        if self.args.dataset == "ImageNet":
            self.num_classes = 1000 
        elif self.args.dataset == "CIFAR100":
            self.num_classes = 100
        else:
            self.num_classes = 10
        

    def select(self, **kwargs):
        self.train_indx = np.arange(self.n_train)

        #filename = os.path.join(self.save_path, 'el2n_scores_2023-03-14 18:44:45.873900.csv')
        #self.norm_mean = np.loadtxt(filename, delimiter=',')
        
        # Initialize a matrix to save norms of each sample 
        self.norm_matrix = torch.zeros([self.n_train], requires_grad=False).to(self.args.device)

        #self.el2n = torch.zeros([self.n_train], requires_grad=False).to(self.args.device)
        #self.disagreement = torch.zeros([self.n_train], requires_grad=False).to(self.args.device)

        # load the target (quantized) model 
        '''
        self.model = nets.__dict__[self.args.model if self.specific_model is None else self.specific_model](
                self.args.bitwidth, self.args.channel, self.num_classes, pretrained=self.torchvision_pretrain,
                im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)
        '''
        # load the full-precision model
        if self.args.dataset == "ImageNet":
            self.fp_model = nets.__dict__[self.full_precision_model](
                self.args.channel, self.num_classes, pretrained=self.torchvision_pretrain,
                im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)
        elif self.args.dataset == "CIFAR10" or self.args.dataset == "CIFAR100":
            self.fp_model = nets.__dict__[self.full_precision_model](
                self.args.channel, self.num_classes, im_size=self.args.im_size).to(self.args.device)
            checkpoint = torch.load(self.args.resume, map_location=self.args.device)
            # Loading model state_dict
            self.fp_model.load_state_dict(checkpoint["state_dict"],strict=True)
        
        if self.args.device == "cpu":
            print("Using CPU.")
        elif self.args.gpu is not None:
            torch.cuda.set_device(self.args.gpu[0])
            #self.model = nets.nets_utils.MyDataParallel(self.model, device_ids=self.args.gpu)
            self.fp_model = nets.nets_utils.MyDataParallel(self.fp_model, device_ids=self.args.gpu)
        elif torch.cuda.device_count() > 1:
            #self.model = nets.nets_utils.MyDataParallel(self.model).cuda()
            self.fp_model = nets.nets_utils.MyDataParallel(self.fp_model).cuda()
        
        self.model.eval()
        self.fp_model.eval()

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train, batch_size=self.args.selection_batch, num_workers=self.args.workers,
            shuffle=False) # shuffle should be False to make sure index are the same!
        
        sample_num = self.n_train

        if self.args.adaptive == 'linear':
            alpha = self.current_epoch/self.args.epochs
            print("Current Coefficient Alpha with linear decay:{}".format(alpha))
        elif self.args.adaptive == 'evonly':
            alpha = 1
            print("Using Error Vector Norm Only")
        elif self.args.adaptive == 'dsonly':
            alpha = 0
            print("Using Disagreement Score Only")
        elif self.args.adaptive == 'quadratic':
            alpha = (self.current_epoch/self.args.epochs)**2
            print("Current Coefficient Alpha with quad decay:{}".format(alpha))
        elif self.args.adaptive == 'sqrt':
            alpha = math.sqrt(self.current_epoch/self.args.epochs)
            print("Current Coefficient Alpha with sqrt decay:{}".format(alpha))
        elif self.args.adaptive == 'cosine':
            alpha = 0.5*(1-np.cos((self.current_epoch/self.args.epochs)*np.pi))
            print("Current Coefficient Alpha with cosine decay:{}".format(alpha))
        else:
            alpha = 0.5
            print("Not implemented strategy, using 1-1 fixed setting")

        with torch.no_grad():
            for i, (input, targets) in enumerate(batch_loader):

                if i % self.args.print_freq == 0:
                    print('| Current Sample [%3d/%3d]' % (i * self.args.selection_batch, sample_num))
            
                outputs = F.softmax(self.model(input.to(self.args.device)), dim=1)

                targets_onehot = F.one_hot(targets.to(self.args.device), num_classes=self.num_classes)
                outputs_fp = F.softmax(self.fp_model(input.to(self.args.device)), dim=1)

                el2n_score_disagree = torch.linalg.vector_norm(x=(outputs - outputs_fp),ord=2,dim=1)
                el2n_score = torch.linalg.vector_norm(x=(outputs - targets_onehot),ord=2,dim=1)

                #for j in range(len(targets_onehot)):
                #    print("Targrts:{}, FP Prediction:{}, Quantized Prediction:{}".format(targets_onehot[j], outputs_fp[j], outputs[j],))
                #for j in range(len(el2n_score_disagree)):
                #    print("Disagreement Score:{}, EL2N Score:{}".format(el2n_score_disagree[j], el2n_score[j]))

                self.norm_matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = (1-alpha)*el2n_score + alpha*el2n_score_disagree

                #self.el2n[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = el2n_score 
                #self.disagreement[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = el2n_score_disagree
                
        self.norm_mean = self.norm_matrix.cpu().detach().numpy()
        # save the EL2N scores

        if self.save_path:
            pass
            #time_now = datetime.now()
            #filename = os.path.join(self.save_path, self.args.adaptive, f'el2n_scores_{time_now}.csv')
            #filename = os.path.join(self.save_path, f'el2n_scores_{self.current_epoch}.csv')
            #np.savetxt(filename, self.el2n.cpu().detach().numpy(), delimiter=',')
            #filename = os.path.join(self.save_path, f'disagreement_scores_{self.current_epoch}.csv')
            #np.savetxt(filename, self.disagreement.cpu().detach().numpy(), delimiter=',')

        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": self.norm_mean}

    