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


class Moderate(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=True, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.dst_train = deepcopy(dst_train)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.balance = balance
        self.torchvision_pretrain = True if self.args.dataset == "ImageNet" else False
        self.specific_model = None

        if self.args.model == 'QResNet18':
            self.feature_extractor = 'ResNet18'
        elif self.args.model == 'QMobilenetv2':
            self.feature_extractor = 'Mobilenetv2'
        else:
            self.feature_extractor = self.args.model

        if self.args.dataset == "ImageNet":
            self.num_classes = 1000 
        elif self.args.dataset == "CIFAR100":
            self.num_classes = 100
        else:
            self.num_classes = 10

    def select(self, **kwargs):
        self.train_indx = np.arange(self.n_train)
        if self.args.dataset == "ImageNet":
            model = nets.__dict__[self.feature_extractor](
                self.args.channel, self.num_classes, pretrained=self.torchvision_pretrain,
                im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)
        else:
            model = nets.__dict__[self.feature_extractor](
                self.args.channel, self.num_classes, pretrained=self.torchvision_pretrain,
                im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)
            checkpoint = torch.load(self.args.resume, map_location=self.args.device)
            # Loading model state_dict
            model.load_state_dict(checkpoint["state_dict"],strict=True)

        model = nets.nets_utils.MyDataParallel(model).cuda()
        model.eval()
        batch_loader = torch.utils.data.DataLoader(
            self.dst_train, batch_size=self.args.selection_batch, num_workers=self.args.workers,
            shuffle=False) # shuffle should be False to make sure index are the same!
        sample_num = self.n_train
        features = torch.zeros([self.n_train, self.num_classes])
        with torch.no_grad():
            for i, (input, target) in enumerate(batch_loader):
                if i % self.args.print_freq == 0:
                    print('| Current Sample [%3d/%3d]' % (i * self.args.selection_batch, sample_num))
                feature = model(input.to(self.args.device)).cpu()
                features[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = feature

        prots = np.zeros((self.num_classes, features.shape[-1]))
        print("Computing Median for each class")
        for c in range(self.num_classes):
            c_indx = self.train_indx[self.dst_train.targets == c]
            prots[c] = np.median(features[c_indx].squeeze(), axis=0, keepdims=False)

        prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))
        for c in range(self.num_classes):
            c_indx = self.train_indx[self.dst_train.targets == c]
            prots_for_each_example[c_indx, :] = prots[c]
        distance = np.linalg.norm(features - prots_for_each_example, axis=1)

        if not self.balance:
            raise NotImplementedError("Moderate Coreset only support class-balanced selection")
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(distance[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": distance}

    '''   
    def select(self, **kwargs):
        self.train_indx = np.arange(self.n_train)
        # Initialize a matrix to save norms of each sample 
        self.norm_matrix = torch.zeros([self.n_train], requires_grad=False).to(self.args.device)

        #self.el2n = torch.zeros([self.n_train], requires_grad=False).to(self.args.device)
        #self.disagreement = torch.zeros([self.n_train], requires_grad=False).to(self.args.device)

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
            new_state_dict = {}
            for k,v in checkpoint["state_dict"].items():
                if 'classifier.1' in k:
                    new_state_dict[k.replace('classifier.1', 'classifier.0')] = v
                else:
                    new_state_dict[k] = v
            self.fp_model.load_state_dict(new_state_dict,strict=True)
        
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

        if not self.balance:
            raise NotImplementedError("Moderate Coreset only support class-balanced selection")
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": self.norm_mean}
        ''' 

    