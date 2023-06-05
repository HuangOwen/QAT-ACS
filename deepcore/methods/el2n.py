from .. import nets
from .earlytrain import EarlyTrain
import torch, time
import numpy as np
from ..nets.nets_utils import MyDataParallel
import torch.nn.functional as F
from datetime import datetime
import os

class EL2N(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, el2n_repeat=1,
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model
        self.repeat = el2n_repeat

        self.balance = balance
        
        if 'save_path' in kwargs:
            self.save_path = kwargs['save_path']
        else:
            self.save_path = None


    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

    def finish_run(self):

        self.model.eval()

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train, batch_size=self.args.selection_batch, num_workers=self.args.workers,
            shuffle=False)

        sample_num = self.n_train
        num_classes = 1000 # this is for ImageNet, need to change later

        with torch.no_grad():
            for i, (input, targets) in enumerate(batch_loader):
            
                outputs = self.model(input.to(self.args.device))
                batch_num = targets.shape[0]
                targets_onehot = F.one_hot(targets.to(self.args.device), num_classes=num_classes)
                el2n_score = torch.linalg.vector_norm(
                    x=(outputs - targets_onehot),
                    ord=2,
                    dim=1
                )

                self.norm_matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num),
                self.cur_repeat] = el2n_score

        self.model.train()


    def select(self, **kwargs):
        # Initialize a matrix to save norms of each sample on idependent runs
        self.norm_matrix = torch.zeros([self.n_train, self.repeat], requires_grad=False).to(self.args.device)

        for self.cur_repeat in range(self.repeat):
            self.run()
            self.random_seed = int(time.time() * 1000) % 100000

        self.norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
        
        # save the EL2N scores
        if self.save_path:
            time_now = datetime.now()
            filename = os.path.join(self.save_path, f'el2n_scores_{time_now}.csv')
        
            np.savetxt(filename, self.norm_mean, delimiter=',')

        
        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])

        return {"indices": top_examples, "scores": self.norm_mean}