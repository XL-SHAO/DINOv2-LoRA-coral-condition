import sys
sys.path.append('/home/songjian/project/CRC')
import argparse

import torch.nn as nn
from torch.optim import AdamW
from datasets.make_dataloader import make_data_loader, CoralReefDataSet, CoralReefDomainAdaptationDataSet
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from tqdm import tqdm
import os
import time
from deep_model.DINOV2_LoRA import DINOV2LoRA
from torch.optim.lr_scheduler import StepLR


def adjust_learning_rate(optimizer, base_lr, i_iter, num_steps, power, warmup_steps, warmup_mode='linear', decay_mode='poly'):
    """Adjusts the learning rate with configurable warm-up and decay phases."""
    if i_iter < warmup_steps:
        # Warm-up phase
        if warmup_mode == 'linear':
            lr = base_lr * (i_iter / warmup_steps)
        elif warmup_mode == 'poly':
            lr = base_lr * ((i_iter / warmup_steps) ** power)
    else:
        # Decay phase
        if decay_mode == 'linear':
            lr = base_lr * (1 - (i_iter - warmup_steps) / (num_steps - warmup_steps))
        elif decay_mode == 'poly':
            lr = base_lr * ((1 - (i_iter - warmup_steps) / (num_steps - warmup_steps)) ** power)

    optimizer.param_groups[0]['lr'] = lr
    return lr


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Compute the probability with sigmoid
        probs = torch.sigmoid(logits)
        
        # Compute binary cross-entropy loss without reduction
        bce_loss = binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute the modulating factor (1 - p_t)^gamma
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply the focal weight to the BCE loss
        loss = self.alpha * focal_weight * bce_loss
        
        # Apply reduction (mean or sum)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.model = DINOV2LoRA(output_dim = 8, vit_type = args.vit_type, freeze_vit=args.freeze_vit, low_rank=args.low_rank)
        self.model = torch.nn.DataParallel(self.model)
        print(self.model)

        # Replace the final fully connected layer
        # Here, 8 is the number of output classes
        self.model.cuda()


        self.optimizer_G = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.loss_fn = binary_cross_entropy_with_logits

        self.train_data_loader = make_data_loader(args)
        self.model_save_path = os.path.join(args.model_save_path, str(time.time()))
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)


    def train(self):
        best_info = []
        best_match_ratio = 0
        # Number of input features to the final fully connected layer
        source_dataset_path = '/data/ggeoinfo/datasets/coral_reef_new/coral_reef_dataset'
        source_data_info_path = '/home/songjian/project/CRC/datasets/train_data_all_season.json'
        # source_dataset = CoralReefDataSet(source_dataset_path, source_data_info_path, max_iters=self.args.max_iters, type='train')
        # source_data_loader = DataLoader(source_dataset, batch_size=self.args.batch_size, num_workers=16, drop_last=False)

        
        # target_dataset = CoralReefDataSet(target_dataset_path, target_data_info_path, max_iters=self.args.max_iters, type='train')
        # target_data_loader = DataLoader(target_dataset, batch_size=self.args.batch_size, num_workers=16, drop_last=False)
        domain_adaptation_dataset = CoralReefDataSet(source_dataset_path, source_data_info_path, max_iters=self.args.max_iters, type='train') # CoralReefDomainAdaptationDataSet(source_dataset_path, source_data_info_path, target_data_info_path, max_iters=self.args.max_iters, type='train')
        domain_adaptation_data_loader = DataLoader(domain_adaptation_dataset, batch_size=self.args.batch_size, num_workers=16, drop_last=False)
        # FDA_prob = 0.5
        # mean_img = torch.zeros(1, 1)
        # std_img = torch.zeros(1, 1)

        self.model.train()
        for _iter, data in enumerate(tqdm(domain_adaptation_data_loader)):
            src_img, src_labels, _ = data
            
            src_img = src_img.cuda()
            src_labels = src_labels[:, 1:]
            src_labels = src_labels.cuda()
         
            outputs = self.model(src_img)

            # Classifier and adversarial training to fool discriminator
            self.optimizer_G.zero_grad()
            
            adjust_learning_rate(self.optimizer_G, self.args.learning_rate, _iter, 30000, 0.9, 1000, 'linear', decay_mode='poly')

            G_total_loss = self.loss_fn(outputs, src_labels)

            G_total_loss.backward()
            self.optimizer_G.step()

            if (_iter + 1) % 500 == 0:
                self.model.eval()

                S_each_class_f1, S_micro_f1, S_macro_f1, S_exact_match_ratio = self.source_validation()
                T_each_class_f1, T_micro_f1, T_macro_f1, T_exact_match_ratio = self.target_validation()

                if S_exact_match_ratio > best_match_ratio:
                    best_match_ratio = S_exact_match_ratio
                    best_info = [_iter + 1, G_total_loss, S_exact_match_ratio, S_micro_f1, S_macro_f1, T_exact_match_ratio, T_micro_f1, T_macro_f1, S_each_class_f1, T_each_class_f1]
                
                    self.model.module.save_parameters( os.path.join(self.model_save_path, f'best_model.pth'))
                    
                print(f'iteration is {_iter + 1}, '
                      f'G loss is {G_total_loss}, '
                      f'match ratio S/T is {S_exact_match_ratio} / {T_exact_match_ratio}, ' 
                      f'micro f1 score S/T is {S_micro_f1} / {T_micro_f1}, '
                      f'macro f1 score S/T is {S_macro_f1} / {T_macro_f1}, '
                      f'each class f1 score S/T is {S_each_class_f1} / {T_each_class_f1}')
                self.model.train()


        print(f'best iteration is {best_info[0]}, '
              f'loss is {best_info[1]}, '
              f'match ratio is {best_info[2]} / {best_info[5]}, '
              f'micro f1 score is {best_info[3]} / {best_info[6]}, '
              f'macro f1 score is {best_info[4]} / {best_info[7]}, '
              f'each class f1 score is {best_info[8]} / {best_info[9]}')

    def batch_entropy_loss(self, probabilities):
        # Calculate entropy for probabilities
        epsilon = 1e-10
        entropy = -(probabilities * torch.log(probabilities + epsilon) + (1 - probabilities) * torch.log(1 - probabilities + epsilon))
        # Average entropy across the batch and sum over classes
        batch_entropy = entropy.mean(dim=0).mean()
        return batch_entropy
    
    def source_validation(self):
        dataset_path = '/data/ggeoinfo/datasets/coral_reef_new/coral_reef_dataset'
        data_info_path = '/home/songjian/project/CRC/datasets/val_data_all_season.json'
        dataset = CoralReefDataSet(dataset_path, data_info_path, max_iters=None, type='test')
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        val_data_loader = DataLoader(dataset, batch_size=8, num_workers=8, drop_last=False)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for _itera, data in enumerate(val_data_loader):
                img, labels, _ = data
                labels = labels[:, 1:]
                
                B, C, H, W = img.shape
                
                img = img.cuda()
                outputs = self.model(img)
                prob = torch.sigmoid(outputs)

                # Apply threshold to get binary predictions
                predictions = (prob > 0.5).int().cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels.int().cpu().numpy())

        y_pred = np.array(all_predictions)
        y_true = np.array(all_labels)
        # F1 Score (average can be 'micro', 'macro', 'weighted', or 'samples')
        each_class_f1 = f1_score(y_true, y_pred, average=None)
        microf1 = f1_score(y_true, y_pred, average='micro')
        macrof1 = f1_score(y_true, y_pred, average='macro')
        # Exact Match Ratio (Subset Accuracy)
        exact_match_ratio = accuracy_score(y_true, y_pred)
        return each_class_f1, microf1, macrof1, exact_match_ratio


    def target_validation(self):
        dataset_path = '/data/ggeoinfo/datasets/coral_reef_new/coral_reef_dataset'
        data_info_path = '/home/songjian/project/CRC/datasets/test_data_four_events.json'
        dataset = CoralReefDataSet(dataset_path, data_info_path, max_iters=None, type='test')
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        val_data_loader = DataLoader(dataset, batch_size=8, num_workers=8, drop_last=False)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for _itera, data in enumerate(val_data_loader):
                img, labels, _ = data
                labels = labels[:, 1:]

                img = img.cuda()
                outputs = self.model(img)
                prob = torch.sigmoid(outputs)

                # Apply threshold to get binary predictions
                predictions = (prob > 0.5).int().cpu().numpy()

                all_predictions.extend(predictions)
                all_labels.extend(labels.int().cpu().numpy())

        y_pred = np.array(all_predictions)
        y_true = np.array(all_labels)
        # F1 Score (average can be 'micro', 'macro', 'weighted', or 'samples')
        each_class_f1 = f1_score(y_true, y_pred, average=None)
        microf1 = f1_score(y_true, y_pred, average='micro')
        macrof1 = f1_score(y_true, y_pred, average='macro')
        # Exact Match Ratio (Subset Accuracy)
        exact_match_ratio = accuracy_score(y_true, y_pred)
        return each_class_f1, microf1, macrof1, exact_match_ratio



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Coral Reef DataLoader Test")
    parser.add_argument('--dataset', type=str, default='coral_reef')
    parser.add_argument('--max_iters', type=int, default=480000)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='/data/ggeoinfo/datasets/coral_reef_new/coral_reef_dataset')
    parser.add_argument('--data_info_path', type=str,
                        default='/home/songjian/project/CRC/datasets/train_data_mix_all_season.json')
    parser.add_argument('--model_save_path', type=str,
                        default='/home/songjian/project/CRC/saved_model/DINOV2')
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--low_rank', type=int, default=6)
    parser.add_argument('--vit_type', type=str)
    parser.add_argument('--freeze_vit', type=bool, default=True)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
