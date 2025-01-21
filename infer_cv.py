
import sys, argparse, os, copy, itertools, glob, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold
from collections import OrderedDict
import json
from tqdm import tqdm
import pickle

import dsmil as mil
# from loss import FocalLoss, sigmoid_focal_loss
from loss import FocalLoss
from warmup_cosine_annealing import CosineAnnealingWarmupRestarts
from torchvision.ops.focal_loss import sigmoid_focal_loss

# read all patch feats for one target
def get_bag_feats(csv_file_df, args):
    # if args.dataset == 'TCGA-lung-default':
    #     feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    # else:
    #     feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats, feats_csv_path



def generate_pt_files(args, df):
    temp_train_dir = "temp_train"
    if os.path.exists(temp_train_dir):
        import shutil
        shutil.rmtree(temp_train_dir, ignore_errors=True)
    os.makedirs(temp_train_dir, exist_ok=True)
    print('Creating intermediate training files.')

    for i in tqdm(range(len(df))):
        label, feats, feats_csv_path = get_bag_feats(df.iloc[i], args)
        bag_label = torch.tensor(np.array([label]), dtype=torch.float32)
        bag_feats = torch.tensor(np.array(feats), dtype=torch.float32)
        repeated_label = bag_label.repeat(bag_feats.size(0), 1)
        stacked_data = torch.cat((bag_feats, repeated_label), dim=1)
        # Save the stacked data into a .pt file
        pt_file_path = os.path.join(temp_train_dir, os.path.splitext(feats_csv_path)[0].split(os.sep)[-1] + ".pt")
        torch.save(stacked_data, pt_file_path)

    # stacked_data strucutre (num_patches, feats_size+1) where 1 is the bag label

def load_pkl(path):
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result

def save_pkl(pkl_save_path, result_dict):
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(result_dict, f)

def load_stacked_data(filename, cur_data_info, input_feats_path):
    # print(cur_data_info)
    cur_feat_path_list = [os.path.join(input_feats_path, x) for x in cur_data_info[1]]
    # print(cur_feat_path_list)
    cur_label = cur_data_info[2]

    cur_feat = []

    for cur_feat_path in cur_feat_path_list:
        tmp_feat = load_pkl(cur_feat_path)
        # print(tmp_feat.shape)
        cur_feat.append(tmp_feat)

    cur_feats = np.concatenate(cur_feat, axis=0)
    # print(cur_feats.shape)



    return cur_label, cur_feats



def dropout_patches(feats, p, random_dropout_patch=False, thresholds=None):
    num_rows = feats.size(0)
    if random_dropout_patch:
        cp = np.random.uniform(0, 1-p)
        cp = 1 - cp
    else:
        cp = p 
    num_rows_to_select = int(num_rows * cp)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows

def infer(args, test_df, milnet, thresholds=None):
    milnet.eval()

    test_labels = []
    test_predictions = []
    data_info = load_pkl(args.input_label_info_dict)
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, item in enumerate(test_df):
            bag_label, bag_feats  = load_stacked_data(item, data_info[item], args.input_feats_path)
            bag_feats = Tensor(bag_feats)
            bag_label = Tensor([bag_label]).unsqueeze(0)
            # bag_label = Tensor([bag_label])
            # bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction = torch.mean(torch.topk(ins_prediction, k=min(args.patch_max_loss_top_k, ins_prediction.size(0)), dim=0)[0], dim=0)
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            if args.average:
                test_predictions.extend([((torch.sigmoid(max_prediction)+torch.sigmoid(bag_prediction)).squeeze().cpu().numpy())/2])
            else: test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_labels = np.squeeze(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, best_thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds:
        print('#### using settintg threshold {} ####'.format(thresholds))
        thresholds_optimal = thresholds
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal] = 1
        class_prediction_bag[test_predictions<thresholds_optimal] = 0
        test_cls_predictions = class_prediction_bag
        print(classification_report(test_labels, test_cls_predictions, digits=4))
        print(confusion_matrix(test_labels, test_cls_predictions))

    print('#### using optimal threshold {} ####'.format(best_thresholds_optimal))
    thresholds_optimal = best_thresholds_optimal
    class_prediction_bag = copy.deepcopy(test_predictions)
    class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
    class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
    test_cls_predictions = class_prediction_bag
    print(classification_report(test_labels, test_cls_predictions, digits=4))
    print(confusion_matrix(test_labels, test_cls_predictions))

    return  test_predictions, test_labels

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        # c_auc = roc_auc_score(label, prediction)
        try:
            c_auc = roc_auc_score(label, prediction)
            print("ROC AUC score:", c_auc)
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                print("ROC AUC score is not defined when only one class is present in y_true. c_auc is set to 1.")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs):

        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 

def get_current_score(avg_score, aucs):
    current_score = (sum(aucs) + avg_score)/2
    return current_score


def print_save_message(args, save_name, thresholds_optimal, is_best_statues):

        content_1 = 'model saved at: '

        if is_best_statues[0]:
            content_1 = 'best score model saved at: ' + save_name
        if is_best_statues[1]:
            content_1 = 'best avg score model saved at: ' + save_name
        if is_best_statues[2]:
            content_1 = 'best auc score model saved at: ' + save_name


        print(content_1)
        content_2 = 'Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal))
        print(content_2)

        rec_file = os.path.splitext(save_name)[0] + '.txt'

        with open(rec_file, 'w') as f:
            f.write(content_1 + '\n')
            f.write(content_2 + '\n')


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=2048, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--patch_max_loss_top_k', default=20, type=int, help='Dimension of the feature size [512]')

    parser.add_argument('--input_feats_path', default=None, type=str, help='input_feats_path')
    parser.add_argument('--input_label_info_dict', default=None, type=str, help='input_label_info_dict')
    parser.add_argument('--input_model_path', default=None, type=str, help='input_label_info_dict')
    parser.add_argument('--output_path', default=None, type=str, help='output_path')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--thresholds', default=None, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=str2bool, default="false", help='Average the score of max-pooling and bag aggregating')

    args = parser.parse_args()

    print("#### infering info ####")
    for k, v in vars(args).items():
        print(k, ':', v)
    print("#### infering info end ####")


    def apply_sparse_init(m):
        # if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        #     nn.init.orthogonal_(m.weight)
        #     if m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def init_model(args):
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        milnet.apply(lambda m: apply_sparse_init(m))

        return milnet
    
    save_path = args.output_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state_dict_weights = torch.load(args.input_model_path)

    all_data_dict =load_pkl(args.input_label_info_dict)

    print(len(all_data_dict.keys()))


    all_path_no_list = list(all_data_dict.keys())
    all_path_no_label_list = [all_data_dict[k][2] for k in all_data_dict]

    all_path_no_list = np.array(all_path_no_list)
    all_path_no_label_list = np.array(all_path_no_label_list)

    milnet = init_model(args)

    milnet.load_state_dict(state_dict_weights)

    milnet.eval()
        
    test_predictions, test_labels = infer(args, all_path_no_list, milnet, thresholds=args.thresholds)

    print(test_predictions.shape, test_labels.shape)


    pd.DataFrame({'path_no': all_path_no_list, 'label': test_labels, 'pred_conf': test_predictions}).to_csv(save_path, index=False)


if __name__ == '__main__':
    main()