
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


def train(args, train_df, milnet, criterion, optimizer):
    milnet.train()
    dirs = shuffle(train_df)
    # dirs = [os.path.join(args.input_feats_path, x) for x in dirs]

    data_info = load_pkl(args.input_label_info_dict)

    total_loss = 0


    Tensor = torch.cuda.FloatTensor
    for i, item in enumerate(dirs):
        optimizer.zero_grad()
        bag_label, bag_feats  = load_stacked_data(item, data_info[item], args.input_feats_path)
        bag_feats = Tensor(bag_feats)
        bag_label = Tensor([bag_label]).unsqueeze(0)
        # bag_label = Tensor([bag_label])
       

        bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch, args.random_dropout_patch)
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        # print('### debug')
        # print(ins_prediction.size())

        # max_prediction, _ = torch.max(ins_prediction, 0)
        avg_top_k_prediction = torch.mean(torch.topk(ins_prediction, k=min(50, ins_prediction.size(0)), dim=0)[0], dim=0)
        # print('')
        # print('bag_prediction:',bag_prediction.view(1, -1))
        # print('max_prediction:',max_prediction.view(1, -1)) 
        # print('bag_label:',bag_label)
        # print('new bag_label:',bag_label.long())
        # print('')
        # bag_pred_sigmoid = torch.sigmoid(bag_prediction)
        # max_pred_sigmoid = torch.sigmoid(max_prediction)
        # print('bag_pred_sigmoid:',bag_pred_sigmoid.view(1, -1))
        # print('max_pred_sigmoid:',max_pred_sigmoid.view(1, -1))


        bag_loss = sigmoid_focal_loss(bag_prediction.view(1, -1), bag_label.view(1, -1))
        # max_loss = sigmoid_focal_loss(max_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = sigmoid_focal_loss(avg_top_k_prediction.view(1, -1), bag_label.view(1, -1))
        # bag_loss = focal_loss(bag_pred_sigmoid.view(1, -1), bag_label.long().view(1, -1))
        # max_loss = focal_loss(max_pred_sigmoid.view(1, -1), bag_label.long().view(1, -1))
        # bag_loss = criterion(bag_prediction.view(1, -1), bag_label)
        # max_loss = criterion(max_prediction.view(1, -1), bag_label)
        loss = 0.9*bag_loss + 0.1*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    sys.stdout.write('\n')
    sys.stdout.flush()
    return total_loss / len(train_df)

def dropout_patches(feats, p, random_dropout_patch=False):
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

def test(args, test_df, milnet, criterion, thresholds=None, return_predictions=False):
    milnet.eval()
    total_loss = 0
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
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            if args.average:
                test_predictions.extend([(torch.sigmoid(max_prediction)+torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
        sys.stdout.write('\n')
        sys.stdout.flush()    
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)




    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print(classification_report(test_labels, test_predictions, digits=4))
        print(confusion_matrix(test_labels, test_predictions))

    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    if return_predictions:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

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

def save_model(args, fold, run, save_path, model, thresholds_optimal, is_best_statues):
    # Construct the filename including the fold number
    save_name = os.path.join(save_path, f'fold_{fold}_{run}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(args, save_name, thresholds_optimal, is_best_statues)
    file_name = os.path.join(save_path, f'fold_{fold}_{run}.json')
    with open(file_name, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)

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
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--min_lr', default=0.00001, type=float, help='Initial min_lr')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--stop_epochs', default=10, type=int, help='Skip remaining epochs if training has not improved after N epochs [10]')
    parser.add_argument('--num_try', default=1, type=int, help='num_try')
    # parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--input_feats_path', default=None, type=str, help='input_feats_path')
    parser.add_argument('--input_label_info_dict', default=None, type=str, help='input_label_info_dict')
    parser.add_argument('--input_train_file', default=None, type=str, help='input_train_file')
    parser.add_argument('--input_val_file', default=None, type=str, help='input_val_file')
    parser.add_argument('--output_dir', default=None, type=str, help='output_dir')

    # parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    # parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=str2bool, default="false", help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--random_dropout_patch', type=str2bool, default="false", help='random_dropout_patch')
    parser.add_argument('--cosine_cycle_steps', default=20, type=int)
    parser.add_argument('--cosine_cycle_warmup', default=2, type=int)
    parser.add_argument('--cosine_cycle_gamma', default=0.95, type=float)
    # parser.add_argument('--eval_scheme', default='5-fold-cv', type=str, help='Evaluation scheme [5-fold-cv | 5-fold-cv-standalone-test | 5-time-train+valid+test ]')

    
    args = parser.parse_args()
    # print(args.eval_scheme)

    # gpu_ids = tuple(args.gpu_index)
    # os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    


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
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCELoss()
        # criterion = nn.SoftMarginLoss()
        # criterion = FocalLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        # optimizer = torch.optim.RAdam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, args.min_lr)
        
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.cosine_cycle_steps, 
                                                  max_lr=args.lr, min_lr=args.min_lr,
                                                  warmup_steps=args.cosine_cycle_warmup, gamma=args.cosine_cycle_gamma)
        # scheduler = CosineAnnealingWarmupRestarts(optimizer)
        # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=args.num_epochs)
        return milnet, criterion, optimizer, scheduler
    
    # if args.dataset == 'TCGA-lung-default':
    #     bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    # else:
    #     bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')

    # generate_pt_files(args, pd.read_csv(bags_csv))
 
    # if args.eval_scheme == '5-fold-cv':
    # bags_path = glob.glob('temp_train/*.pt')

    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    save_path = os.path.join(args.output_dir, datetime.datetime.now().strftime("%Y%m%d%H%M"))
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))

    

    train_file_list = []
    val_file_list = []

    with open(args.input_train_file, 'r') as f:
        for line in f.readlines():
            train_file_list.append(line.rstrip())

    with open(args.input_val_file, 'r') as f:
        for line in f.readlines():
            val_file_list.append(line.rstrip())
    
    print(len(train_file_list))
    print(len(val_file_list))


    for try_num in range(args.num_try):
        fold_best_score = 0
        best_ac = 0
        best_auc = [0]
        best_comb_ac = 0
        best_comb_auc = 0
        is_best = [False, False, False]
        # counter = 0

        milnet, criterion, optimizer, scheduler = init_model(args)

        for epoch in range(1, args.num_epochs+1):
            # counter += 1
            print('***** current epoch {}/{}, current lr: {:f} *****'.format(str(epoch), str(args.num_epochs), optimizer.param_groups[0]['lr']))
            rtrain_loss_bag = train(args, train_file_list, milnet, criterion, optimizer) # iterate all bags
            
            train_loss_bag, train_avg_score, train_aucs, train_thresholds_optimal = test(args, train_file_list, milnet, criterion) # iterate all bags
            test_loss_bag, test_avg_score, test_aucs, test_thresholds_optimal = test(args, val_file_list, milnet, criterion)
            
            print('train info:')
            print_epoch_info(epoch, args, rtrain_loss_bag, train_loss_bag, train_avg_score, train_aucs)
            print('test info:')
            print_epoch_info(epoch, args, rtrain_loss_bag, test_loss_bag, test_avg_score, test_aucs)
            scheduler.step()

            # print('##### DEBUG 0 #####', aucs)

            test_current_score = get_current_score(test_avg_score, test_aucs)

            # print('##### DEBUG 1 #####', aucs)

            if test_current_score > fold_best_score:
                fold_best_score = test_current_score
                best_comb_ac = test_avg_score
                best_comb_auc = test_aucs
                is_best[0] = True
            
            if test_avg_score > best_ac:
                best_ac = test_avg_score
                is_best[1] = True

            if sum(test_aucs) > sum(best_auc):
                best_auc = test_aucs
                is_best[2] = True

            if sum(is_best) > 0:
                save_model(args, try_num, epoch, save_path, milnet, test_thresholds_optimal, is_best)
                is_best = [False, False, False]

        fold_results.append((best_comb_ac, best_comb_auc))

    mean_ac = np.mean(np.array([i[0] for i in fold_results]))
    mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
    # Print mean and std deviation for each class
    print(f"Final results: Mean Accuracy: {mean_ac}")
    for i, mean_score in enumerate(mean_auc):
        print(f"Class {i}: Mean AUC = {mean_score:.4f}")




    # for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
    #     print(f"Starting CV fold {fold}.")
        
    #     train_path = [bags_path[i] for i in train_index]
    #     test_path = [bags_path[i] for i in test_index]
    #     fold_best_score = 0
    #     best_ac = 0
    #     best_auc = 0
    #     counter = 0

    #     for epoch in range(1, args.num_epochs+1):
    #         counter += 1
    #         train_loss_bag = train(args, train_path, milnet, criterion, optimizer) # iterate all bags
    #         test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_path, milnet, criterion)
            
    #         print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
    #         scheduler.step()

    #         current_score = get_current_score(avg_score, aucs)
    #         if current_score > fold_best_score:
    #             counter = 0
    #             fold_best_score = current_score
    #             best_ac = avg_score
    #             best_auc = aucs
    #             save_model(args, fold, run, save_path, milnet, thresholds_optimal)
    #         if counter > args.stop_epochs: break
    #     fold_results.append((best_ac, best_auc))

    # mean_ac = np.mean(np.array([i[0] for i in fold_results]))
    # mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
    # # Print mean and std deviation for each class
    # print(f"Final results: Mean Accuracy: {mean_ac}")
    # for i, mean_score in enumerate(mean_auc):
    #     print(f"Class {i}: Mean AUC = {mean_score:.4f}")


    # elif args.eval_scheme == '5-time-train+valid+test':
    #     bags_path = glob.glob('temp_train/*.pt')
    #     # bags_path = bags_path.sample(n=50, random_state=42)
    #     fold_results = []

    #     save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
    #     os.makedirs(save_path, exist_ok=True)
    #     run = len(glob.glob(os.path.join(save_path, '*.pth')))

    #     for iteration in range(5):
    #         print(f"Starting iteration {iteration + 1}.")
    #         milnet, criterion, optimizer, scheduler = init_model(args)

    #         bags_path = shuffle(bags_path)
    #         total_samples = len(bags_path)
    #         train_end = int(total_samples * (1-args.split-0.1))
    #         val_end = train_end + int(total_samples * 0.1)

    #         train_path = bags_path[:train_end]
    #         val_path = bags_path[train_end:val_end]
    #         test_path = bags_path[val_end:]

    #         fold_best_score = 0
    #         best_ac = 0
    #         best_auc = 0
    #         counter = 0

    #         for epoch in range(1, args.num_epochs + 1):
    #             counter += 1
    #             train_loss_bag = train(args, train_path, milnet, criterion, optimizer) # iterate all bags
    #             test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, val_path, milnet, criterion)
                
    #             print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
    #             scheduler.step()

    #             current_score = get_current_score(avg_score, aucs)
    #             if current_score > fold_best_score:
    #                 counter = 0
    #                 fold_best_score = current_score
    #                 best_ac = avg_score
    #                 best_auc = aucs
    #                 save_model(args, iteration, run, save_path, milnet, thresholds_optimal)
    #                 best_model = copy.deepcopy(milnet)
    #             if counter > args.stop_epochs: break
    #         test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_path, best_model, criterion, args)
    #         fold_results.append((best_ac, best_auc))
    #     mean_ac = np.mean(np.array([i[0] for i in fold_results]))
    #     mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
    #     # Print mean and std deviation for each class
    #     print(f"Final results: Mean Accuracy: {mean_ac}")
    #     for i, mean_score in enumerate(mean_auc):
    #         print(f"Class {i}: Mean AUC = {mean_score:.4f}")

    # if args.eval_scheme == '5-fold-cv-standalone-test':
    #     bags_path = glob.glob('temp_train/*.pt')
    #     bags_path = shuffle(bags_path)
    #     reserved_testing_bags = bags_path[:int(args.split*len(bags_path))]
    #     bags_path = bags_path[int(args.split*len(bags_path)):]
    #     kf = KFold(n_splits=5, shuffle=True, random_state=42)
    #     fold_results = []
    #     fold_models = []

    #     save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
    #     os.makedirs(save_path, exist_ok=True)
    #     run = len(glob.glob(os.path.join(save_path, '*.pth')))

    #     for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
    #         print(f"Starting CV fold {fold}.")
    #         milnet, criterion, optimizer, scheduler = init_model(args)
    #         train_path = [bags_path[i] for i in train_index]
    #         test_path = [bags_path[i] for i in test_index]
    #         fold_best_score = 0
    #         best_ac = 0
    #         best_auc = 0
    #         counter = 0
    #         best_model = []

    #         for epoch in range(1, args.num_epochs+1):
    #             counter += 1
    #             train_loss_bag = train(args, train_path, milnet, criterion, optimizer) # iterate all bags
    #             test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_path, milnet, criterion)
                
    #             print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
    #             scheduler.step()

    #             current_score = get_current_score(avg_score, aucs)
    #             if current_score > fold_best_score:
    #                 counter = 0
    #                 fold_best_score = current_score
    #                 best_ac = avg_score
    #                 best_auc = aucs
    #                 save_model(args, fold, run, save_path, milnet, thresholds_optimal)
    #                 best_model = [copy.deepcopy(milnet.cpu()), thresholds_optimal]
    #                 milnet.cuda()
    #             if counter > args.stop_epochs: break
    #         fold_results.append((best_ac, best_auc))
    #         fold_models.append(best_model)

    #     fold_predictions = []
    #     for item in fold_models:
    #         best_model = item[0]
    #         optimal_thresh = item[1]
    #         test_loss_bag, avg_score, aucs, thresholds_optimal, test_predictions, test_labels = test(args, reserved_testing_bags, best_model.cuda(), criterion, thresholds=optimal_thresh, return_predictions=True)
    #         fold_predictions.append(test_predictions)
    #     predictions_stack = np.stack(fold_predictions, axis=0)
    #     mode_result = mode(predictions_stack, axis=0)
    #     combined_predictions = mode_result.mode[0]
    #     combined_predictions = combined_predictions.squeeze()

    #     if args.num_classes > 1:
    #         # Compute Hamming Loss
    #         hammingloss = hamming_loss(test_labels, combined_predictions)
    #         print("Hamming Loss:", hammingloss)
    #         # Compute Subset Accuracy
    #         subset_accuracy = accuracy_score(test_labels, combined_predictions)
    #         print("Subset Accuracy (Exact Match Ratio):", subset_accuracy)
    #     else:
    #         accuracy = accuracy_score(test_labels, combined_predictions)
    #         print("Accuracy:", accuracy)
    #         balanced_accuracy = balanced_accuracy_score(test_labels, combined_predictions)
    #         print("Balanced Accuracy:", balanced_accuracy)

    #     os.makedirs('test', exist_ok=True)
    #     with open("test/test_list.json", "w") as file:
    #         json.dump(reserved_testing_bags, file)

    #     for i, item in enumerate(fold_models):
    #         best_model = item[0]
    #         optimal_thresh = item[1]
    #         torch.save(best_model.state_dict(), f"test/mil_weights_fold_{i}.pth")
    #         with open(f"test/mil_threshold_fold_{i}.json", "w") as file:
    #             optimal_thresh = [float(i) for i in optimal_thresh]
    #             json.dump(optimal_thresh, file)
                

if __name__ == '__main__':
    main()