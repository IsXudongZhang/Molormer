import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
torch.manual_seed(2)
np.random.seed(3)
from argparse import ArgumentParser
from dataset import Dataset
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collator import collator

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='Molormer Prediction.')
parser.add_argument('-b', '--batch-size', default=16, type=int,metavar='N')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
            p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
            label) in enumerate(tqdm(data_generator)):

        score = model(d_node.cuda(), d_attn_bias.cuda(), d_spatial_pos.cuda(),
                      d_in_degree.cuda(), d_out_degree.cuda(), d_edge_input.cuda(),p_node.cuda(),
                      p_attn_bias.cuda(), p_spatial_pos.cuda(), p_in_degree.cuda(),
                      p_out_degree.cuda(), p_edge_input.cuda())
       
        label = Variable(torch.from_numpy(np.array(label-1)).long()).cuda()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(score, label)
        loss_accumulate += loss
        count += 1

        outputs = score.argmax(dim=1).detach().cpu().numpy() + 1
        label_ids = label.to('cpu').numpy() + 1

        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + outputs.flatten().tolist()
     
    loss = loss_accumulate / count

    accuracy = accuracy_score(y_label, y_pred)
    micro_precision = precision_score(y_label, y_pred, average='micro')
    micro_recall = recall_score(y_label, y_pred, average='micro')
    micro_f1 = f1_score(y_label, y_pred, average='micro')


    macro_precision = precision_score(y_label, y_pred, average='macro')
    macro_recall = recall_score(y_label, y_pred, average='macro')
    macro_f1 = f1_score(y_label, y_pred, average='macro')

    print("[Validation metrics]: loss:{:.4f} mean_accuracy:{:.4f} micro_precision:{:.4f} micro_recall:{:.4f} micro_f1:{:.4f} macro_precision:{:.4f} macro_recall:{:.4f} macro_f1:{:.4f}".format(
        loss, accuracy, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1))


def main():
    args = parser.parse_args()

    model = torch.load('save_model/0.9693686184621796_model.pth')

    model = model.to(device)


    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': args.workers,
              'drop_last': False,
              'collate_fn': collator}

    df_test = pd.read_csv('dataset/test.csv')


    testing_set = Dataset(df_test.index.values, df_test.Label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Predicting ---')
    with torch.set_grad_enabled(False):
        test(testing_generator, model)

    torch.cuda.empty_cache()

main()
print("Done!")
