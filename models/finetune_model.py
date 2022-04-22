import torch.nn.functional as F
from models.downstream_model import IDCNN
import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertModel
from pretrain_model_utils.nezha.modeling_nezha import NeZhaModel

MODEL_NAME = {
    'nezha_wwm': 'NeZhaModel',
    'nezha_base': 'NeZhaModel',
    'roberta': 'BertModel'
}


class Model(nn.Module):
    def __init__(self, bert_config, args):
        super(Model, self).__init__()
        self.bert = globals()[MODEL_NAME[args.model_type]](config=bert_config)

        if not args.use_avg:
            args.avg_size = 1
        self.args = args

        if args.struc == 'cls':
            self.fc = nn.Linear(768 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bilstm':
            self.bilstm = nn.LSTM(768, args.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.lstm_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bigru':
            self.bigru = nn.GRU(768, args.gru_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.gru_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'idcnn':
            self.idcnn = IDCNN(input_size=768, filters=64)
            self.fc = nn.Linear(32 + 1 - args.avg_size, args.num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(args.dropout_num)])

    def forward(self, x):
        output = self.bert(**x)[0]  # 0:sequence_output  1:pooler_output
        if self.args.struc == 'cls':
            output = output[:, 0, :]  # cls

        else:
            if self.args.struc == 'bilstm':
                _, hidden = self.bilstm(output)
                last_hidden = hidden[0].permute(1, 0, 2)
                output = last_hidden.contiguous().view(-1, self.args.lstm_dim * 2)
            elif self.args.struc == 'bigru':
                _, hidden = self.bigru(output)
                last_hidden = hidden.permute(1, 0, 2)
                output = last_hidden.contiguous().view(-1, self.args.gru_dim * 2)
            elif self.args.struc == 'idcnn':
                output = self.idcnn(output)
                output = torch.mean(output, dim=1)
        if self.args.use_avg:
            if self.args.struc == 'idcnn':
                output = F.avg_pool1d(output.unsqueeze(1), kernel_size=32, stride=1).squeeze(1)
            else:
                output = F.avg_pool1d(output.unsqueeze(1), kernel_size=self.args.avg_size, stride=1).squeeze(1)
        # output = self.dropout(output)
        if self.args.dropout_num == 1:
            output = self.dropouts[0](output)
            output = self.fc(output)
        else:
            out = None
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(output)
                    out = self.fc(out)
                else:
                    temp_out = dropout(output)
                    out = out + self.fc(temp_out)
            output = out / len(self.dropouts)

        output = torch.sigmoid(output)

        return output

class NeuralNet(nn.Module):
    def __init__(self, hidden_size=768, num_class=2):
        super(NeuralNet, self).__init__()

        self.config = BertConfig.from_pretrained('/home/wangzhili/YangYang/pretrainModel/ernie-gram/', num_labels=args.num_classes)
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained('/home/wangzhili/YangYang/pretrainModel/ernie-gram/', config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size*2, num_class)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2) for _ in range(5)
        ])

    def forward(self, input_ids, input_mask, segment_ids, y=None, loss_fn=None):
        output = self.bert(input_ids, token_type_ids=segment_ids,attention_mask=input_mask)

        last_hidden = output.last_hidden_state
        all_hidden_states = output.hidden_states
        batch_size = input_ids.shape[0]
        # concat每个编码层输出的cls状态向量
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])

        f = torch.mean(last_hidden, 1)
        feature = torch.cat((feature, f), 1)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
                if loss_fn is not None:
                    loss = loss_fn(h, y)
            else:
                hi = self.fc(dropout(feature))
                h = h + hi
                if loss_fn is not None:
                    loss = loss + loss_fn(hi, y)
        if loss_fn is not None:
            return h / len(self.dropouts), loss / len(self.dropouts)
        return h / len(self.dropouts)

class ModelForDynamicLen(nn.Module):
    def __init__(self, bert_config, args):
        super(ModelForDynamicLen, self).__init__()
        MODEL_NAME = {'nezha_wwm': 'NeZhaModel', 'nezha_base': 'NeZhaModel', 'roberta': 'BertModel'}
        self.bert = globals()[MODEL_NAME[args.model_type]](config=bert_config)
        self.args = args

        if args.struc == 'cls':
            self.fc = nn.Linear(768 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bilstm':
            self.bilstm = nn.LSTM(768, args.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.lstm_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bigru':
            self.bigru = nn.GRU(768, args.gru_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.gru_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'idcnn':
            self.idcnn = IDCNN(input_size=768, filters=64)
            self.fc = nn.Linear(32 + 1 - args.avg_size, args.num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(args.dropout_num)])

    def forward(self, input_ids):
        output = None
        if self.args.struc == 'cls':
            output = torch.stack(
                [self.bert(input_id.to(self.args.device))[0][0][0]
                 for input_id in input_ids])

        if self.args.AveragePooling:
            output = F.avg_pool1d(output.unsqueeze(1), kernel_size=self.args.avg_size, stride=1).squeeze(1)

        # output = self.dropout(output)
        if self.args.dropout_num == 1:
            output = self.dropouts[0](output)
            output = self.fc(output)
        else:
            out = None
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(output)
                    out = self.fc(out)
                else:
                    temp_out = dropout(output)
                    out = out + self.fc(temp_out)
            output = out / len(self.dropouts)

        return output
