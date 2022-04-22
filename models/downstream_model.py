import torch.nn as nn


class IDCNN(nn.Module):
    def __init__(self, input_size, filters, kernel_size=32, num_block=2):
        super(IDCNN, self).__init__()
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}]
        net = nn.Sequential()
        # norms_1 = nn.ModuleList([LayerNorm(36) for _ in range(len(self.layers))])
        # norms_2 = nn.ModuleList([LayerNorm(9) for _ in range(num_block)])
        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size + dilation - 1)
            net.add_module("layer%d" % i, single_block)
            net.add_module("relu", nn.ReLU())
            # net.add_module("layernorm", norms_1[i])

        self.linear = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()

        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())
            # self.idcnn.add_module("layernorm", norms_2[i])

    def forward(self, embeddings):
        embeddings = self.linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        output = self.idcnn(embeddings).permute(0, 2, 1)
        return output
