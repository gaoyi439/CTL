import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        # self.relu1 = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class co_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(co_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # 自动学习标签关联性
        self.co_block = nn.Sequential(
             nn.Sigmoid(),
             nn.Linear(output_dim, 64),
             nn.ELU(),
        )
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        out_1 = self.co_block(out)
        out_1 = self.output(out_1)+out
        return out_1

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class co_MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(co_MLP2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # 自动学习标签关联性
        self.co_block = nn.Sequential(
             nn.Sigmoid(),
             nn.Linear(output_dim, 64),
             nn.ELU(),
             nn.Linear(64, output_dim)
        )
        self.co_block_1 = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, 64),
            nn.ELU(),
        )
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        out_1 = self.co_block(out)+out
        out_1 = self.co_block_1(out_1)
        out_1 = self.output(out_1)+out
        return out_1

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MLPC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPC, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.out = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, output_dim)  # correlation matrix
        )

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out_Y = self.out(out)
        return out, out_Y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))  # 把图片展成一维的
        out = self.linear(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class co_linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(co_linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 模型原始输出

        # 自动学习标签关联性
        self.co_block = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, 64),
            nn.ELU(),
        )
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))  # 把图片展成一维的
        out = self.linear(out)

        out_1 = self.co_block(out)
        out_1 = self.output(out_1) + out
        return out_1

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class co_linear2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(co_linear2, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 模型原始输出

        # 自动学习标签关联性
        self.co_block = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, 64),
            nn.ELU(),
            nn.Linear(64, output_dim)
        )
        self.co_block_1 = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, 64),
            nn.ELU(),
        )
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))  # 把图片展成一维的
        out = self.linear(out)

        out_1 = self.co_block(out) + out
        out_1 = self.co_block_1(out_1)
        out_1 = self.output(out_1) + out
        return out_1

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class linearC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linearC, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.out = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, output_dim)  # correlation matrix
        )

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))  # 把图片展成一维的
        out = self.linear(out)
        out_Y = self.out(out)
        return out, out_Y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features