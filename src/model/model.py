import torch.nn as nn


class GroundNet(nn.Module):
    # https://github.com/martin-velas/but_velodyne_cnn/blob/master/networks/L05-deconv/net_train_val.prototxt
    def __init__(self):
        super(GroundNet, self).__init__()
        self.net = self._make_network()

    def _make_network(self):
        net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 24, 5, 2, 2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=4, stride=1, padding=2),
        )
        return net

    def forward(self, input_matrix):
        output = self.net(input_matrix)
        return output
