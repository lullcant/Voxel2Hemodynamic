import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        return self.conv(x)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernel=(2, 2), pool_stride=(2, 2)):
        super(ConvEncoder, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels),
            nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(2, 2), stride=(2, 2), bias=True)
        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        up_height, up_width = x.size()[2:]
        sk_height, sk_width = skip.size()[2:]

        diff_y = sk_height - up_height
        diff_x = sk_width - up_width

        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channels=[32, 64, 128, 256]):
        super(Unet, self).__init__()

        self.channels = channels
        self.num_layers = len(self.channels) - 1

        self.incoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels[0] // 2, kernel_size=(3, 3), padding=1, bias=True),
            nn.LeakyReLU(inplace=False)
        )

        self.encoders = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.encoders.append(
                ConvEncoder(in_channels=channels[index - 1] if index > 0 else channels[0] // 2,
                            out_channels=channel)
            )

        self.decoders = nn.ModuleList()
        for index, channel in enumerate(channels):
            index = self.num_layers - index
            self.decoders.append(
                ConvDecoder(in_channels=channels[index],
                            out_channels=channels[index - 1] if index > 0 else channels[0] // 2)
            )

        self.outcoder = nn.Conv2d(in_channels=channels[0] // 2,
                                  out_channels=out_channels, kernel_size=(1, 1), stride=1)

    def forward(self, image):
        x = self.incoder(image)
        skipes = [x]
        for index, encoder in enumerate(self.encoders):
            skipes.append(encoder(skipes[index]))

        for index, decoder in enumerate(self.decoders):
            x = decoder(x=x if index > 0 else skipes[-1], skip=skipes[-index - 2])

        x = torch.sigmoid(self.outcoder(x))
        return x, skipes


def test_ConvDecoder():
    x = torch.rand([1, 128, 8, 8])
    skip = torch.rand([1, 64, 16, 17])

    net = ConvDecoder(in_channels=128, out_channels=64)
    x = net(x, skip)
    print(x.size())


def test_Unet():
    x = torch.rand([1, 1, 512, 512]).cuda()
    net = Unet(in_channels=1, out_channels=3, channels=[32, 64, 128, 256]).cuda()
    x, skips = net(x)
    print(x.size())
    print([i.size() for i in skips])


def create_model(in_channels=1, out_channels=1, channels=[32, 64, 128, 256], *args):
    unet = Unet(in_channels=in_channels, out_channels=out_channels, channels=channels)
    return unet


if __name__ == "__main__":
    test_Unet()
