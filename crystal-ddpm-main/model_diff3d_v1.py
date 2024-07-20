import torch
import torch.nn as nn
import torch.nn.functional as F

# num_atom = 86
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.res_bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = self.res_conv(x)
        residual = self.res_bn(residual)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        return F.relu(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet3D, self).__init__()
        self.encoder1 = ConvBlock3D(in_channels, 32)
        self.encoder2 = ConvBlock3D(32, 64)
        self.encoder3 = ConvBlock3D(64, 128)

        self.pool = nn.MaxPool3d(2)

        self.middle = ConvBlock3D(128, 256)

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock3D(256, 128)

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock3D(128, 64)

        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock3D(64, 32)

        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))

        # Middle path
        mid = self.middle(self.pool(enc3))

        # Decoder path
        dec3 = self.upconv3(mid)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)



# if __name__ == "__main__":
#     # Example usage
#     model = UNet3D(in_channels=1)  # 修改输入通道数
#     batch_size = 4
#     grid_size = 64
#     input_tensor = torch.randn(batch_size, 1, grid_size, grid_size, grid_size)  # 修改输入张量形状
#     output = model(input_tensor)
#     print(output.shape)  # torch.Size([4, 1, 64, 64, 64])
