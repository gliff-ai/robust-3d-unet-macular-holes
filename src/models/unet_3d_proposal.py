import torch


# Modified version of Cicek's model to cope better with our Intogral's dataset sizes
# Input is 128x128x49 and output is 128x128x49
class UNet3DProposal(torch.nn.Module):
    def __init__(self):
        super(UNet3DProposal, self).__init__()
        self.conv_down_00 = torch.nn.Conv3d(1, 32, 3, padding=1)
        self.bn_down_00 = torch.nn.BatchNorm3d(32)
        self.relu_down_00 = torch.nn.ReLU()
        self.conv_down_01 = torch.nn.Conv3d(32, 32, 3, padding=1)
        self.bn_down_01 = torch.nn.BatchNorm3d(32)
        self.relu_down_01 = torch.nn.ReLU()
        self.conv_down_02 = torch.nn.Conv3d(32, 64, 3, padding=1)
        self.bn_down_02 = torch.nn.BatchNorm3d(64)
        self.relu_down_02 = torch.nn.ReLU()
        self.conv_down_03 = torch.nn.Conv3d(64, 64, 3, padding=1)
        self.bn_down_03 = torch.nn.BatchNorm3d(64)
        self.relu_down_03 = torch.nn.ReLU()
        self.max_pool_down_00 = torch.nn.MaxPool3d(2)

        self.conv_down_10 = torch.nn.Conv3d(64, 128, 3, padding=1)
        self.bn_down_10 = torch.nn.BatchNorm3d(128)
        self.relu_down_10 = torch.nn.ReLU()
        self.conv_down_11 = torch.nn.Conv3d(128, 128, 3, padding=1)
        self.bn_down_11 = torch.nn.BatchNorm3d(128)
        self.relu_down_11 = torch.nn.ReLU()
        self.max_pool_down_10 = torch.nn.MaxPool3d(2)

        self.conv_bottom_20 = torch.nn.Conv3d(128, 256, 3, padding=1)
        self.bn_bottom_20 = torch.nn.BatchNorm3d(256)
        self.relu_bottom_20 = torch.nn.ReLU()
        self.conv_bottom_21 = torch.nn.Conv3d(256, 256, 3, padding=1)
        self.bn_bottom_21 = torch.nn.BatchNorm3d(256)
        self.relu_bottom_21 = torch.nn.ReLU()
        self.conv_bottom_22 = torch.nn.Conv3d(256, 128, 1)

        self.conv_up_10 = torch.nn.Conv3d(256, 128, 3, padding=1)
        self.bn_up_10 = torch.nn.BatchNorm3d(128)
        self.relu_up_10 = torch.nn.ReLU()
        self.conv_up_11 = torch.nn.Conv3d(128, 128, 3, padding=1)
        self.bn_up_11 = torch.nn.BatchNorm3d(128)
        self.relu_up_11 = torch.nn.ReLU()
        self.conv_up_12 = torch.nn.Conv3d(128, 64, 1)

        self.conv_up_00 = torch.nn.Conv3d(128, 64, 3, padding=1)
        self.bn_up_00 = torch.nn.BatchNorm3d(64)
        self.relu_up_00 = torch.nn.ReLU()
        self.conv_up_01 = torch.nn.Conv3d(64, 64, 3, padding=1)
        self.bn_up_01 = torch.nn.BatchNorm3d(64)
        self.relu_up_01 = torch.nn.ReLU()
        # TODO: This is different than 2,1,1 which original used. Look into this
        #self.conv_up_02 = torch.nn.ConvTranspose3d(64, 1, kernel_size=[1, 1, 2], stride=1, padding=0)
        self.conv_up_02 = torch.nn.ConvTranspose3d(64, 1, kernel_size=[2, 1, 1], stride=1, padding=0)

    def forward(self, x):
        # DOWN CONV
        x = self.relu_down_00(self.bn_down_00(self.conv_down_00(x)))
        x = self.relu_down_01(self.bn_down_01(self.conv_down_01(x)))
        x = self.relu_down_02(self.bn_down_02(self.conv_down_02(x)))
        x = self.relu_down_03(self.bn_down_03(self.conv_down_03(x)))
        first_layer_output = x
        x = self.max_pool_down_00(x)

        x = self.relu_down_10(self.bn_down_10(self.conv_down_10(x)))
        x = self.relu_down_11(self.bn_down_11(self.conv_down_11(x)))
        second_layer_output = x
        x = self.max_pool_down_10(x)

        # BOTTOM
        x = self.relu_bottom_20(self.bn_bottom_20(self.conv_bottom_20(x)))
        x = self.relu_bottom_21(self.bn_bottom_21(self.conv_bottom_21(x)))
        x = self.conv_bottom_22(torch.nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=False))

        # UP CONV
        dx = x.size(-1) - second_layer_output.size(-1)
        dy = x.size(-2) - second_layer_output.size(-2)
        dz = x.size(-3) - second_layer_output.size(-3)
        second_layer_output = torch.nn.functional.pad(second_layer_output, (dx//2, (dx+1)//2, dy//2, (dy+1)//2, dz//2, (dz+1)//2))
        x = torch.cat((x, second_layer_output), dim=1)
        x = self.relu_up_10(self.bn_up_10(self.conv_up_10(x)))
        x = self.relu_up_11(self.bn_up_11(self.conv_up_11(x)))
        x = self.conv_up_12(torch.nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=False))

        dx = x.size(-1) - first_layer_output.size(-1)
        dy = x.size(-2) - first_layer_output.size(-2)
        dz = x.size(-3) - first_layer_output.size(-3)
        first_layer_output = torch.nn.functional.pad(first_layer_output, (dx//2, (dx+1)//2, dy//2, (dy+1)//2, dz//2, (dz+1)//2))
        x = torch.cat((x, first_layer_output), dim=1)
        x = self.relu_up_00(self.bn_up_00(self.conv_up_00(x)))
        x = self.relu_up_01(self.bn_up_01(self.conv_up_01(x)))
        x = self.conv_up_02(x)

        return x
