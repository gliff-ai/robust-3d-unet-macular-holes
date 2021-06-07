import torch


# Modified version of residual model to cope better with our dataset sizes
# Input is 128x128x49 and output is 128x128x49
class UNet3DResidualProposal(torch.nn.Module):
    def __init__(self):
        super(UNet3DResidualProposal, self).__init__()
        # begin layer 0 part 1
        self.conv_down_00 = torch.nn.Conv3d(1, 32, 3, padding=1)
        self.bn_down_00 = torch.nn.BatchNorm3d(32)
        self.relu_down_00 = torch.nn.ReLU()
        # begin residual 1
        self.conv_down_01 = torch.nn.Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        self.bn_down_01 = torch.nn.BatchNorm3d(32)
        self.relu_down_01 = torch.nn.ReLU()
        self.conv_down_02 = torch.nn.Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        self.bn_down_02 = torch.nn.BatchNorm3d(32)
        # end residual 1
        self.relu_down_02 = torch.nn.ReLU()
        # begin residual 2
        self.conv_down_03 = torch.nn.Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        self.bn_down_03 = torch.nn.BatchNorm3d(32)
        self.relu_down_03 = torch.nn.ReLU()
        self.conv_down_04 = torch.nn.Conv3d(32, 32, 3, stride=1, padding=1, bias=False)
        self.bn_down_04 = torch.nn.BatchNorm3d(32)
        # end residual 2
        self.relu_down_04 = torch.nn.ReLU()
        # end layer 0 part 1
        ## begin layer 0 part 2
        self.conv_down_05 = torch.nn.Conv3d(32, 64, 3, padding=1)
        self.bn_down_05 = torch.nn.BatchNorm3d(64)
        self.relu_down_05 = torch.nn.ReLU()
        # begin residual 1
        self.conv_down_06 = torch.nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn_down_06 = torch.nn.BatchNorm3d(64)
        self.relu_down_06 = torch.nn.ReLU()
        self.conv_down_07 = torch.nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn_down_07 = torch.nn.BatchNorm3d(64)
        # end residual 1
        self.relu_down_07 = torch.nn.ReLU()
        # begin residual 2
        self.conv_down_08 = torch.nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn_down_08 = torch.nn.BatchNorm3d(64)
        self.relu_down_08 = torch.nn.ReLU()
        self.conv_down_09 = torch.nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn_down_09 = torch.nn.BatchNorm3d(64)
        # end residual 2
        self.relu_down_09 = torch.nn.ReLU()
        ## end layer 0 part 2
        self.max_pool_down_00 = torch.nn.MaxPool3d(2)

        self.conv_down_10 = torch.nn.Conv3d(64, 128, 3, padding=1)
        self.bn_down_10 = torch.nn.BatchNorm3d(128)
        self.relu_down_10 = torch.nn.ReLU()
        # begin residual 1
        self.conv_down_11 = torch.nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn_down_11 = torch.nn.BatchNorm3d(128)
        self.relu_down_11 = torch.nn.ReLU()
        self.conv_down_12 = torch.nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn_down_12 = torch.nn.BatchNorm3d(128)
        # end residual 1
        self.relu_down_12 = torch.nn.ReLU()
        # begin residual 2
        self.conv_down_13 = torch.nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn_down_13 = torch.nn.BatchNorm3d(128)
        self.relu_down_13 = torch.nn.ReLU()
        self.conv_down_14 = torch.nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn_down_14 = torch.nn.BatchNorm3d(128)
        # end residual 2
        self.relu_down_14 = torch.nn.ReLU()
        self.max_pool_down_10 = torch.nn.MaxPool3d(2)

        self.conv_bottom_20 = torch.nn.Conv3d(128, 256, 3, padding=1)
        self.bn_bottom_20 = torch.nn.BatchNorm3d(256)
        self.relu_bottom_20 = torch.nn.ReLU()
        # begin residual 1
        self.conv_bottom_21 = torch.nn.Conv3d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn_bottom_21 = torch.nn.BatchNorm3d(256)
        self.relu_bottom_21 = torch.nn.ReLU()
        self.conv_bottom_22 = torch.nn.Conv3d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn_bottom_22 = torch.nn.BatchNorm3d(256)
        # end residual 1
        self.relu_bottom_22 = torch.nn.ReLU()
        # begin residual 2
        self.conv_bottom_23 = torch.nn.Conv3d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn_bottom_23 = torch.nn.BatchNorm3d(256)
        self.relu_bottom_23 = torch.nn.ReLU()
        self.conv_bottom_24 = torch.nn.Conv3d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn_bottom_24 = torch.nn.BatchNorm3d(256)
        # end residual 2
        self.relu_bottom_24 = torch.nn.ReLU()
        self.conv_bottom_25 = torch.nn.Conv3d(256, 128, 1)

        self.conv_up_10 = torch.nn.Conv3d(256, 128, 3, padding=1)
        self.bn_up_10 = torch.nn.BatchNorm3d(128)
        self.relu_up_10 = torch.nn.ReLU()
        # begin residual 1
        self.conv_up_11 = torch.nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn_up_11 = torch.nn.BatchNorm3d(128)
        self.relu_up_11 = torch.nn.ReLU()
        self.conv_up_12 = torch.nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn_up_12 = torch.nn.BatchNorm3d(128)
        # end residual 1
        self.relu_up_12 = torch.nn.ReLU()
        # begin residual 2
        self.conv_up_13 = torch.nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn_up_13 = torch.nn.BatchNorm3d(128)
        self.relu_up_13 = torch.nn.ReLU()
        self.conv_up_14 = torch.nn.Conv3d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn_up_14 = torch.nn.BatchNorm3d(128)
        # end residual 2
        self.relu_up_14 = torch.nn.ReLU()
        self.conv_up_15 = torch.nn.Conv3d(128, 64, 1)

        self.conv_up_00 = torch.nn.Conv3d(128, 64, 3, padding=1)
        self.bn_up_00 = torch.nn.BatchNorm3d(64)
        self.relu_up_00 = torch.nn.ReLU()
        # begin residual 1
        self.conv_up_01 = torch.nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn_up_01 = torch.nn.BatchNorm3d(64)
        self.relu_up_01 = torch.nn.ReLU()
        self.conv_up_02 = torch.nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn_up_02 = torch.nn.BatchNorm3d(64)
        # end residual 1
        self.relu_up_02 = torch.nn.ReLU()
        # begin residual 2
        self.conv_up_03 = torch.nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn_up_03 = torch.nn.BatchNorm3d(64)
        self.relu_up_03 = torch.nn.ReLU()
        self.conv_up_04 = torch.nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn_up_04 = torch.nn.BatchNorm3d(64)
        # end residual 2
        self.relu_up_04 = torch.nn.ReLU()
        # TODO: This is different than 2,1,1 which original used. Look into this
        self.conv_up_05 = torch.nn.ConvTranspose3d(64, 1, kernel_size=[2, 1, 1], stride=1, padding=0)

    def forward(self, x):
        # DOWN CONV
        # begin layer 0 part 1
        x = self.relu_down_00(self.bn_down_00(self.conv_down_00(x)))
        # begin residual 1
        x_res = self.relu_down_01(self.bn_down_01(self.conv_down_01(x)))
        x_res = self.bn_down_02(self.conv_down_02(x_res))
        x = x + x_res
        # end residual 1
        x = self.relu_down_02(x)
        # begin residual 2
        x_res = self.relu_down_03(self.bn_down_03(self.conv_down_03(x)))
        x_res = self.bn_down_04(self.conv_down_04(x_res))
        x = x + x_res
        # end residual 2
        x = self.relu_down_04(x)
        # end layer 0 part 1
        ## begin layer 0 part 2
        x = self.relu_down_05(self.bn_down_05(self.conv_down_05(x)))
        # begin residual 1
        x_res = self.relu_down_06(self.bn_down_06(self.conv_down_06(x)))
        x_res = self.bn_down_07(self.conv_down_07(x_res))
        x = x + x_res
        # end residual 1
        x = self.relu_down_07(x)
        # begin residual 2
        x_res = self.relu_down_08(self.bn_down_08(self.conv_down_08(x)))
        x_res = self.bn_down_09(self.conv_down_09(x_res))
        x = x + x_res
        # end residual 2
        x = self.relu_down_09(x)
        ## end layer 0 part 2
        first_layer_output = x
        x = self.max_pool_down_00(x)

        x = self.relu_down_10(self.bn_down_10(self.conv_down_10(x)))
        # begin residual 1
        x_res = self.relu_down_11(self.bn_down_11(self.conv_down_11(x)))
        x_res = self.bn_down_12(self.conv_down_12(x_res))
        x = x + x_res
        # end residual 1
        x = self.relu_down_12(x)
        # begin residual 2
        x_res = self.relu_down_13(self.bn_down_13(self.conv_down_13(x)))
        x_res = self.bn_down_14(self.conv_down_14(x_res))
        x = x + x_res
        # end residual 2
        x = self.relu_down_14(x)
        second_layer_output = x
        x = self.max_pool_down_10(x)

        # BOTTOM
        x = self.relu_bottom_20(self.bn_bottom_20(self.conv_bottom_20(x)))
        # begin residual 1
        x_res = self.relu_bottom_21(self.bn_bottom_21(self.conv_bottom_21(x)))
        x_res = self.bn_bottom_22(self.conv_bottom_22(x_res))
        x = x + x_res
        # end residual 1
        x = self.relu_bottom_22(x)
        # begin residual 2
        x_res = self.relu_bottom_23(self.bn_bottom_23(self.conv_bottom_23(x)))
        x_res = self.bn_bottom_24(self.conv_bottom_24(x_res))
        x = x + x_res
        # end residual 2
        x = self.relu_bottom_24(x)
        x = self.conv_bottom_25(torch.nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=False))

        # UP CONV
        dx = x.size(-1) - second_layer_output.size(-1)
        dy = x.size(-2) - second_layer_output.size(-2)
        dz = x.size(-3) - second_layer_output.size(-3)
        second_layer_output = torch.nn.functional.pad(second_layer_output, (dx//2, (dx+1)//2, dy//2, (dy+1)//2, dz//2, (dz+1)//2))
        x = torch.cat((x, second_layer_output), dim=1)
        x = self.relu_up_10(self.bn_up_10(self.conv_up_10(x)))
        # begin residual 1
        x_res = self.relu_up_11(self.bn_up_11(self.conv_up_11(x)))
        x_res = self.bn_up_12(self.conv_up_12(x_res))
        x = x + x_res
        # end residual 1
        x = self.relu_up_12(x)
        # begin residual 2
        x_res = self.relu_up_13(self.bn_up_13(self.conv_up_13(x)))
        x_res = self.bn_up_14(self.conv_up_14(x_res))
        x = x + x_res
        # end residual 2
        x = self.relu_up_14(x)
        x = self.conv_up_15(torch.nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=False))

        dx = x.size(-1) - first_layer_output.size(-1)
        dy = x.size(-2) - first_layer_output.size(-2)
        dz = x.size(-3) - first_layer_output.size(-3)
        first_layer_output = torch.nn.functional.pad(first_layer_output, (dx//2, (dx+1)//2, dy//2, (dy+1)//2, dz//2, (dz+1)//2))
        x = torch.cat((x, first_layer_output), dim=1)
        x = self.relu_up_00(self.bn_up_00(self.conv_up_00(x)))
        # begin residual 1
        x_res = self.relu_up_01(self.bn_up_01(self.conv_up_01(x)))
        x_res = self.bn_up_02(self.conv_up_02(x_res))
        x = x + x_res
        # end residual 1
        x = self.relu_up_02(x)
        # begin residual 2
        x_res = self.relu_up_03(self.bn_up_03(self.conv_up_03(x)))
        x_res = self.bn_up_04(self.conv_up_04(x_res))
        x = x + x_res
        # end residual 2
        x = self.relu_up_04(x)

        x = self.conv_up_05(x)

        return x
