import torch
import torch.nn as nn

# JHY: NOTE: version 3, accept RGB 3 channels instead of 1 gray channel

class GroundingDownsampler(nn.Module):
    def __init__(self, resize_input=256, out_dim=8, hidden_dim=4):
        super().__init__()

        # JHY: NOTE: input is RGB
        self.INPUT_DIM = 3
        
        # JHY: NOTE: output is the same size as img latent
        self.OUTPUT_IMG_SIZE = 64

        self.resize_input = resize_input
        self.out_dim = out_dim 
        self.hidden_dim = hidden_dim

        # conv layers
        self.layers = nn.Sequential(
            # [3, 256, 256] -> [4, 128, 128]
            nn.Conv2d(in_channels=self.INPUT_DIM, out_channels=self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            # [4, 128, 128] -> [8, 64, 64]
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.out_dim, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, grounding_extra_input):

        out = torch.nn.functional.interpolate(grounding_extra_input, (self.resize_input,self.resize_input), mode='bicubic')
        out = self.layers(out)

        assert out.shape[1] == self.out_dim 

        assert out.shape[2] == self.OUTPUT_IMG_SIZE and out.shape[3] == self.OUTPUT_IMG_SIZE

        return out


if __name__ == "__main__":
    # 初始化模型
    model = GroundingDownsampler(resize_input=256, out_dim=8)

    # 创建一个示例输入张量，形状为 [batch_size, channels, height, width]
    example_input = torch.rand(1, 3, 512, 512)  # RGB图像输入

    # 前向传播，计算输出
    output = model(example_input)

    # 打印输出尺寸
    print("输出尺寸:", output.shape)
