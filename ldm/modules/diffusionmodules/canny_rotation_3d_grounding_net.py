import torch
import torch.nn as nn
from .convnext import convnext_tiny
import numpy as np

# JHY: NOTE: version 3, separately process rotation and add 3d information


class PositionNet(nn.Module):
    def __init__(self, resize_input=448, out_dim=768, pointcloud_encoder=None):
        super().__init__()
        self.resize_input = resize_input
        self.down_factor = 32 # determined by the convnext backbone 
        self.out_dim = out_dim
        assert self.resize_input % self.down_factor == 0
        
        self.convnext_tiny_backbone = convnext_tiny(pretrained=True)
        
        '''
        resize_input = 448 输入图像的边长
        down_factor = 32 特征提取器的下采样因子

        计算下采样后的特征图尺寸：
        resize_input // down_factor = 448 // 32 = 14
        特征图的空间尺寸为 (14, 14)。
        
        因此，特征图中的总令牌数量为：

        num_tokens = 14 * 14 = 196
        '''
        self.num_tokens = (self.resize_input // self.down_factor) ** 2
        
        '''
        dimension for each token
        '''
        convnext_feature_dim = 768
        self.convnext_feature_dim = convnext_feature_dim
      
        # For Transformers
        self.linears = nn.Sequential(
            nn.Linear( convnext_feature_dim, 512),
            nn.SiLU(),
            nn.Linear( 512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_feature = torch.nn.Parameter(torch.zeros([convnext_feature_dim]))

        # JHY: NOTE: version 2, process rotation info
        self.fc_rotation = nn.Sequential(
                                nn.Linear(convnext_feature_dim + 2, 1024),
                                nn.SiLU(),
                                nn.Linear(1024, convnext_feature_dim)
                            )
        
        # JHY: NOTE: version 3, process point cloud
        self.pointcloud_encoder = pointcloud_encoder
        if pointcloud_encoder == "ULIP2_pointbert_embedding":
            self.pc_shape_dim = 1
            self.pc_dim = 1280

            # self.null_pc_tokens = torch.nn.Parameter(torch.zeros([self.pc_dim]))

            self.fc_pointcloud = nn.Linear(self.pc_dim, convnext_feature_dim)

            # JHY: NOTE: since the original embeddings are not normalized, use layer norm to normalize
            self.pointcloud_layer_norm = nn.LayerNorm(self.pc_dim)
            self.if_pc_layer_norm = True

            # For Transformers
            self.pos_embedding = nn.Parameter(
                                    torch.empty(
                                        1, 
                                        self.num_tokens + 1, 
                                        convnext_feature_dim
                                        ).normal_(std=0.02)
                                    )  # from BERT
        else:
            self.if_pc_layer_norm = False

            # For Transformers
            self.pos_embedding = nn.Parameter(
                                    torch.empty(
                                        1, 
                                        self.num_tokens, 
                                        convnext_feature_dim
                                        ).normal_(std=0.02)
                                    )  # from BERT


    def forward(self, canny_edge, mask, rotation, pointcloud_tokens):
        B = canny_edge.shape[0] 

        # --- Encode the image features

        # token from edge map 
        canny_edge = torch.nn.functional.interpolate(canny_edge, self.resize_input)
        canny_edge_feature = self.convnext_tiny_backbone(canny_edge)

        '''
        canny_edge_feature 是从特征提取器 convnext_tiny_backbone 得到的特征图
        其形状通常是 (B, C, H, W)，其中：

        B 是批量大小 batch size 。
        C 是特征图的通道数。
        H 和 W 是特征图的高度和宽度。

        假设 num_tokens 是特征图的空间位置数, 例如 H*W 
        C 是特征通道数，那么 canny_edge_feature 的形状实际上可能是 
        (B, C, num_tokens) = (B, C, H*W)
        因此, reshape 操作实际上是为了明确地表达这一形状，确保特征通道和空间位置之间的区分。

        and here C = convnext_feature_dim
        '''
        objs = canny_edge_feature.reshape(B, -1, self.num_tokens)

        # (B, C, H*W) -> (B, H*W (num_tokens), C (token_dim))
        objs = objs.permute(0, 2, 1) # N * Num_tokens * token_dim

        # --- Inpainting process

        # expand null token
        null_objs = self.null_feature.view(1,1,-1)
        null_objs = null_objs.repeat(B, self.num_tokens, 1)
        
        # mask replacing 
        mask = mask.view(-1,1,1)
        objs = objs*mask + null_objs*(1-mask)

        # --- JHY: NOTE: version 2, add rotation info

        # Encode the rotation angle
        theta = torch.deg2rad(rotation)
        theta_encoded = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        theta_encoded = theta_encoded.unsqueeze(1).expand(-1, self.num_tokens, -1)

        # Concatenate rotation information with image features
        combined_features = torch.cat((objs, theta_encoded), dim=-1)

        # Map combined features back to the original dimension
        combined_features = self.fc_rotation(combined_features)

        # --- JHY: NOTE: version 3, add pointcloud info

        # do the layer norm first if the tokens/embeddings are not normalized yet
        if self.if_pc_layer_norm:
            pointcloud_tokens = self.pointcloud_layer_norm(pointcloud_tokens)

        # project to the dim of img tokens
        # [B, D_pc] -> [B, D]
        pointcloud_tokens = self.fc_pointcloud(pointcloud_tokens)

        if self.pc_shape_dim == 1:
            # Unsqueeze at dim=1 to get [B, 1, D]
            pointcloud_tokens = pointcloud_tokens.unsqueeze(1)

        # Append the projected embedding to the tokens
        # concatenated_tokens: [B, N+1, D]
        combined_features = torch.cat((combined_features, pointcloud_tokens), dim=1)

        # --- Transformers preprocess

        # add pos 
        combined_features = combined_features + self.pos_embedding

        # fuse them 
        combined_features = self.linears(combined_features)
      
        return combined_features


if __name__ == "__main__":
    # 初始化模型
    model = PositionNet(pointcloud_encoder="ULIP2_pointbert_embedding")

    example_img = torch.rand(4, 3, 512, 512)
    example_pc_embedding = torch.rand(4, 1280)
    rotation = torch.rand(4,)
    mask = torch.tensor(1.0)

    # 前向传播，计算输出
    output = model(example_img, mask, rotation, example_pc_embedding)

    # 打印输出尺寸
    print("输出尺寸:", output.shape)    # torch.Size([4, 196 + 1, 768])

    # run by: python -m ldm.modules.diffusionmodules.canny_rotation_3d_grounding_net
