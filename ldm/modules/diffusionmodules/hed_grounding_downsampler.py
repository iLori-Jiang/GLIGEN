import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F



class GroundingDownsampler(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        # No learnable params for hed edge map, just downsample it with bicubic

    def forward(self, grounding_extra_input):
        
        # JHY: NOTE: HED grounding

        # print("")
        # print("GroundingDownsampler: BEFORE: grounding_extra_input.shape: ")
        # print(grounding_extra_input.shape)
        # print("")

        '''
        before
        grounding_extra_input.shape
        torch.Size([4, 3, 512, 512])
        '''

        # this is actually gary scale, but converted to rgb in dataset, information redudant 
        grounding_extra_input = grounding_extra_input[:,0].unsqueeze(1)

        # print("")
        # print("GroundingDownsampler: AFTER: grounding_extra_input.shape: ")
        # print(grounding_extra_input.shape)
        # print("")

        '''
        after
        grounding_extra_input.shape
        torch.Size([4, 1, 512, 512])
        '''


        out = torch.nn.functional.interpolate(grounding_extra_input, (64,64), mode='bicubic')

        # print("")
        # print("GroundingDownsampler: out.shape: ")
        # print(out.shape)
        # print("")

        '''
        print(out.shape)
        torch.Size([4, 1, 64, 64])
        '''


        assert out.shape[1] == self.out_dim 

        # print("")
        # print("GroundingDownsampler: self.out_dim: ")
        # print(self.out_dim)
        # print("")

        '''
        print(self.out_dim)
        1
        '''

        return out


