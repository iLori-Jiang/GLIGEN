import os 
import torch as th 

# JHY: NOTE: take a raw batch from the dataloader to prepare the input for position_net and downsample_net

class GroundingDSInput:
    def __init__(self):
        pass 

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        extra input for diffusion model. 
        """
        return batch['canny_edge']

