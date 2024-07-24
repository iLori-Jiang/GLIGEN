import os 
import torch as th 

# JHY: NOTE: take a raw batch from the dataloader to prepare the input for position_net and downsample_net

# JHY: NOTE: version 2, to separately encode rotation information

class GroundingNetInput:
    def __init__(self):
        self.set = False 

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        input only for the ground tokenizer. 
        """

        self.set = True

        canny_edge=batch['canny_edge'] 
        mask=batch['mask']

        # JHY: NOTE: add rotation
        rotation = batch['rotation']
        assert rotation.shape[0] == canny_edge.shape[0], "batch size of rotation is different from canny_edge"

        self.batch, self.C, self.H, self.W = canny_edge.shape
        self.device = canny_edge.device
        self.dtype = canny_edge.dtype

        return {"canny_edge":canny_edge, "mask":mask, "rotation": rotation}


    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this funcion"
        batch =  self.batch  if batch  is None else batch
        device = self.device if device is None else device
        dtype = self.dtype   if dtype  is None else dtype

        canny_edge = th.zeros(self.batch, self.C, self.H, self.W).type(dtype).to(device) 
        mask = th.zeros(batch).type(dtype).to(device) 

        # JHY: NOTE: add rotation
        rotation = th.zeros(batch).type(dtype).to(device) 

        return {"canny_edge":canny_edge, "mask":mask, "rotation": rotation}







