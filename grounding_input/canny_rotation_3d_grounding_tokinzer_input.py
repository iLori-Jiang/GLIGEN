import os 
import torch as th 

# JHY: NOTE: take a raw batch from the dataloader to prepare the input for position_net and downsample_net

# JHY: NOTE: version 3, to separately encode rotation information, and take 3d point cloud embedding

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

        # JHY: NOTE: add 3d info
        pc_embedding = batch['pointcloud_embedding_tensor']
        if len(pc_embedding.shape) == 2:
            self.pc_embedding_shape_len = 2
            self.pc_embedding_dim = pc_embedding.shape[1]
        elif len(pc_embedding.shape) == 3:
            pass

        self.batch, self.C, self.H, self.W = canny_edge.shape
        self.device = canny_edge.device
        self.dtype = canny_edge.dtype

        return {"canny_edge":canny_edge, "mask":mask, "rotation": rotation, "pointcloud_tokens": pc_embedding}


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
        mask = th.zeros(self.batch).type(dtype).to(device) 

        # JHY: NOTE: add rotation
        rotation = th.zeros(self.batch).type(dtype).to(device) 

        # JHY: NOTE: add 3d
        if self.pc_embedding_shape_len == 2:
            pc_embedding = th.zeros(self.batch, self.pc_embedding_dim).type(dtype).to(device) 

        return {"canny_edge":canny_edge, "mask":mask, "rotation": rotation, "pointcloud_tokens": pc_embedding}







