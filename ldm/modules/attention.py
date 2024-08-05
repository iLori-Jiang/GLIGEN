from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

# from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
from torch.utils import checkpoint


# JHY: NOTE: attention functions and classes


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):

    # JHY: NOTE: anti-bottleneck stucture
    # dim -> 4 * dim -> dim

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)





class CrossAttention(nn.Module):

    # JHY: NOTE: 用于在一个输入序列（query）和另一个序列（key、value）之间计算注意力

    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        # NOTE: project the dimension of the token to the same dim
        # in this way, the dimension of the token of query, key, value 
        # !!!DON'T NEED TO BE THE SAME!!!
        # can consider this as project different information from different space
        # into a common space
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)


        self.to_out = nn.Sequential( nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )


    def fill_inf_from_mask(self, sim, mask):

        # 用于处理 mask（掩码）以屏蔽无效位置

        if mask is not None:
            B,M = mask.shape
            mask = mask.unsqueeze(1).repeat(1,self.heads,1).reshape(B*self.heads,1,-1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim 


    def forward(self, x, key, value, mask=None):

        # for example, in the prompt X image,
        # x -> image tokens
        # key -> prompt tokens
        # value -> prompt tokens (key and value are the same)

        '''
        NOTE: key和value是外部信息源, 可以来自同一个外部来源(如文本提示), key用于决定每个外部信息的重要性(即与查询的相关性), 而value是实际从外部注入的信息
        
        根据查询在key空间中的相似度来决定哪些value是相关的。这类似于信息检索的过程: 我们有一个查询query, 然后在一组key中找到最相关的, 并取回对应的value。
        这种方式确保了查询主导了哪些信息会被引入, 查询query是信息融合的主导者
        '''

        # NOTE: following dimension also show:
        # query (from x) is of N num_token
        # key value are both of M num_token

        # B * N * query_dim -> B * N * (H*C)
        q = self.to_q(x)
        # B * M * key_dim -> B * M * (H*C)
        k = self.to_k(key)
        # B * M * value_dim -> B * M * (H*C)
        v = self.to_v(value)
   
        B, N, HC = q.shape 
        _, M, _ = key.shape
        H = self.heads
        C = HC // H     # C is the dim of each head

        # 将head分离出来，对每个head进行单独的cross attention
        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H) * N * C
        k = k.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H) * M * C
        v = v.view(B,M,H,C).permute(0,2,1,3).reshape(B*H,M,C) # (B*H) * M * C

        # 计算注意力得分similarity
        # query @ key.T
        # (N, C) @ (C, M) -> (N, M)
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale # (B*H) * N * M

        # add mask 应用掩码
        self.fill_inf_from_mask(sim, mask)

        # 计算注意力权重
        attn = sim.softmax(dim=-1) # (B*H) * N * M

        # 计算加权值
        # atten @ value
        # (N, M) @ (M, C) -> (N, C)
        out = torch.einsum('b i j, b j d -> b i d', attn, v) # (B*H) * N * C

        # 重新调换顺序，并将head与每个head的dim融合
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B * N * (H*C)

        # 将融合后的信息project到与query相同的dim，与input一致
        # 与后续的query作相加（Residual）
        return self.to_out(out)




class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward(self, x):
        q = self.to_q(x) # B * N * (H*C)
        k = self.to_k(x) # B * N * (H*C)
        v = self.to_v(x) # B * N * (H*C)

        B, N, HC = q.shape 
        H = self.heads
        C = HC // H 

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        # similarity
        # query @ key.T
        # (N, C) @ (C, N) -> (N, N)
        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N

        # attention
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        # attention @ value
        # (N, N) @ (N, C) -> (N, C)
        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C

        # merge the multi head info
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        # project back to input dim
        return self.to_out(out)



class GatedCrossAttentionDense(nn.Module):

    # JHY: NOTE: simply the cross attention between image latent and control signal
    # the same with prompt

    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()
        
        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head) 
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  

    def forward(self, x, objs):

        # query: x (image)
        # key: objs (control)
        # value: objs (control)
        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn( self.norm1(x), objs, objs)  
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) ) 
        
        return x 


class GatedSelfAttentionDense(nn.Module):

    # JHY: NOTE: concatenate query(image) and context(control), then do self attention
    # then cut the context part off

    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        # NOTE: match the dim of token from different source
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs):

        # shape: B, N, D
        # N_visual = number of token of image latent
        N_visual = x.shape[1]

        # project the token dim of control to the same as image latent
        objs = self.linear(objs)

        # concatenate the [image latent, control]
        # do the self attention
        # only keep the [image latent] part, discard the [control] part
        x = x + self.scale*torch.tanh(self.alpha_attn) * self.attn(  self.norm1(torch.cat([x,objs],dim=1))  )[:,0:N_visual,:]
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x 






class GatedSelfAttentionDense2(nn.Module):

    # JHY: NOTE: 2th version of gated self attention

    def __init__(self, query_dim, context_dim,  n_heads, d_head):
        super().__init__()
        
        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)) )
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)) )

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1  


    def forward(self, x, objs):

        B, N_visual, _ = x.shape
        B, N_ground, _ = objs.shape

        objs = self.linear(objs)
        
        # sanity check 
        size_v = math.sqrt(N_visual)
        size_g = math.sqrt(N_ground)
        assert int(size_v) == size_v, "Visual tokens must be square rootable"
        assert int(size_g) == size_g, "Grounding tokens must be square rootable"
        size_v = int(size_v)
        size_g = int(size_g)

        # select grounding token and resize it to visual token size as residual 
        # concatenate the [image latent, control]
        # do the self attention
        # only keep the [control] part
        out = self.attn(  self.norm1(torch.cat([x,objs],dim=1))  )[:,N_visual:,:]

        # -> (B, token_dim, num_token)
        # -> (B, token_dim, sqrt(num_token), sqrt(num_token))
        out = out.permute(0,2,1).reshape( B,-1,size_g,size_g )

        # interpolate to the same size as sqrt(num_token) of img
        out = torch.nn.functional.interpolate(out, (size_v,size_v), mode='bicubic')

        # -> (B, token_dim, num_token_img)
        # -> (B, num_token_img, token_dim)
        residual = out.reshape(B,-1,N_visual).permute(0,2,1)
        
        # add residual to visual feature 
        x = x + self.scale*torch.tanh(self.alpha_attn) * residual
        x = x + self.scale*torch.tanh(self.alpha_dense) * self.ff( self.norm2(x) )  
        
        return x 





class BasicTransformerBlock(nn.Module):

    # JHY: NOTE: the detailed transformer block of the spatial transformer class below

    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True):
        super().__init__()
        self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)  
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head)  
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint

        # JHY: NOTE: attention type with control signal
        if fuser_type == "gatedSA":
            # note key_dim here actually is context_dim
            self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head) 
        elif fuser_type == "gatedSA2":
            # note key_dim here actually is context_dim
            self.fuser = GatedSelfAttentionDense2(query_dim, key_dim, n_heads, d_head) 
        elif fuser_type == "gatedCA":
            self.fuser = GatedCrossAttentionDense(query_dim, key_dim, value_dim, n_heads, d_head) 
        else:
            assert False 


    def forward(self, x, context, objs):
#        return checkpoint(self._forward, (x, context, objs), self.parameters(), self.use_checkpoint)
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs)
        else:
            return self._forward(x, context, objs)

    def _forward(self, x, context, objs): 
        x = self.attn1( self.norm1(x) ) + x 
        x = self.fuser(x, objs) # identity mapping in the beginning 
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


# JHY: NOTE: the transformer for the U Net (downsample, bottleneck, upsample)
class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, fuser_type=None, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        query_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        
        self.proj_in = nn.Conv2d(in_channels,
                                 query_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context, objs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context, objs)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
