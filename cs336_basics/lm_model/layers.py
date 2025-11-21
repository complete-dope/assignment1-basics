# code all the layer that are required by the LM models
import math
from typing import Optional
import torch 
import torch.nn as nn 

class Linear(nn.Module):
    def __init__(self, in_features:int , out_features:int, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype= dtype
        factory_kwargs = {"dtype": dtype, 'device':device}

        std = math.sqrt(2.0 / (in_features + out_features))  # correct formula

        weight_init_for_linear_layer =  torch.nn.init.trunc_normal_(torch.empty(self.out_features, self.in_features, **factory_kwargs), mean = 0.0 , std = std , a = -3*std , b = 3*std)

        self.weights = torch.nn.Parameter(data = weight_init_for_linear_layer)


    def forward(self,x:torch.Tensor)-> torch.Tensor:

        assert x.shape[-1] == self.in_features, 'The shapes of tensors are mismatching'

        output = x @ self.weights.T # (d_out x d_in) @ (d_in x 1)

        if self.device:
            output = output.to(self.device)        
        return output


class Embedding(nn.Module):
    '''
    These are defined in torch longtensor format 

    num_embeddings are the size of vocab
    embedding_dim are the dims of embedding vectors 
    '''
    
    def __init__(self, num_embeddings:int, embedding_dim:int, device =None, dtype=None) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype':dtype} 
        self.weights = torch.nn.init.trunc_normal_(torch.empty(size = (num_embeddings, embedding_dim) ,**factory_kwargs), a = -3, b=3)

        self.embedding_matrix = torch.nn.Parameter(data = self.weights)
    
    def forward(self, token_ids : torch.Tensor)->torch.Tensor:

        return self.embedding_matrix[token_ids]


# this is applied parameter wise
class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float =1e-5, device = None , dtype=None) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device':device, 'dtype':dtype} 
        self.gain = torch.nn.Parameter(data=torch.ones(size = (d_model,), **factory_kwargs)) # this way all the input tokens across all batches will have the same layer gain values ( see feature dim5 got upscaled maybe you should also upscale it , that is rule transfer, different from batchnorm) 

    def forward(self, x:torch.Tensor)->torch.Tensor:
        '''
        input shape of x is :  N x T x E , and the gain is(1,1, E) 
        '''

        assert x.shape[-1] == self.d_model, f'X shape is : {x.shape} and d_model is {self.d_model}'

        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt((torch.sum(x*x, dim =-1) + self.eps) / self.d_model) # reverse square root value

        print(x.shape) #3,5
        print(rms.shape) # 3 
        print(self.gain.shape) # 5

        result =  x*self.gain
        # print('result is : ', result.shape)
        
        result = result*rms.unsqueeze(-1)
        
        return result.to(in_dtype)


# activation function using SwiGLU used in llama and qwen (that)
class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        pass

# left for Later evening
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float , d_k:int, max_seq_len:int ,device=None ) -> None:
        super().__init__()

        memory_space = torch.empty(size = (d_k , d_k))
        
        no_of_angles = d_k//2  #as every 2 dims take a single angle
        denominators = theta ** (2*torch.arange(1,no_of_angles) - 2)/d_k  #k dimensions 
        angles = ...

        pass

#3.5.4
def softmax(x:torch.Tensor, dim:int =-1):
    max_values = torch.argmax(x, dim = dim) # N size 
    
    maximum_in_each_row = x.gather(dim,index = max_values.unsqueeze(dim))
    
    # assert maximum_in_each_row.shape == max_values.shape, f'shape mismatch : {maximum_in_each_row.shape} and {max_values.shape} '

    normalised_x = x-maximum_in_each_row # broadcast operation 

    exp_norm_x = torch.exp(normalised_x)

    each_row_summed = torch.sum(exp_norm_x, dim =dim).unsqueeze(dim) # N x 1 

    return exp_norm_x / each_row_summed



def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask :Optional[torch.Tensor] = None)->torch.Tensor:

    *leading_dims,D = q.shape 
    
    print('q shape is : ',q.shape)
    print('k shape is : ', k.shape)
    out = torch.matmul(q, k.transpose(-1,-2))
    scaled_out = out * (D ** (-0.5))
    
    # masking : add a mask to this !  
    if mask is not None:
        # assert mask.shape == torch.Size((N,N)) ,f'Shape should be : {N} but found it to be {mask.shape}'
        masked_out = torch.where(mask == 0 , -torch.inf , scaled_out) # true =1 , false = 0,  
        assert masked_out.shape == scaled_out.shape
    
    else:
        masked_out = scaled_out

    scaled_softmax_out = softmax(masked_out, -1)

    attention_score = torch.matmul(scaled_softmax_out ,  v)
    return attention_score


# rest later on ! 

if __name__ == '__main__':
    random_value = torch.randn(size = (2,4))
    mask = torch.tensor(data = [[1,1], [1,0]], dtype = torch.int32)
    rmsnorm = RMSNorm(d_model = random_value.shape[-1])
    
    output = rmsnorm(random_value)
    print(output)
    print(output.shape)

    out = softmax(random_value, dim=-1)
    print('softmax out ',  out)

    out = scaled_dot_product_attention(random_value, random_value,random_value, mask)
    print('attention out :', out.shape)
