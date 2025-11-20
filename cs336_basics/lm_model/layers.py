# code all the layer that are required by the LM models
import math
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
    
    def forward(self, token_ids : torch.Tensor)->torch.Tensor:
        
        output_table = torch.randn(size = (self.num_embeddings, self.embedding_dim), dtype = torch.LongTensor, requires_grad = True)

        return output_table[token_ids]

