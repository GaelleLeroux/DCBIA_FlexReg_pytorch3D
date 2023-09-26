from typing import Any
import torch

# def wrap_error(func):
#     nbloop =1
#     def wrapper(arg_point,F):
#         error = False
#         nbloop = 1
#         neighbours = torch.tensor([]).cuda()
#         while error :
#             arg_point_loop = 
#             try :
#                 neighbours_ = func(arg_point,F)
#             except RuntimeError :
#                 error = True

#             neighbours = torch.cat([neighbours,neighbours_])

#         return neighbours

#     return wrapper


# class WrappError:
#     def __init__(self,func) -> None:
#         self._func = func
#         self.nbloop = 1

#     def __call__(self,arg_point,F) :
#         error = False
#         neighbours = torch.tensor([]).cuda()
#         while error :
#             arg_point_loop = 
#             try :
#                 neighbours_ = func(arg_point,F)
#             except RuntimeError :
#                 error = True

#             neighbours = torch.cat([neighbours,neighbours_])

#         return neighbours
    

#     def limitTest()


def Difference(t1,t2):
    # print(f't2 : {t2}')
    # print(f't1 shape {t1.shape}')
    t1 = t1.unsqueeze(0).expand(len(t2),-1)
    t2 = t2.unsqueeze(1)
    d = torch.count_nonzero(t1 -t2,dim=-1)
    arg = torch.argwhere(d == t1.shape[1])
    dif = torch.unique(t2[arg])
    return dif


def Neighbours(arg_point,F):
    neighbours = torch.tensor([]).cuda()
    F2 = F.unsqueeze(0).expand(len(arg_point),-1,-1)
    arg_point = arg_point.unsqueeze(1).unsqueeze(2)
    arg = torch.argwhere((F2-arg_point) == 0)
    # print(f' arg {arg.shape}, arg : {arg}')

    neighbours = torch.unique(F[arg[:,1],:])
    return neighbours







def Dilation(arg_point,F,texture):
    # print(f'arg point {arg_point}')
    arg_point = torch.tensor([arg_point]).cuda().to(torch.int64)
    F = F.cuda()
    texture = texture.cuda()
    neighbour = Neighbours(arg_point,F)
    arg_texture = torch.argwhere(texture == 1).squeeze()
    # dif = NoIntersection(arg_texture,neighbour)
    dif = neighbour.to(torch.int64)
    dif  = Difference(arg_texture,dif)
    n = 0
    while len(dif)!= 0 :#and n < 50:
        # print(f'n = {n}, len : {len(dif)}')
        texture[dif] = 1
        neighbour = Neighbours(dif,F)
        arg_texture = torch.argwhere(texture == 1).squeeze()
        # dif = NoIntersection(arg_texture,neighbour)
        dif = neighbour.to(torch.int64)
        dif  = Difference(arg_texture,dif)
        n+=1
    return texture