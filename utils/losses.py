import torch
from torch import nn
import torch.distributed as dist

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class MLLoss(nn.Module):
    def __init__(self, s=64.0):
        super(MLLoss, self).__init__()
        self.s = s
    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta.mul_(self.s)
        return cos_theta

class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret



class ACosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(ACosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, embeddings_t, label,momentum):
        embbedings = l2_norm(embbedings, axis=1)
        embeddings_t = l2_norm(embeddings_t, axis=1)
        with torch.no_grad():
            self.kernel[:,label]= momentum *self.kernel[:,label] + (1. - momentum) *(embeddings_t.T)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret

class AdaptiveACosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35, adaptive_weighted_alpha=True):
        super(AdaptiveACosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.cosine_sim=torch.nn.CosineSimilarity()
        self.adaptive_weighted_alpha=adaptive_weighted_alpha

    def forward(self, embbedings, embeddings_t, label):
        embbedings = l2_norm(embbedings, axis=1)
        embeddings_t = l2_norm(embeddings_t, axis=1)
        with torch.no_grad():
            cos_theta_tmp = self.cosine_sim(embbedings, embeddings_t)  # # get alpha value
            cos_theta_tmp = cos_theta_tmp.clamp(-1, 1)
            if self.adaptive_weighted_alpha: # weighted alpha
                lam = self.cosine_sim(self.kernel[:, label].T, embeddings_t).clamp(1e-6, 1).view(-1, 1) # similarity between  teacher embeddings and class centers
                target_logit = cos_theta_tmp.view(-1, 1) * lam
            else:
                target_logit = cos_theta_tmp.view(-1, 1) - self.t_alpha
            target_logit_mean = target_logit.clamp(1e-6, 1.0).T
            self.kernel[:, label] = (target_logit_mean) * self.kernel[:, label] + (1. - target_logit_mean) * (
                embeddings_t.T)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret , target_logit_mean.mean(), lam.mean(), cos_theta_tmp.mean()



class AdaptiveAArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35, adaptive_weighted_alpha=True):
        super(AdaptiveAArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.cosine_sim=torch.nn.CosineSimilarity()
        self.adaptive_weighted_alpha=adaptive_weighted_alpha

    def forward(self, embbedings, embeddings_t, label):
        embbedings = l2_norm(embbedings, axis=1)
        embeddings_t = l2_norm(embeddings_t, axis=1)
        # updated the classification kernel values
        with torch.no_grad():
            cos_theta_tmp = self.cosine_sim(embbedings, embeddings_t)  # get alpha value
            cos_theta_tmp = cos_theta_tmp.clamp(-1, 1)
            if self.adaptive_weighted_alpha:  # weighted alpha
                lam = self.cosine_sim(self.kernel[:, label].T, embeddings_t).clamp(1e-6, 1).view(-1, 1) # similarity between  teacher embeddings and class centers
                target_logit = cos_theta_tmp.view(-1, 1) * lam
            else:
                target_logit = cos_theta_tmp.view(-1, 1) - self.t_alpha
            target_logit_mean = target_logit.clamp(1e-6, 1.0).T
            self.kernel[:, label] = (target_logit_mean) * self.kernel[:, label] + (1. - target_logit_mean) * (
                embeddings_t.T)
        # Caculate the loss
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta , target_logit_mean.mean(), lam.mean(), cos_theta_tmp.mean()


@torch.no_grad()
def all_gather_tensor(input_tensor, dim=0):
    """ allgather tensor from all workers
    """
    world_size = dist.get_world_size()
    tensor_size = torch.tensor([input_tensor.shape[0]], dtype=torch.int64).cuda()
    tensor_size_list = [torch.ones_like(tensor_size) for _ in range(world_size)]
    dist.all_gather(tensor_list=tensor_size_list, tensor=tensor_size, async_op=False)

    max_size = torch.cat(tensor_size_list, dim=0).max()
    padded = torch.empty(max_size.item(), *input_tensor.shape[1:], dtype=input_tensor.dtype).cuda()
    padded[:input_tensor.shape[0]] = input_tensor
    padded_list = [torch.ones_like(padded) for _ in range(world_size)]
    dist.all_gather(tensor_list=padded_list, tensor=padded, async_op=False)

    slices = []
    for ts, t in zip(tensor_size_list, padded_list):
        slices.append(t[:ts.item()])
    return torch.cat(slices, dim=0)

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


