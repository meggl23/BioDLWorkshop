import torch
import torch.nn.functional as F
import numpy as np
import warnings


def similarity(vec1, vec2):
    if torch.linalg.norm(vec1) < 1e-8 or torch.linalg.norm(vec2) < 1e-8:
        warnings.warn("Small vector in similarity!")
    #     return torch.tensor(1.0, device=vec1.device)
    # else:
    vec1 = vec1.type(torch.float64)
    vec2 = vec2.type(torch.float64)
    # return F.cosine_similarity(vec1 / torch.linalg.norm(vec1), vec2 / torch.linalg.norm(vec2), dim=0).clip(-1.0, 1.0)
    return F.cosine_similarity(vec1 / torch.linalg.norm(vec1), vec2 / torch.linalg.norm(vec2), dim=0, eps=1e-20).clip(-1.0, 1.0)
    # return F.cosine_similarity(vec1, vec2, dim=0).clip(-1.0, 1.0)
    #return F.cosine_similarity(vec1.type(torch.float64), vec2.type(torch.float64), dim=0).clip(-1.0, 1.0)


def generate_pd_matrix(dim, device, dtype):
    from sklearn.datasets import make_spd_matrix
    m = torch.tensor(make_spd_matrix(dim)).to(device).type(dtype)
    return m


def generate_positive_full_rank_matrix(dim, device, dtype):
    import math
    m = torch.rand((dim,dim)).to(device).to(dtype)
    m /= torch.sum(m, dim=1).reshape(-1, 1)
    return m


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


# def topk_correct(output, target, topk=(1,)):
#     # computes the accuracy over the k top predictions for the specified values of k
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].sum().item()
#             res.append(correct_k)
#         return res

def topk_correct(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_correct = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1)  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.sum(dim=0, keepdim=True).item()  # [kB] -> [1]
            list_topk_correct.append(tot_correct_topk)
        return list_topk_correct  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


if __name__ == "__main__":
    output = torch.rand(size=(32, 10))
    target = torch.Tensor([4] * 32)
    topk = (1,5)

    top1_correct, top5_correct = topk_correct(output, target, topk)
    print(top1_correct, top5_correct)