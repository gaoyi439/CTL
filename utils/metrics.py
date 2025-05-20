import torch

def HammingLoss(pred_labels, target_labels):
    '''
    Computing Hamming loss

    Parameters
    ----------
    pred_labels : Tensor
        MxQ Tensor storing the predicted labels of the classifier, if the ith 
        instance belongs to the jth class, then pred_labels[i,j] equals to +1, 
        otherwise pred_labels[i,j] equals to 0.
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    hammingloss : float
    ''' 
    return torch.mean((pred_labels != target_labels).float()).item()

def MacroF1(pred_labels, target_labels):
    '''
    Computing macro-averaging F1

    Parameters
    ----------
    pred_labels : Tensor
        MxQ Tensor storing the predicted labels of the classifier, if the ith 
        instance belongs to the jth class, then pred_labels[i,j] equals to +1, 
        otherwise pred_labels[i,j] equals to 0.
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    macrof1 : float
    '''
    pred_pos_ind = (pred_labels == 1)
    target_pos_ind = (target_labels == 1)
    num_pred_pos = pred_pos_ind.sum(dim=0)
    num_target_pos = target_pos_ind.sum(dim=0)
    num_tp = torch.sum(pred_pos_ind & target_pos_ind, dim=0)
    
    precision = num_tp.float() / (num_pred_pos.float() + 1e-6)
    recall = num_tp.float() / (num_target_pos.float() + 1e-6)
    macrof1 = torch.mean(2 * precision * recall / (precision + recall + 1e-6)).item()
    return macrof1

def MicroF1(pred_labels, target_labels):
    '''
    Computing micro-averaging F1

    Parameters
    ----------
    pred_labels : Tensor
        MxQ Tensor storing the predicted labels of the classifier, if the ith 
        instance belongs to the jth class, then pred_labels[i,j] equals to +1, 
        otherwise pred_labels[i,j] equals to 0.
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    microf1 : float
    '''
    pred_pos_ind = (pred_labels == 1)
    target_pos_ind = (target_labels == 1)
    num_pred_pos = pred_pos_ind.sum().item()
    num_target_pos = target_pos_ind.sum().item()
    num_tp = torch.sum(pred_pos_ind & target_pos_ind).item()
    
    precision = num_tp / (num_pred_pos + 1e-6)
    recall = num_tp / (num_target_pos + 1e-6)
    microf1 = 2 * precision * recall / (precision + recall + 1e-6)
    return microf1
    
def OneError(pred_scores, target_labels):
    '''
    Computing one error

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    oneerror : float
    '''
    _, index = torch.max(pred_scores, dim=1)
    
    oneerror = 0.0
    num_data = pred_scores.size(0)
    for i in range(num_data):
        if target_labels[i, index[i]] != 1:
            oneerror += 1
            
    return oneerror / num_data

def Coverage(pred_scores, target_labels):
    '''
    Computing coverage

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    coverage : float
    '''
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    
    coverage = 0.0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        if has_label[i,:].sum() > 0:
            r = torch.max(order[i, has_label[i,:]]).item() + 1
            coverage += r
    coverage = coverage / num_data - 1.0
    return coverage / num_classes
    
def RankingLoss(pred_scores, target_labels):
    '''
    Computing ranking loss

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    rankingloss : float
    '''
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    
    rankingloss = 0.0
    count = 0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        m = torch.sum(has_label[i,:]).item()
        n = num_classes - m
        if m != 0 and n != 0:
            rankingloss = rankingloss + (torch.sum(order[i, has_label[i, :]]).item()
                                         - m*(m-1)/2.0) / (m*n)
            count += 1
            
    return rankingloss / count

def AveragePrecision(pred_scores, target_labels):
    '''
    Computing average precision

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    ap : float
    '''
    _, index = torch.sort(pred_scores, 1, descending=True)
    _, order = torch.sort(index, 1)
    has_label = target_labels == 1
    
    ap = 0.0
    count = 0
    num_data, num_classes = pred_scores.size()
    for i in range(num_data):
        m = torch.sum(has_label[i,:]).item()
        if m != 0:
            sorts, _ = torch.sort(order[i, has_label[i, :]])
            temp = 0.0
            for j in range(sorts.size(0)):
                temp += (j+1.0) / (sorts[j].item() + 1)
            ap += temp / m
            count += 1
    
    return ap / count

def MacroAUC(pred_scores, target_labels):
    '''
    Computing macro-averaging AUC

    Parameters
    ----------
    pred_scores : Tensor
        MxQ Tensor storing the predicted scores of the classifier, the scores
        of the ith instance belonging to the jth class is stored in pred_scores[i,j]
    target_labels : Tensor
        MxQ Tensor storing the real labels, if the ith instance belongs to the 
        jth class, then pred_labels[i,j] equals to +1, otherwise 
        pred_labels[i,j] equals to 0.

    Returns
    -------
    macroauc : float
    '''
    _, index = torch.sort(pred_scores, 0)
    _, order = torch.sort(index, 0)
    has_label = target_labels == 1
    
    macroauc = 0.0
    count = 0
    num_data, num_classes = pred_scores.size()
    for i in range(num_classes):
        m = torch.sum(has_label[:,i]).item()
        n = num_data - m
        if m != 0 and n != 0:
            macroauc = macroauc + (torch.sum(order[has_label[:,i], i]).item()
                                         - m*(m-1)/2.0) / (m*n)
            count += 1
            
    return macroauc / count