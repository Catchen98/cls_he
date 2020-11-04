import torch 
import torch.nn.functional as F 
from concurrent.futures import ThreadPoolExecutor, as_completed
from embedding.dataset.data import read_image

def GenerateCoordinates(h,w):
    '''generate coorinates
    Returns: [h*w, 2] FloatTensor
    '''
    # x = torch.floor(torch.arange(0, w*h) / w)
    x = torch.arange(0, w*h) / w
    y = torch.arange(0, w).repeat(h)

    coord = torch.stack([x,y], dim=1)
    return coord

def CalculateReceptiveBoxes(height,
                            width,
                            rf,
                            stride,
                            padding):

    '''
    caculate receptive boxes from original image for each feature point.
    Args:
        height: The height of feature map.
        width: The width of feature map.
        rf: The receptive field size.
        stride: The effective stride between two adjacent feature points.
        padding: The effective padding size.
    Returns:
        rf_boxes: [N, 4] recpetive boxes tensor. (N = height x weight).
        each box is represented by [ymin, xmin, ymax, xmax].
    '''
    coordinates = GenerateCoordinates(h=height,
                                      w=width)
    # create [ymin, xmin, ymax, xmax]
    point_boxes = torch.cat([coordinates, coordinates], dim=1)
    bias = torch.FloatTensor([-padding, -padding, -padding + rf - 1, -padding + rf - 1])
    rf_boxes = stride * point_boxes + bias
    return rf_boxes 

def CalculateKeypointCenters(rf_boxes):
    '''compute feature centers, from receptive field boxes (rf_boxes).
    Args:
        rf_boxes: [N, 4] FloatTensor.
    Returns:
        centers: [N, 2] FloatTensor.
    '''
    xymin = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([0,1]))
    xymax = torch.index_select(rf_boxes, dim=1, index=torch.LongTensor([2,3]))
    return (xymax + xymin) / 2.0

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Returns:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
def ApplyPcaAndWhitening(data,
                         pca_matrix,
                         pca_mean,
                         pca_vars,
                         pca_dims,
                         use_whitening=False):
    '''apply PCA/Whitening to data.
    Args: 
        data: [N, dim] FloatTensor containing data which undergoes PCA/Whitening.
        pca_matrix: [dim, dim] numpy array PCA matrix, row-major.
        pca_mean: [dim] numpy array mean to subtract before projection.
        pca_dims: # of dimenstions to use in output data, of type int.
        pca_vars: [dim] numpy array containing PCA variances. 
                   Only used if use_whitening is True.
        use_whitening: Whether whitening is to be used. usually recommended.
    Returns:
        output: [N, output_dim] FloatTensor with output of PCA/Whitening operation.
    (Warning: element 0 in pca_variances might produce nan/inf value.) 
    '''
    pca_mean = torch.from_numpy(pca_mean).float()
    pca_vars = torch.from_numpy(pca_vars).float()
    pca_matrix = torch.from_numpy(pca_matrix).float()

    data = data - pca_mean
    output = data.matmul(pca_matrix.narrow(0, 0, pca_dims).transpose(0,1))
    
    if use_whitening:
        output = output.div((pca_vars.narrow(0, 0, pca_dims) ** 0.5))
    return output


def DelfFeaturePostProcessing(
    boxes,
    descriptors,
    use_pca,
    pca_mean,
    pca_vars,
    pca_matrix,
    pca_dims,):

    ''' Delf feature post-processing.
    (1) apply L2 Normalization.
    (2) apply PCA and Whitening.
    (3) apply L2 Normalization once again.
    Args:
        descriptors: (w x h, fmap_depth) descriptor Tensor.
    Retturn:
        descriptors: (w x h, pca_dims) desciptor Tensor.
    '''

    locations = CalculateKeypointCenters(boxes)

    # L2 Normalization.
    descriptors = descriptors.squeeze()
    l2_norm = descriptors.norm(p=2, dim=1, keepdim=True)        # (1, w x h)
    descriptors = descriptors.div(l2_norm.expand_as(descriptors))  # (N, w x h)

    if use_pca:
        # apply PCA and Whitening.
        descriptors = ApplyPcaAndWhitening(
            descriptors,
            pca_matrix,
            pca_mean,
            pca_vars,
            pca_dims,
            True)
        # L2 Normalization (we found L2 Norm is not helpful. DO NOT UNCOMMENT THIS.)
        #descriptors = descriptors.view(descriptors.size(0), -1)     # (N, w x h)
        #l2_norm = descriptors.norm(p=2, dim=0, keepdim=True)        # (1, w x h)
        #descriptors = descriptors.div(l2_norm.expand_as(descriptors))  # (N, w x h)
    
    return descriptors
def GetDelfFeatureFromSingleScale(
    x,
    model,
    scale,
    rf,
    stride,
    padding,
    attn_thres,
    use_pca=False,
    pca_mean=None,
    pca_vars=None,
    pca_matrix=None,
    pca_dims=None):

    # scale image then get features and attention.
    new_h = int(round(x.size(2)*scale))
    new_w = int(round(x.size(3)*scale))
    scaled_x = F.upsample(x, size=(new_h, new_w), mode='bilinear')
    scaled_features, scaled_scores = model.forward_for_serving(scaled_x)

    # save original size attention (used for attention visualization.)
    selected_original_scale_attn = None
    if scale == 1.0:
        selected_original_scale_attn = torch.clamp(scaled_scores*255, 0, 255) # 1 1 h w
        
    # calculate receptive field boxes.
    rf_boxes = CalculateReceptiveBoxes(
        height=scaled_features.size(2),
        width=scaled_features.size(3),
        rf=rf,
        stride=stride,
        padding=padding)
    
    # re-projection back to original image space.
    rf_boxes = rf_boxes / scale
    scaled_scores = scaled_scores.view(-1)
    scaled_features = scaled_features.view(scaled_features.size(1), -1).t()

    # do post-processing for dimension reduction by PCA.
    scaled_features = DelfFeaturePostProcessing(
        rf_boxes, 
        scaled_features,
        use_pca,
        pca_mean,
        pca_vars,
        pca_matrix,
        pca_dims)

    # use attention score to select feature.
    indices = None
    while(indices is None or len(indices) == 0):
        indices = torch.gt(scaled_scores, attn_thres).nonzero().squeeze()
        attn_thres = attn_thres * 0.5   # use lower threshold if no indexes are found.
        if attn_thres < 0.001:
            break;
   
    try:
        selected_boxes = torch.index_select(rf_boxes, dim=0, index=indices)
        selected_features = torch.index_select(scaled_features, dim=0, index=indices)
        selected_scores = torch.index_select(scaled_scores, dim=0, index=indices)
        selected_scales = torch.ones_like(selected_scores) * scale
    except Exception as e:
        selected_boxes = None
        selected_features = None
        selected_scores = None
        selected_scales = None
        print(e)
        pass;
        
    return selected_boxes, selected_features, selected_scales, selected_scores, selected_original_scale_attn

def GetDelfFeatureFromMultiScale(
    x,
    model,
    filename,
    rf,
    stride,
    padding,
    top_k,
    scale_list,
    iou_thres,
    attn_thres,
    use_pca=False,
    pca_mean=None,
    pca_vars=None,
    pca_matrix=None,
    pca_dims=None,
    workers=8):
    '''GetDelfFeatureFromMultiScale
    warning: use workers = 1 for serving otherwise out of memory error could occurs.
    (because uwsgi uses multi-threads by itself.)
    '''

    # helper func.
    def __concat_tensors_in_list__(tensor_list, dim):
        res = None
        tensor_list = [x for x in tensor_list if x is not None]
        for tensor in tensor_list:
            if res is None:
                res = tensor
            else:
                res = torch.cat((res, tensor), dim=dim)
        return res

    # extract features for each scale, and concat.
    output_boxes = []
    output_features = []
    output_scores = []
    output_scales = []
    output_original_scale_attn = None

    # multi-threaded feature extraction from different scales.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # assign jobs.
        futures = {
            pool.submit(
                GetDelfFeatureFromSingleScale,
                    x,
                    model,
                    scale,
                    rf,
                    stride,
                    padding,
                    attn_thres,
                    use_pca,
                    pca_mean,
                    pca_vars,
                    pca_matrix,
                    pca_dims):
            scale for scale in scale_list
        }
        for future in as_completed(futures):
            (selected_boxes, selected_features, 
            selected_scales, selected_scores, 
            selected_original_scale_attn) = future.result()
            # append to list.  
            output_boxes.append(selected_boxes) if selected_boxes is not None else output_boxes
            output_features.append(selected_features) if selected_features is not None else output_features
            output_scales.append(selected_scales) if selected_scales is not None else output_scales
            output_scores.append(selected_scores) if selected_scores is not None else output_scores
            if selected_original_scale_attn is not None:
                output_original_scale_attn = selected_original_scale_attn

    # if scale == 1.0 is not included in scale list, just show noisy attention image.
    if output_original_scale_attn is None:
        output_original_scale_attn = x.clone().uniform()

    # concat tensors precessed from different scales.
    output_boxes = __concat_tensors_in_list__(output_boxes, dim=0)
    output_features = __concat_tensors_in_list__(output_features, dim=0)
    output_scales = __concat_tensors_in_list__(output_scales, dim=0)
    output_scores = __concat_tensors_in_list__(output_scores, dim=0)

    # perform Non Max Suppression(NMS) to select top-k bboxes arrcoding to the attn_score.
    keep_indices, count = nms(boxes = output_boxes,
                              scores = output_scores,
                              overlap = iou_thres,
                              top_k = top_k)
    keep_indices = keep_indices[:top_k]
    output_boxes = torch.index_select(output_boxes, dim=0, index=keep_indices)
    output_features = torch.index_select(output_features, dim=0, index=keep_indices)
    output_scales = torch.index_select(output_scales, dim=0, index=keep_indices)
    output_scores = torch.index_select(output_scores, dim=0, index=keep_indices)
    output_locations = CalculateKeypointCenters(output_boxes)
    
    data = {
        'filename':filename,
        'location_np_list':output_locations.cpu().numpy(), # y, x
        'descriptor_np_list':output_features.cpu().numpy(),
        'feature_scale_np_list':output_scales.cpu().numpy(),
        'attention_score_np_list':output_scores.cpu().numpy(),
        'attention_np_list':output_original_scale_attn.cpu().numpy()
    }
    
    # free GPU memory.
    # del output_locations
    # del output_boxes, selected_boxes
    # del output_features, selected_features
    # del output_scales, selected_scales
    # del output_scores, selected_scores
    # del output_original_scale_attn, selected_original_scale_attn 
    #torch.cuda.empty_cache()            # it releases all unoccupied cached memory!! (but it makes process slow)

    return data

def GetDelfFeatureFromSingleFile(
    img,
    model,
    transform,
    rf,
    stride,
    padding,
    attn_thres,
    use_pca=False,
    pca_mean=None,
    pca_vars=None,
    pca_matrix=None,
    pca_dims=None):
    x = transform(read_image(img)).unsqueeze(0).cuda()
    return img, GetDelfFeatureFromSingleScale(
        x,
        model,
        1.0,
        rf,
        stride,
        padding,
        attn_thres,
        use_pca,
        pca_mean,
        pca_vars,
        pca_matrix,
        pca_dims)

def GetDelfFeatureFromMultiFile(
    imgs,
    model,
    transform,
    filenames,
    rf,
    stride,
    padding,
    top_k,
    iou_thres,
    attn_thres,
    use_pca=False,
    pca_mean=None,
    pca_vars=None,
    pca_matrix=None,
    pca_dims=None,
    workers=6):
    '''GetDelfFeatureFromMultiFile
    warning: use workers = 1 for serving otherwise out of memory error could occurs.
    (because uwsgi uses multi-threads by itself.)
    '''

    # helper func.
    def __concat_tensors_in_list__(tensor_list, dim):
        res = None
        tensor_list = [x for x in tensor_list if x is not None]
        for tensor in tensor_list:
            if res is None:
                res = tensor
            else:
                res = torch.cat((res, tensor), dim=dim)
        return res
    results = {}
    # multi-threaded feature extraction from different scales.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # assign jobs.
        futures = {
            pool.submit(
                GetDelfFeatureFromSingleFile,
                    img,
                    model,
                    transform,
                    rf,
                    stride,
                    padding,
                    attn_thres,
                    use_pca,
                    pca_mean,
                    pca_vars,
                    pca_matrix,
                    pca_dims):
            img for img in imgs
        }
        for fidx, future in enumerate(as_completed(futures)):
            # extract features for each scale, and concat.
            output_boxes = []
            output_features = []
            output_scores = []
            output_scales = []
            output_original_scale_attn = None
            
            img, (selected_boxes, selected_features, 
            selected_scales, selected_scores, 
            selected_original_scale_attn) = future.result()
            # append to list.  
            output_boxes.append(selected_boxes) if selected_boxes is not None else output_boxes
            output_features.append(selected_features) if selected_features is not None else output_features
            output_scales.append(selected_scales) if selected_scales is not None else output_scales
            output_scores.append(selected_scores) if selected_scores is not None else output_scores
            # if selected_original_scale_attn is not None:
            #     output_original_scale_attn = selected_original_scale_attn

            # if scale == 1.0 is not included in scale list, just show noisy attention image.
            # if output_original_scale_attn is None:
            #     output_original_scale_attn = imgs[fidx].clone().uniform()

            # concat tensors precessed from different scales.
            output_boxes = __concat_tensors_in_list__(output_boxes, dim=0)
            output_features = __concat_tensors_in_list__(output_features, dim=0)
            output_scales = __concat_tensors_in_list__(output_scales, dim=0)
            output_scores = __concat_tensors_in_list__(output_scores, dim=0)

            # perform Non Max Suppression(NMS) to select top-k bboxes arrcoding to the attn_score.
            keep_indices, count = nms(boxes = output_boxes,
                                    scores = output_scores,
                                    overlap = iou_thres,
                                    top_k = top_k)
            keep_indices = keep_indices[:top_k]
            output_boxes = torch.index_select(output_boxes, dim=0, index=keep_indices)
            output_features = torch.index_select(output_features, dim=0, index=keep_indices)
            output_scales = torch.index_select(output_scales, dim=0, index=keep_indices)
            output_scores = torch.index_select(output_scores, dim=0, index=keep_indices)
            output_locations = CalculateKeypointCenters(output_boxes)
            
            data = {
                'filename': img,
                'location_np_list':output_locations.cpu().numpy(), # y, x
                'descriptor_np_list':output_features.cpu().numpy(),
                'feature_scale_np_list':output_scales.cpu().numpy(),
                'attention_score_np_list':output_scores.cpu().numpy(),
                # 'attention_np_list':output_original_scale_attn.cpu().numpy()
            }
            results[img] = data
    return results
if __name__ == "__main__":
    rf_boxes = CalculateReceptiveBoxes(7,7,291,16,145.0) # layer3
    rf_boxes = CalculateReceptiveBoxes(7,7,483.0,16,241.0) # layer4 
    # rf_boxes = CalculateReceptiveBoxes(7,7,211,16,105.0) # layer4 
    # rf_boxes = CalculateReceptiveBoxes(7,7,267,16,133) # layer4 
    kepoints = CalculateKeypointCenters(rf_boxes)
    import pdb; pdb.set_trace()