#!/usr/bin/env python3
'''
script including main functions
'''

import cv2
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import scipy.stats as stats
from skimage.measure import block_reduce
from calibrationframework.netcal.metrics import ECE

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
np.random.seed(0)

from global_defs import CONFIG
from prepare_data import cs_labels

trainId2label = { label.trainId : label for label in reversed(cs_labels) }
Id2label = { label.Id : label for label in reversed(cs_labels) }

def load_sem_seg(prob_path):
    prob = np.load(prob_path)
    prob = np.exp(prob)/np.sum(np.exp(prob),0) 
    return prob


def load_pp(prob_path):
    prob = np.load(prob_path)
    prob = prob / (1-prob) # undo sigmoid -> exp(logit)
    return prob


def vis_pred_i(item):

    classes = np.concatenate( (np.arange(0,19),[255]), axis=0)

    image = item[0]
    label = item[1]
    prob_sem_seg = load_sem_seg(item[4])
    seg = np.argmax(prob_sem_seg, axis=0)
    prob_pp = load_pp(item[5])

    I1 = image.copy()
    I2 = image.copy()
    I3 = image.copy()
    I4 = image.copy()

    for c in classes:
        I1[label==c,:] = np.asarray(trainId2label[c].color)*0.6 + I1[label==c,:]*0.4
        I2[seg==c,:] = np.asarray(trainId2label[c].color)*0.6 + I2[seg==c,:]*0.4
    
    plt.imsave(CONFIG.VIS_PRED_DIR + item[3] + '_tmp1.png', 1-prob_sem_seg[0], cmap='inferno')
    I3_heat = np.asarray( Image.open(CONFIG.VIS_PRED_DIR + item[3] + '_tmp1.png').convert('RGB') )
    os.remove(CONFIG.VIS_PRED_DIR + item[3] + '_tmp1.png')

    prob_pp_scaled = cv2.GaussianBlur(np.power(prob_pp,0.5), ksize=(25, 25), sigmaX=5)
    plt.imsave(CONFIG.VIS_PRED_DIR + item[3] + '_tmp2.png', prob_pp_scaled, cmap='inferno')
    I4_heat = np.asarray( Image.open(CONFIG.VIS_PRED_DIR + item[3] + '_tmp2.png').convert('RGB') )
    os.remove(CONFIG.VIS_PRED_DIR + item[3] + '_tmp2.png')

    I3 = I3_heat * 0.6 + I3 * 0.4
    I4 = I4_heat * 0.6 + I4 * 0.4

    img12   = np.concatenate( (I1,I2), axis=1 )
    img34  = np.concatenate( (I3,I4), axis=1 )
    img   = np.concatenate( (img12,img34), axis=0 )

    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = img.resize((int(item[0].shape[1]),int(item[0].shape[0])))
    img.save(CONFIG.VIS_PRED_DIR + item[3] + '.png')
    plt.close()
    print('stored:', item[3]+'.png')


def compute_calib(loader, save_path, box_size, spl_per_img=50):

    if not os.path.isfile(save_path+'p_pp.npy'):

        # 1: free/drivable
        free_sem_seg = []
        free_pp = []

        # probability that the box is free/drivable
        p_sem_seg = []
        p_pp = []

        for item in loader:
            print(item[3])

            # gt_sem_seg: semantic segmentation gt - 0:street 
            # gt_pp: gt of point process - 0:object center
            gt_sem_seg = item[1]
            gt_pp = np.ones((gt_sem_seg.shape[0],gt_sem_seg.shape[1]))

            gt_inst = np.asarray(Image.open(item[2])).copy()
            gt_inst[gt_inst<=33] = 0
            for c in np.unique(gt_inst)[1:]:
                indices = np.where(gt_inst == c)
                id_x = np.min(indices[0])+(np.max(indices[0])-np.min(indices[0]))/2
                id_y = np.min(indices[1])+(np.max(indices[1])-np.min(indices[1]))/2
                gt_pp[int(id_x),int(id_y)] = 0

            prob_sem_seg = load_sem_seg(item[4])[0]
            prob_pp = load_pp(item[5])

            counter = 0
            while counter < spl_per_img:
                w = np.random.randint(10, int(box_size/10))
                h = int(box_size/w)
                x = np.random.randint(0, 2048-w)
                y = np.random.randint(0, 1024-h)

                box_gt_sem_seg = gt_sem_seg[y:y+h, x:x+w]
                box_gt_pp = gt_pp[y:y+h, x:x+w]
                box_sem_seg = prob_sem_seg[y:y+h, x:x+w]
                box_pp = prob_pp[y:y+h, x:x+w]

                free_sem_seg.append( np.sum(box_gt_sem_seg!=0)==0 ) # no other pixels than street
                free_pp.append( np.sum(box_gt_pp==0)==0 ) # no object center
                p_sem_seg.append( np.prod(box_sem_seg) ) 
                p_pp.append( np.exp(-1.0*np.sum(box_pp) / (512*1024)) ) 

                counter += 1           
        
        free_sem_seg = np.array(free_sem_seg)
        free_pp = np.array(free_pp)
        p_sem_seg = np.array(p_sem_seg)
        p_pp = np.array(p_pp)
        
        np.save(save_path+'free_sem_seg.npy', free_sem_seg)
        np.save(save_path+'free_pp.npy', free_pp)
        np.save(save_path+'p_sem_seg.npy', p_sem_seg)
        np.save(save_path+'p_pp.npy', p_pp)
    else:
        free_sem_seg = np.load(save_path+'free_sem_seg.npy')
        free_pp = np.load(save_path+'free_pp.npy')
        p_sem_seg = np.load(save_path+'p_sem_seg.npy')
        p_pp = np.load(save_path+'p_pp.npy')

    ## confidence uniform distributed 0-1 
    num_bins = 10
    num_bin_free_sem_seg = []
    num_bin_sem_seg = []
    avg_conf_sem_seg = []
    num_bin_free_pp = []
    num_bin_pp = []
    avg_conf_pp = []

    for l in np.arange(num_bins):
        lb = 0.1*l
        ub = 0.1*(l+1) if l < (num_bins-1) else 1.1

        num_bin_free_sem_seg.append( np.sum(free_sem_seg[np.logical_and(p_sem_seg>=lb, p_sem_seg<ub)]) )
        num_bin_sem_seg.append( np.sum(np.logical_and(p_sem_seg>=lb, p_sem_seg<ub)) )
        avg_conf_sem_seg.append( np.mean(p_sem_seg[np.logical_and(p_sem_seg>=lb, p_sem_seg<ub)]) )

        num_bin_free_pp.append( np.sum(free_pp[np.logical_and(p_pp>=lb, p_pp<ub)]) )
        num_bin_pp.append( np.sum(np.logical_and(p_pp>=lb, p_pp<ub)) )
        avg_conf_pp.append( np.mean(p_pp[np.logical_and(p_pp>=lb, p_pp<ub)]) )

    num_bin_free_sem_seg = np.array(num_bin_free_sem_seg)
    num_bin_sem_seg = np.array(num_bin_sem_seg)
    num_bin_free_pp = np.array(num_bin_free_pp)
    num_bin_pp = np.array(num_bin_pp)

    ece = ECE(num_bins)
    ece_sem_seg = ece.measure(p_sem_seg, free_sem_seg)
    ece_pp = ece.measure(p_pp, free_pp)

    size_font = 16
    plt.hist(p_sem_seg, bins=np.linspace(0., 1., 10), alpha=0.3, color='tab:blue', label='sem seg')
    plt.hist(p_pp, bins=np.linspace(0., 1., 10), alpha=0.3, color='tab:pink', label='pp')
    plt.xlabel("confidence")
    plt.legend()
    plt.savefig(save_path + 'histo.png', dpi=400, bbox_inches='tight')
    plt.close()

    plt.plot([0,1], [0,1], c="gray")
    plt.scatter(0.1*np.arange(10)+0.05, num_bin_free_sem_seg/num_bin_sem_seg, c="royalblue", s=80, label='ECE$_S$ = '+str(np.round(ece_sem_seg,4)))
    plt.scatter(0.1*np.arange(10)+0.05, num_bin_free_pp/num_bin_pp, c="darkorange", s=80, label='ECE$_P$ = '+str(np.round(ece_pp,4)))
    plt.grid(True)
    plt.xlabel("confidence", fontsize=size_font)
    plt.xticks(0.1*np.arange(11), fontsize=size_font)
    plt.ylabel("accuracy", fontsize=size_font)
    plt.yticks(fontsize=size_font)
    plt.legend(prop={'size': 15})
    plt.savefig(save_path + 'calibration.png', dpi=400, bbox_inches='tight')
    plt.close()


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2-x1+1)*max(0, y2-y1+1)
    union = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1) + (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1) - intersection
    return intersection / union if union > 0 else 0


def calculate_ap(gts, preds, scores, id_gt, id_pred, iou_threshold): 

    if gts.shape[0] == 0 or preds.shape[0] == 0:
        return 0, 0, preds.shape[0], gts.shape[0]

    # sort predictions by score from high to low
    indices = np.argsort(scores)[::-1]
    preds = preds[indices]
    scores = scores[indices]
    id_pred = id_pred[indices]

    gt_used = [False] * len(gts)
    tp = []  
    fp = []  

    for p in range(preds.shape[0]):
        best_iou = 0
        best_gt_idx = -1
        for g in range(gts.shape[0]):
            if not gt_used[g]:
                current_iou = calculate_iou(preds[p,:], gts[g,:])
                if current_iou > best_iou and id_gt[g] == id_pred[p]:
                    best_iou = current_iou
                    best_gt_idx = g

        if best_iou >= iou_threshold:
            tp.append(1)  
            fp.append(0)
            gt_used[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    fn = gts.shape[0] - tp
    precisions = tp / (tp + fp + 1e-8)
    recalls = tp / (tp + fn + 1e-8)

    recalls = np.concatenate(([0], recalls, [1]))
    precisions = np.concatenate(([0], precisions, [0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap, tp.max(), fp.max(), fn.min()


def calculate_map(gt, preds, scores, class_gt, class_pred, id_gt, id_pred, iou_threshold=0.5):

    aps = []
    tps = 0
    fps = 0
    fns = 0
    for c in [24,26]:
        ap, tp, fp, fn = calculate_ap(gt[class_gt==c], preds[class_pred==c], scores[class_pred==c], id_gt[class_gt==c], id_pred[class_pred==c], iou_threshold)
        aps.append(ap)
        tps += tp
        fps += fp
        fns += fn

        with open(os.path.join( CONFIG.BOX_DIR, 'results.txt'), 'a') as fi:
            print('class: ', c, file=fi)
            print('ap: ', ap, file=fi)
            print('TP: ', tp, file=fi)
            print('FP: ', fp, file=fi)
            print('FN: ', fn, file=fi)
            print(' ', file=fi)
    map = np.mean(aps)
    with open(os.path.join( CONFIG.BOX_DIR, 'results.txt'), 'a') as fi:
        print('map: ', map, file=fi)
        print('TPs: ', tps, file=fi)
        print('FPs: ', fps, file=fi)
        print('FNs: ', fns, file=fi)
        print('IoU th: ', iou_threshold, file=fi)


def detect_boxes(loader, save_path, box_size, spl_per_img=50, flag_plot=False):

    # compute AV
    if not os.path.isfile(CONFIG.BOX_DIR+'npy_preds.npy'):
        print('compute AV')

        bs = 10 # rectangle size: (2*bs)*(2*bs)
        gt = []
        preds = []
        scores = []
        class_gt = []
        class_pred = []
        id_gt = []
        id_pred = []

        for item,i in zip(loader,range(len(loader))):
            print(item[3])

            image = item[0]
            if flag_plot:
                I1 = Image.fromarray(image.astype('uint8'), 'RGB')
                I2 = Image.fromarray(image.astype('uint8'), 'RGB')
                draw1 = ImageDraw.Draw(I1)
                draw2 = ImageDraw.Draw(I2)
            
            # load ground truth bounding boxes
            gt_inst = np.asarray(Image.open(item[2])).copy()
            gt_inst[gt_inst<=33] = 0
            bboxes_gt = []
            cl_gt = []
            for c in np.unique(gt_inst)[1:]:
                indices = np.where(gt_inst == c)
                class_i = int(str(c)[:2])
                bboxes_gt.append(np.asarray([np.min(indices[1]),np.min(indices[0]),np.max(indices[1]),np.max(indices[0]) ]) )
                if class_i == 25:
                    class_i = 24
                elif class_i == 27 or class_i == 28 or class_i == 31 or class_i == 32 or class_i == 33:
                    class_i = 26
                cl_gt.append(class_i)

                if flag_plot:
                    draw1.rectangle([int(bboxes_gt[-1][0]), int(bboxes_gt[-1][1]), int(bboxes_gt[-1][2]), int(bboxes_gt[-1][3])], outline=Id2label[cl_gt[-1]].color,width=5)
            bboxes_gt = np.array(bboxes_gt)
            cl_gt = np.array(cl_gt)

            # load bounding box predictions
            # dict_keys(['labels', 'scores', 'bboxes', 'lam', 'wh_map', 'class_map'])
            with open(item[6]) as f:
                dictionary = json.load(f)
            prob_pp = load_pp(item[5])
            wh_map = np.array(dictionary['wh_map']) 
            class_map = np.array(dictionary['class_map']) 
            factor = 2

            num_inst = int(np.round(np.mean(prob_pp)*4))
            coords_pp = []
            scores_pp = []
            counter = 0
            while counter < num_inst:
                c_pp = np.unravel_index(np.argmax(prob_pp, axis=None), prob_pp.shape)
                sum_pp_i = np.sum(prob_pp[np.max((c_pp[0]-bs,0)):np.min((c_pp[0]+bs,1024)), np.max((c_pp[1]-bs,0)):np.min((c_pp[1]+bs,2048))]) / (512*1024)
                prob_pp[np.max((c_pp[0]-bs,0)):np.min((c_pp[0]+bs,1024)), np.max((c_pp[1]-bs,0)):np.min((c_pp[1]+bs,2048))] = 0

                coords_pp.append(c_pp)
                scores_pp.append( sum_pp_i )
                counter += 1
            coords_pp = np.array(coords_pp)
            scores_pp = np.array(scores_pp)
            
            bboxes_pp = np.zeros((scores_pp.shape[0],4))
            cl_pp = np.zeros((scores_pp.shape[0]))
            for c in range(scores_pp.shape[0]):
                id_x = int(coords_pp[c,0]/factor)
                id_y = int(coords_pp[c,1]/factor)
                val_w= np.round(wh_map[0,0,id_x,id_y]*factor)
                val_h= np.round(wh_map[0,1,id_x,id_y]*factor)
                bboxes_pp[c,0] = max(0,coords_pp[c,1]-int(val_w/2))
                bboxes_pp[c,2] = min(2048,coords_pp[c,1]+int(val_w/2))
                bboxes_pp[c,1] = max(0,coords_pp[c,0]-int(val_h/2))
                bboxes_pp[c,3] = min(1024,coords_pp[c,0]+int(val_h/2))
                class_box = np.argmax(class_map[0,:,np.max((id_x-5,0)):np.min((id_x+5,int(1024/factor))), np.max((id_y-5,0)):np.min((id_y+5,int(2048/factor)))],0)
                cl_pp[c] = stats.mode(class_box, axis=None, keepdims=False).mode + 24 
                if flag_plot:
                    draw2.rectangle([int(bboxes_pp[c][0]), int(bboxes_pp[c][1]), int(bboxes_pp[c][2]), int(bboxes_pp[c][3])], outline=Id2label[cl_pp[c]].color,width=5)

            gt.extend(bboxes_gt)
            preds.extend(bboxes_pp)
            scores.extend(scores_pp)
            class_gt.extend(cl_gt)
            class_pred.extend(cl_pp)
            id_gt.extend(np.ones((bboxes_gt.shape[0]))*i)
            id_pred.extend(np.ones((bboxes_pp.shape[0]))*i)

            if flag_plot:
                img = np.concatenate( (I1,I2), axis=1 )
                img = Image.fromarray(img.astype('uint8'), 'RGB')
                img = img.resize((int(item[0].shape[1]),int(item[0].shape[0]/2)))
                img.save(CONFIG.BOX_DIR + item[3] + '_' + str(len(bboxes_gt)) + '_' + str(len(bboxes_pp)) + '.png')
        
        gt = np.array(gt)
        preds = np.array(preds)
        scores = np.array(scores)
        class_gt = np.array(class_gt)
        class_pred = np.array(class_pred)
        id_gt = np.array(id_gt)
        id_pred = np.array(id_pred)
        np.save(CONFIG.BOX_DIR+'npy_gt.npy', gt)
        np.save(CONFIG.BOX_DIR+'npy_preds.npy', preds)
        np.save(CONFIG.BOX_DIR+'npy_scores.npy', scores)
        np.save(CONFIG.BOX_DIR+'npy_class_gt.npy', class_gt)
        np.save(CONFIG.BOX_DIR+'npy_class_pred.npy', class_pred)
        np.save(CONFIG.BOX_DIR+'npy_id_gt.npy', id_gt)
        np.save(CONFIG.BOX_DIR+'npy_id_pred.npy', id_pred)
    else:
        gt = np.load(CONFIG.BOX_DIR+'npy_gt.npy')
        preds = np.load(CONFIG.BOX_DIR+'npy_preds.npy')
        scores = np.load(CONFIG.BOX_DIR+'npy_scores.npy')
        class_gt = np.load(CONFIG.BOX_DIR+'npy_class_gt.npy')
        class_pred = np.load(CONFIG.BOX_DIR+'npy_class_pred.npy')
        id_gt = np.load(CONFIG.BOX_DIR+'npy_id_gt.npy')
        id_pred = np.load(CONFIG.BOX_DIR+'npy_id_pred.npy')

    calculate_map(gt,preds,scores,class_gt,class_pred,id_gt,id_pred)

    # compute calibration
    if not os.path.isfile(save_path+'npy_p_pp.npy'):
        print('compute calibration')

        free_pp = [] # 1: free/drivable
        p_pp = [] # probability that the box is free/drivable

        for item,i in zip(loader,range(len(loader))):
            print(item[3])
            
            # load ground truth bounding boxes
            gt_inst = np.asarray(Image.open(item[2])).copy()
            gt_inst[gt_inst<=33] = 0
            bboxes_gt = []
            for c in np.unique(gt_inst)[1:]:
                indices = np.where(gt_inst == c)
                class_i = int(str(c)[:2])
                bboxes_gt.append(np.asarray([np.min(indices[1]),np.min(indices[0]),np.max(indices[1]),np.max(indices[0]) ]) )
            bboxes_gt = np.array(bboxes_gt)

            # load bounding box predictions
            # dict_keys(['labels', 'scores', 'bboxes', 'lam', 'wh_map', 'class_map'])
            with open(item[6]) as f:
                dictionary = json.load(f)
            prob_pp = load_pp(item[5])
            wh_map = np.array(dictionary['wh_map'])
            class_map = np.array(dictionary['class_map']) 
            factor = 2

            bboxes_gt_small = np.round(bboxes_gt / factor)
            prob_pp_small = block_reduce(load_pp(item[5]), block_size=(factor, factor), func=np.sum)
            w_map = wh_map[0,0,:]
            h_map = wh_map[0,1,:]
            coords = np.stack((np.indices((wh_map.shape[2], wh_map.shape[3]))), axis=-1) # shape (256,512,2)

            counter = 0
            while counter < spl_per_img:
                w = np.random.randint(10, int(box_size/10))
                h = int(box_size/w)
                w = int(np.round(w/factor))
                h = int(np.round(h/factor))
                x = np.random.randint(0, wh_map.shape[3]-w) 
                y = np.random.randint(0, wh_map.shape[2]-h) 
                cp_sample = [int(x+w/2),int(y+h/2)]

                flag_free = 1
                for g in range(bboxes_gt_small.shape[0]):
                    iou = calculate_iou(bboxes_gt_small[g],[x,y,x+w,y+h])
                    if iou > 0:
                        flag_free = 0
                        break
                free_pp.append( flag_free ) # no overlap between gt box and sampled box

                if CONFIG.MODEL_NAME == 'DeepLab_rs50':
                    sigma_scale = 3.31437977213126
                elif CONFIG.MODEL_NAME == 'HRnet':
                    sigma_scale = 3.54501126808279

                cdf_x = 2* np.abs(cp_sample[0]-coords[:,:,1])-w_map 
                cdf_w = stats.laplace.cdf(cdf_x, loc=w_map, scale=sigma_scale)
                cdf_w = 1-cdf_w
                cdf_y = 2* np.abs(cp_sample[1]-coords[:,:,0])-h_map
                cdf_h = stats.laplace.cdf(cdf_y, loc=h_map, scale=sigma_scale)
                cdf_h = 1-cdf_h

                if flag_plot and counter == 0:
                    prob_scaled = np.power(prob_pp_small,0.5)* np.power(cdf_w*cdf_h ,0.0001)
                    prob_scaled = 0.7*prob_scaled + 0.3*np.power(prob_pp_small,0.5)
                    prob_scaled = cv2.GaussianBlur(prob_scaled, ksize=(25, 25), sigmaX=5)
                    plt.imsave(save_path + 'heatmaps/' + item[3] + '_hm_pred_tmp.png', prob_scaled, cmap='inferno')
                    img_heat = np.asarray( Image.open(save_path + 'heatmaps/' + item[3] + '_hm_pred_tmp.png').convert('RGB') )
                    os.remove(save_path + 'heatmaps/' + item[3] + '_hm_pred_tmp.png')
                    img_rgb = Image.fromarray(item[0].astype('uint8'), 'RGB')
                    img_rgb = np.asarray( img_rgb.resize((int(img_heat.shape[1]),int(img_heat.shape[0]))) )
                    img = img_heat * 0.6 + img_rgb * 0.4
                    img = Image.fromarray(img.astype('uint8'), 'RGB')
                    img.save(save_path + 'heatmaps/' + item[3] + '_hm_pred.png')

                cdf_w[y:y+h, x:x+w] = 0
                cdf_h[y:y+h, x:x+w] = 0
                p_pp.append( np.exp(-1.0*np.sum(prob_pp_small*cdf_w*cdf_h) / (512*1024)) ) 

                counter += 1
        
        free_pp = np.array(free_pp)
        p_pp = np.array(p_pp)  
        np.save(save_path+'npy_free_pp.npy', free_pp)
        np.save(save_path+'npy_p_pp.npy', p_pp)
    else:
        free_pp = np.load(save_path+'npy_free_pp.npy')
        p_pp = np.load(save_path+'npy_p_pp.npy')

    num_bins = 10 # confidence uniform distributed 0-1 
    num_bin_free_pp = []
    num_bin_pp = []
    avg_conf_pp = []

    for l in np.arange(num_bins):
        lb = 0.1*l
        ub = 0.1*(l+1) if l < (num_bins-1) else 1.1

        num_bin_free_pp.append( np.sum(free_pp[np.logical_and(p_pp>=lb, p_pp<ub)]) )
        num_bin_pp.append( np.sum(np.logical_and(p_pp>=lb, p_pp<ub)) )
        avg_conf_pp.append( np.mean(p_pp[np.logical_and(p_pp>=lb, p_pp<ub)]) )
    
    num_bin_free_pp = np.array(num_bin_free_pp)
    num_bin_pp = np.array(num_bin_pp)

    ece = ECE(num_bins)
    ece_pp = ece.measure(p_pp, free_pp)

    size_font = 16
    plt.hist(p_pp, bins=np.linspace(0., 1., 10), alpha=0.3, color='tab:pink', label='pp')
    plt.xlabel("confidence")
    plt.legend()
    plt.savefig(save_path + 'img_histo.png', dpi=400, bbox_inches='tight')
    plt.close()

    plt.plot([0,1], [0,1], c="gray")
    plt.scatter(0.1*np.arange(10)+0.05, num_bin_free_pp/num_bin_pp, c="mediumseagreen", s=80, label='ECE$_{BB}$ = '+str(np.round(ece_pp,4)))
    plt.grid(True)
    plt.xlabel("confidence", fontsize=size_font)
    plt.xticks(0.1*np.arange(11), fontsize=size_font)
    plt.ylabel("accuracy", fontsize=size_font)
    plt.yticks(fontsize=size_font)
    plt.legend(prop={'size': 15})
    plt.savefig(save_path + 'img_calibration.png', dpi=400, bbox_inches='tight')
    plt.close()

        
        


