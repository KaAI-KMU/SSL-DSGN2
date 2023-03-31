import torch
import torchvision
import numpy as np
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

import cv2

class CCNN(nn.Module):    
    def __init__(self, model_cfg):
        super(CCNN, self).__init__()
        self.model_cfg = model_cfg

        kernel_size = 3
        filters = 64
        fc_filters = self.model_cfg.FC_FILTERS
        padding = kernel_size // 2
        k = self.model_cfg.TOP_K_VOLUMES.TOP_K

        self.conv = nn.Sequential(
            # nn.Conv2d(288, 124, kernel_size, stride=1, padding=padding),
            # nn.ReLU(),
            # nn.Conv2d(288, filters, kernel_size, stride=1, padding=padding),
            # nn.Conv2d(1, filters, kernel_size, stride=1, padding=padding),
            # nn.ReLU(),
            # nn.Conv2d(filters, filters, kernel_size, stride=1, padding=padding),
            # nn.ReLU(),
            # nn.Conv2d(filters, filters, kernel_size, stride=1, padding=padding),
            # nn.ReLU(),
            # nn.Conv2d(filters, filters, kernel_size, stride=1, padding=padding),
            # nn.ReLU(),
            nn.Conv2d(k, 128, kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Dropout(self.model_cfg.DP_RATIO),
            nn.Conv2d(128, filters, kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Dropout(self.model_cfg.DP_RATIO),
            nn.Conv2d(filters, 1, kernel_size, stride=1, padding=padding),
        )
        # self.fc = nn.Sequential(
        #     nn.Conv2d(288, fc_filters, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(fc_filters, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(fc_filters, 1, 1),
        # )
        self.activate = nn.Sigmoid()
    
    def gt_depth_confidence_map(self, batch_dict):   
        batch_cf_map_label = []
        for i in range(batch_dict['batch_size']):
            gt_disparity_map = batch_dict['depth_gt_img'][i].detach().cpu().numpy()
            valid_pixels_mask = gt_disparity_map > 0
            depth_preds = batch_dict['depth_preds'][i].detach().cpu().numpy()
            depth_preds = batch_dict['calib'][i].depth_to_disparity(depth_preds)
            gt_disparity_map = batch_dict['calib'][i].depth_to_disparity(gt_disparity_map)

            confidence_map_label = torch.tensor(abs(depth_preds * valid_pixels_mask - gt_disparity_map) < 3, dtype=torch.float32)
            batch_cf_map_label.append(confidence_map_label)
        
        confidence_map_label = torch.cat(batch_cf_map_label, dim=0)

        return confidence_map_label, valid_pixels_mask
    
    def top_k_volumes(self, depth_volumes):
        flatness = self.model_cfg.TOP_K_VOLUMES.FLATNESS_OF_COST
        k = self.model_cfg.TOP_K_VOLUMES.TOP_K

        rep_volumes = (-1)*depth_volumes/flatness
        volumes_norm = nn.Softmax(dim=1)
        prob_volumes = volumes_norm(rep_volumes)
        prob_volumes, volume_index = torch.sort(prob_volumes, dim=1, descending=True)
        topk_volumes = prob_volumes[:,:k,:,:]

        return topk_volumes
        
    def forward(self, batch_dict):
        batch_size = batch_dict['batch_size']
        for i in range(batch_size):
            depth_volumes = batch_dict["depth_volumes"][i]
            # out = self.conv(input)
            # out = self.fc(depth_volumes)
            # out = (out - out.min(dim=1).values) / (out.max(dim=1).values - out.min(dim=1).values)
            topk_volumes = self.top_k_volumes(depth_volumes)

            out = self.conv(topk_volumes)
            out = self.activate(out)

            batch_dict['batch_feature_depth'] = out
            confidence_map_label, valid_pixels_mask = self.gt_depth_confidence_map(batch_dict)
            batch_dict['confidence_map_label'] = confidence_map_label
            batch_dict['valid_pixels_mask'] = valid_pixels_mask

            if self.training:
                self.forward_ret_dict = batch_dict
            # else:
            #     epoch_acc = 0.0
            #     acc = calculate_accuracy(batch_dict)
            #     epoch_acc += acc.item() * depth_volumes.size(0)
            #     batch_dict["depth_confidence_module_acc"] = epoch_acc

            show = False
            if show:
                showConfidenceMap(batch_dict)
                show = False
                
        return batch_dict
    
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        dcm_loss = 0
        dcm_acc = 0
        dcm_loss_cls, cls_tb_dict = self.get_depth_confidence_loss(self.forward_ret_dict)
        dcm_loss += dcm_loss_cls
        tb_dict.update(cls_tb_dict)
        dcm_acc_cls = calculate_accuracy(self.forward_ret_dict)
        dcm_acc += dcm_acc_cls

        tb_dict['dcm_acc'] = dcm_acc.item()
        tb_dict['dcm_loss'] = dcm_loss.item()
        return dcm_loss, tb_dict

    def get_depth_confidence_loss(self, forward_ret_dict):
        confidence_map_label = forward_ret_dict['confidence_map_label'].cuda()
        depth_feature = forward_ret_dict['batch_feature_depth']
        valid_pixels_mask = torch.tensor(forward_ret_dict['valid_pixels_mask']).cuda()
        
        for i in range(forward_ret_dict['batch_size']):
            depth_feature = depth_feature[i]
            loss_func = nn.BCELoss()
            batch_loss_dcm = loss_func(
                depth_feature*valid_pixels_mask, confidence_map_label
            )
            dcm_loss_weights = self.model_cfg.LOSS_WEIGHTS
            batch_loss_dcm = batch_loss_dcm * dcm_loss_weights
            tb_dict = {'dcm_loss': batch_loss_dcm.item()}
        return batch_loss_dcm, tb_dict


def calculate_accuracy(forward_ret_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    valid_pixels_mask = torch.tensor(forward_ret_dict['valid_pixels_mask']).to(device)
    outputs  = forward_ret_dict['batch_feature_depth'].to(device)
    label = forward_ret_dict['confidence_map_label'].to(device)
    
    # Mean Squared Error
    se = (outputs * valid_pixels_mask - label) ** 2
    mse = se.sum() / (valid_pixels_mask == True).sum().item()
    acc = 1 - mse
    # mean  = (outputs * valid_pixels_mask).sum() / (valid_pixels_mask == True).sum().item()
    # var = ((outputs * valid_pixels_mask - mean) ** 2).sum() / ((valid_pixels_mask == True).sum().item() - 1)
    # acc = mse/var

    return acc


def showConfidenceMap(data_dict):
    for i in range(data_dict['batch_size']):
        frame_id = data_dict['frame_id'][i]

        left = data_dict['left_img'][i]
        left = np.transpose(left.cpu().detach().numpy(), (1,2,0))

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        left = (left * std + mean) * 255
        left = np.round(left).astype(np.uint8)
        def bgr_to_rgb(img):
            b, g, r = cv2.split(img)
            img = cv2.merge([r,g,b])
            return img
        left = bgr_to_rgb(left)

        confidence_map = np.transpose(data_dict['batch_feature_depth'][i].cpu().detach().numpy(), (1,2,0)) * 255
        confidence_map = confidence_map.astype(np.uint8)
        confidence_map = np.repeat(confidence_map,3,axis=2)

        gt_map = np.transpose(data_dict['confidence_map_label'].cpu().detach().numpy(), (1,2,0))*255
        gt_map = gt_map.astype(np.uint8)
        gt_map = np.repeat(gt_map,3,axis=2)
        
        # horizontal_concat = np.concatenate((input, confidence_map, gt_map), axis=0)
        horizontal_concat = np.concatenate((confidence_map, gt_map, left), axis=0)
        cv2.imshow(str(frame_id), horizontal_concat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()