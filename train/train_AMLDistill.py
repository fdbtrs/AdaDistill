import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

from backbones.mobilefacenet import MobileFaceNet
from utils import losses
from config.config import config as cfg
from utils.dataset import MXFaceDataset, DataLoaderX,FaceDatasetFolder
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100, iresnet50, iresnet18

torch.backends.cudnn.benchmark = True

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)
    if ( cfg.db_file_format !="rec"):
        trainset = FaceDatasetFolder(root_dir=cfg.data_path, local_rank=local_rank,number_sample=cfg.sample)
    else:
        trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=16, pin_memory=True, drop_last=True)

    # create model
    if cfg.teacher == "iresnet100":
        backbone_t = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.teacher == "iresnet50":
        backbone_t = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank) #iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.teacher == "iresnet18":
        backbone_t = iresnet18(num_features=cfg.embedding_size, use_se=cfg.SE).to(
            local_rank)
    if cfg.network== "mobilefacenet":
        backbone = MobileFaceNet(input_size=(112,112)).to(local_rank)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet18":
        backbone = iresnet18(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    else:
        backbone = None
        logging.info("create backbone failed!")
        exit()
    try:
        backbone_pth = os.path.join(cfg.pretrained_teacher_path)
        backbone_t.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
        if rank == 0:
            logging.info("teacher loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("teacher init, failed!")


    if cfg.global_step:
        try:
            backbone_pth = os.path.join(cfg.output, str(cfg.global_step) + "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load backbone resume init, failed!")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)
    backbone_t = DistributedDataParallel(
        module=backbone_t, broadcast_buffers=False, device_ids=[local_rank])
    backbone_t.eval()
    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    # get header
    if cfg.loss == "ArcFace":
        header = losses.ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "CosFace":
        header = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(
            local_rank)
    else:
        print("Header not implemented")
    # load the teacher header
    try:
        header_pth = os.path.join(cfg.pretrained_teacher_header_path)
        header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))
        if rank == 0:
            logging.info("header resume loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("load header resume init, failed!")
    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.eval()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)


    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)

    criterion = CrossEntropyLoss()
    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    if cfg.global_step:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch

        # --------- this could be solved more elegant ----------------
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_last_lr()[0]

        # ------------------------------------------------------------

    callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = cfg.global_step
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)
            features = backbone(img)
            thetas  = header(F.normalize(features), label)
            loss_v =  criterion(thetas, label)
            loss_v.backward()
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            opt_backbone.step()
            opt_backbone.zero_grad()
            loss.update(loss_v.item(), 1)
            callback_logging(global_step, loss, epoch ,0.0,0.0,0.)
            callback_verification(global_step, backbone)
        scheduler_backbone.step()
        callback_checkpoint(global_step, backbone, None)
    callback_verification(-1, backbone)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch AMLDistill training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)
