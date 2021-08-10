import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from PIL import Image
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2

writer = SummaryWriter('runs/G1G2')
SIZE = 320
NC = 14


def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256, 192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256, 192)

    return label_batch


def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)

    return input_label


def complete_compose(img, mask, label):
    label = label.cpu().numpy()
    M_f = label > 0
    M_f = M_f.astype(np.int)
    M_f = torch.FloatTensor(M_f).cuda()
    masked_img = img * (1 - mask)
    M_c = (1 - mask.cuda()) * M_f
    M_c = M_c + torch.zeros(img.shape).cuda()  ##broadcasting
    return masked_img, M_c, M_f


def compose(label, mask, color_mask, edge, color, noise):
    # check=check>0
    # print(check)
    masked_label = label * (1 - mask)
    masked_edge = mask * edge
    masked_color_strokes = mask * (1 - color_mask) * color
    masked_noise = mask * noise
    return masked_label, masked_edge, masked_color_strokes, masked_noise


def changearm(old_label):
    label = old_label
    arm1 = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.int))
    arm2 = torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.int))
    noise = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.int))
    label = label * (1 - arm1) + arm1 * 4
    label = label * (1 - arm2) + arm2 * 4
    label = label * (1 - noise) + noise * 4
    return label

def iou_mean(pred, target, n_classes = 1):
#n_classes ：the number of classes in your dataset,not including background
# for mask and ground-truth label, not probability map
  ious = []
  iousSum = 0
  pred = torch.from_numpy(pred)
  pred = pred.view(-1)
  target = np.array(target)
  target = torch.from_numpy(target)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
      iousSum += float(intersection) / float(max(union, 1))
  return iousSum/n_classes

os.makedirs('sample', exist_ok=True)
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('# Inference images = %d' % dataset_size)

model = create_model(opt)

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

step = 0

mIoU_loss = 0

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):

        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        # save_fake = total_steps % opt.display_freq == display_delta
        save_fake = True

        ##add gaussian noise channel
        ## wash the label
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
        #
        # data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
        mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
        img_fore = data['image'] * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(data['label'])

        ############## Forward Pass ######################
        losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
            Variable(data['label'].cuda()), Variable(data['edge'].cuda()), Variable(img_fore.cuda()),
            Variable(mask_clothes.cuda())
            , Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()),
            Variable(data['pose'].cuda()), Variable(data['image'].cuda()), Variable(mask_fore.cuda()))

        ############## Display results and errors ##########

        ### display output images
        a = generate_label_plain(input_label).float().cuda()
        # b = real_image.float().cuda()
        # c = fake_image.float().cuda()
        # d = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
        # combine = torch.cat([a[0], d[0], b[0], c[0], rgb[0]], 2).squeeze()
        combine = a[0].squeeze()


        # combine=c[0].squeeze()
        cv_img = combine.permute(0, 1).detach().cpu().numpy().astype(np.uint8)
        extract_mask = cv_img < 14
        cv_img = extract_mask.astype(np.uint8) * cv_img
        print(np.unique(cv_img))
        weak_clothes_mask = (data['label'].cpu().numpy().astype(np.uint8) == 4)
        weak_clothes = (weak_clothes_mask * data['label'].cpu().numpy().astype(np.int))
        final_data = weak_clothes.squeeze() + cv_img
        final_data[final_data == 17] = 13
        final_data[final_data == 16] = 12
        final_data[final_data == 15] = 11
        final_data[final_data == 14] = 10

        mIoU_loss += iou_mean(pred=final_data.astype(np.uint8), target=data['label'], n_classes=13)
        print(mIoU_loss)

        print(np.unique(final_data))
        if step % 1 == 0:
            # writer.add_image('combine', (combine.data + 1) / 2.0, step)
            rgb = (final_data).astype(np.uint8)
            print(np.unique(rgb))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            n = str(step) + '.png'
            cv2.imwrite('sample/' + data['name'][0], bgr.astype(np.uint8))
            img = cv2.imread('sample/' + data['name'][0])
            npy = np.asarray(img)
            print(np.unique(npy))
        step += 1
        print(step)
        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            # print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            # model.module.save('latest')
            # np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            pass
        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    break

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        # np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
