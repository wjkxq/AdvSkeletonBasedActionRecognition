# encoding: utf-8

"""
@author: huguyuehuhu
@time: 18-3-25 下午3:54
Permission is given to modify the code, any problem please contact huguyuehuhu@gmail.com
"""
import sys
import argparse
import logging
import os
import random
import numpy as np
import torch
import json
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau
from torch.autograd import Variable
from tqdm import tqdm

tqdm.monitor_interval = 0

import torch.backends.cudnn as cudnn

from utils import utils
from utils.utils import str2bool
import data_loader

from torch.utils.data import DataLoader
import torchvision.utils as vutil

from model import HCN
from model import SGN
from model import TwoAGCN

# import setGPU

from defense import NTU_DEFENSE
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

# python main.py --mode train --model_name HCN --dataset_name NTU-RGB-D-CV --num 01
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./data/NTU', help="root directory for all the datasets")
parser.add_argument('--dataset_name', default='NTU-RGB-D-CV', help="dataset name")  # NTU-RGB-D-CS,NTU-RGB-D-CV
parser.add_argument('--model_dir', default='./', help="parents directory of model")

parser.add_argument('--model_name', default='HCN', help="model name")
parser.add_argument('--load_model',
                    help='Optional, load trained models')
parser.add_argument('--load',
                    type=str2bool,
                    default=False,
                    help='load a trained model or not ')
parser.add_argument('--mode', default='train', help='train,test,or load_train')
parser.add_argument('--num', default='01', help='num of trials (type: list)')

# 可视化
IMAGE_FOLDER = './visualizations_new/HCN_adv/'
INSTANCE_FOLDER = None


def train(model, optimizer, loss_fn, dataloader, metrics, params, logger,
          add_noise=False, noise_sigma=0.1):
    # set model to training mode
    model.train()

    if add_noise:
        defense = NTU_DEFENSE(mode='train', sigma=noise_sigma)

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    # confusion_meter = torchnet.meter.ConfusionMeter(params.model_args["num_class"], normalized=True)
    # confusion_meter.reset()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            # print(data_batch.size())
            # [64, 3, 32, 25, 2]
            # move to GPU if available
            if params.cuda:
                if params.data_parallel:
                    data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
                else:
                    data_batch, labels_batch = data_batch.cuda(params.gpu_id), labels_batch.cuda(params.gpu_id)

            # convert to torch Variables
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            if add_noise:
                # print(noise_sigma)
                data_batch.data = data_batch.data + noise_sigma * torch.randn_like(data_batch.data)

            # compute model output and loss
            output_batch = model(data_batch, target=labels_batch)

            loss_bag = loss_fn(output_batch, labels_batch, current_epoch=params.current_epoch, params=params)
            loss = loss_bag['ls_all']

            output_batch = output_batch
            # confusion_meter.add(output_batch.data,
            #                     labels_batch.data)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip * params.batch_size_train)

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while # not every epoch count in train accuracy
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data
                labels_batch = labels_batch.data

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.item()
                for l, v in loss_bag.items():
                    summary_batch[l] = v.data.item()

                summ.append(summary_batch)

            # update the average loss # main for progress bar, not logger
            loss_running = loss.data.item()
            loss_avg.update(loss_running)

            t.set_postfix(loss_running='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logger.info("- Train metrics: " + metrics_string)

    return metrics_mean


def evaluate(model, loss_fn, dataloader, metrics, params, logger, add_noise=False, noise_sigma=0.1):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    # model.train()
    if params.mode == 'test':
        pass
    else:
        model.eval()

    # summary for current eval loop
    summ = []
    # confusion_meter = torchnet.meter.ConfusionMeter(params.model_args["num_class"], normalized=True)
    # confusion_meter.reset()

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            if params.data_parallel:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
            else:
                data_batch, labels_batch = data_batch.cuda(params.gpu_id), labels_batch.cuda(params.gpu_id)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        if add_noise:
            data_batch.data = data_batch.data + noise_sigma * torch.randn_like(data_batch.data)
        # compute model output
        output_batch = model(data_batch)

        loss_bag = loss_fn(output_batch, labels_batch, current_epoch=params.current_epoch, params=params)
        loss = loss_bag['ls_all']

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data
        labels_batch = labels_batch.data

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        for l, v in loss_bag.items():
            summary_batch[l] = v.data.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logger.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def single_evaluate(model, loss_fn, dataloader, metrics, params, logger):
    # set model to evaluation mode
    # model.train()
    if params.mode == 'single_test' or params.mode == 'test':
        pass
    else:
        model.eval()

    # summary for current eval loop
    summ = []
    logits = []
    preds = []
    # confusion_meter = torchnet.meter.ConfusionMeter(params.model_args["num_class"], normalized=True)
    # confusion_meter.reset()

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            if params.data_parallel:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
            else:
                data_batch, labels_batch = data_batch.cuda(params.gpu_id), labels_batch.cuda(params.gpu_id)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        out = model(data_batch)
        output_batch = out

        logit, pred = F.log_softmax(output_batch, dim=1).topk(k=5, dim=1, largest=True, sorted=True)
        logits.append(logit)
        preds.append(pred)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data
        labels_batch = labels_batch.data

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        # summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logger.info("- Eval metrics : " + metrics_string)

    logits = torch.cat(logits, dim=0)
    preds = torch.cat(preds, dim=0)

    return metrics_mean, logits, preds


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir, logger, restore_file=None,
                       add_noise=False, noise_sigma=0.1):
    best_val_acc = 0.0
    # reload weights from restore_file if specified
    if restore_file is not None:
        logging.info("Restoring parameters from {}".format(restore_file))
        checkpoint = utils.load_checkpoint(restore_file, model, optimizer)
        params.start_epoch = checkpoint['epoch']

        best_val_acc = checkpoint['best_val_acc']
        print('best_val_acc=', best_val_acc, flush=True)
        print(optimizer.state_dict()['param_groups'][0]['lr'], checkpoint['epoch'], flush=True)

    # learning rate schedulers for different models:
    if params.lr_decay_type == None:
        logging.info("no lr decay")
    else:
        assert params.lr_decay_type in ['multistep', 'exp', 'plateau']
        logging.info("lr decay:{}".format(params.lr_decay_type))
    if params.lr_decay_type == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=params.lr_step, gamma=params.scheduler_gamma,
                                last_epoch=params.start_epoch - 1)

    elif params.lr_decay_type == 'exp':
        scheduler = ExponentialLR(optimizer, gamma=params.scheduler_gamma2,
                                  last_epoch=params.start_epoch - 1)
    elif params.lr_decay_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params.scheduler_gamma3, patience=params.patience,
                                      verbose=False,
                                      threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                      eps=1e-08)

    for epoch in range(params.start_epoch, params.num_epochs):
        params.current_epoch = epoch
        if params.lr_decay_type != 'plateau':
            scheduler.step()

        # Run one epoch
        logger.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics, params, logger, add_noise=add_noise,
                              noise_sigma=noise_sigma)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, logger, add_noise=add_noise,
                               noise_sigma=noise_sigma)

        # vis logger
        accs = [100. * (1 - train_metrics['accuracytop1']), 100. * (1 - train_metrics['accuracytop5']),
                100. * (1 - val_metrics['accuracytop1']), 100. * (1 - val_metrics['accuracytop5']), ]

        losses = [train_metrics['loss'], val_metrics['loss']]

        if params.lr_decay_type == 'plateau':
            scheduler.step(val_metrics['ls_all'])

        val_acc = val_metrics['accuracytop1']
        is_best = val_acc >= best_val_acc
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'best_val_acc': best_val_acc
                               },
                              epoch=epoch + 1,
                              is_best=is_best,
                              save_best_ever_n_epoch=params.save_best_ever_n_epoch,
                              checkpointpath=params.experiment_path + '/checkpoint',
                              start_epoch=params.start_epoch)

        val_metrics['best_epoch'] = epoch + 1
        # If best_eval, best_save_path, metric
        if is_best:
            logger.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(params.experiment_path, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(params.experiment_path, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


def test_only(model, train_dataloader, val_dataloader, optimizer,
              loss_fn, metrics, params, model_dir, logger, restore_file=None):
    # reload weights from restore_file if specified
    if restore_file is not None:
        logging.info("Restoring parameters from {}".format(restore_file))
        checkpoint = utils.load_checkpoint(restore_file, model, optimizer)

        best_val_acc = checkpoint['best_val_acc']
        params.current_epoch = checkpoint['epoch']
        print('best_val_acc=', best_val_acc, flush=True)
        print(optimizer.state_dict()['param_groups'][0]['lr'], checkpoint['epoch'], flush=True)

    model.eval()
    # train_metrics = evaluate(model, loss_fn, train_dataloader, metrics, params, logger)

    model.eval()
    val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, logger)

    # 可视化
    visualization(model)

    pass


def test_one_file(model):
    model.eval()
    data = np.load('experiments/NTU-RGB-D-CV/HCN01/beta_10_untargeted_adv_data.npz',
                   mmap_mode=None)
    print('测试集大小为：', len(data['natdata']))
    for i in range(len(data['natdata'])):
        data_sample_adv = data['advdata'][i]
        data_sample = data['natdata'][i]
        label = data['natlabels'][i]
        data_sample_adv, data_sample, label = torch.tensor(data_sample_adv).cuda(), torch.tensor(data_sample).cuda(), torch.tensor(label).cuda()

        C, T, V, M = data_sample.shape
        data_sample = data_sample.reshape((1, C, T, V, M))
        data_sample_adv = data_sample_adv.reshape((1, C, T, V, M))
        output = model(data_sample)
        # output_adv = model(data_sample_adv)
        _, pred = output.topk(1, 1, True, True)
        # _, pred_adv = output_adv.topk(1, 1, True, True)
        # if pred.data.item() == label.data.item():# and pred_adv.data.item() == 0:
        #     print(i)
        print(pred.data.item(), label.data.item())#, pred_adv.data.item())
    print('end')
    # 可视化
    visualization(model)


def hook_func(model, input, output):
    image_name = get_image_name_for_hook(model)
    data = output.clone().detach()
    print(data.size())

    if len(data.size()) == 4 and data.size()[0] == 1:
        data = data.permute(1, 0, 2, 3)
        vutil.save_image(data, image_name, pad_value=0.5)


def get_image_name_for_hook(module):
    """
    Generate image filename for hook function

    Parameters:
    -----------
    module: module of neural network
    """
    # print(INSTANCE_FOLDER)
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    base_name = str(module).split('(')[0]
    index = 0
    image_name = '.'  # '.' is surely exist, to make first loop condition True
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(
            INSTANCE_FOLDER, '%s_%d.png' % (base_name, index))
    return image_name


def visualization(model):
    model.eval()
    modules_for_plot = (torch.nn.ReLU, torch.nn.Conv2d,
                        torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d)
    for name, module in model.named_modules():
        if isinstance(module, modules_for_plot):
            module.register_forward_hook(hook_func)

    params.batch_size_test = 1
    test_loader = data_loader.fetch_dataloader('test', params)

    # test_loader = DataLoader(dataset=test_dl,
    #                          batch_size=1,
    #                          shuffle=False)
    index = 1
    for data, classes in test_loader:
        global INSTANCE_FOLDER
        INSTANCE_FOLDER = os.path.join(
            IMAGE_FOLDER, '%d-%d' % (index, classes.item()))
        # print(INSTANCE_FOLDER, index)
        data, classes = data.cuda(), classes.cuda()
        outputs = model(data)

        index += 1
        if index > 20:
            break



# python main.py --mode train --model_name HCN --dataset_name
# NTU-RGB-D-CV --num 01
if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    experiment_path = os.path.join(args.model_dir, 'experiments', args.dataset_name, args.model_name + args.num)
    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)

    json_file = os.path.join(experiment_path, 'params.json')
    if not os.path.isfile(json_file):
        with open(json_file, 'w') as f:
            print("No json configuration file found at {}".format(json_file), flush=True)
            f.close()
            print('successfully made file: {}'.format(json_file), flush=True)

    params = utils.Params(json_file)

    if args.load:
        print("args.load=", args.load, flush=True)
        if args.load_model:
            params.restore_file = args.load_model
        else:
            params.restore_file = experiment_path + '/checkpoint/best.pth.tar'

    params.dataset_dir = args.dataset_dir
    params.dataset_name = args.dataset_name
    params.model_version = args.model_name
    params.experiment_path = experiment_path
    params.mode = args.mode
    if params.gpu_id >= -1:
        params.cuda = True

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    if params.gpu_id >= -1:
        torch.cuda.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = False  # must be True to if you want reproducible,but will slow the speed

    cudnn.benchmark = True  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.cuda.empty_cache()  # release cache
    # Set the logger
    if params.mode == 'train':
        utils.set_logger(os.path.join(experiment_path, 'train.log'))
    elif params.mode == 'test':
        utils.set_logger(os.path.join(experiment_path, 'test.log'))
    elif params.mode == 'load_train':
        utils.set_logger(os.path.join(experiment_path, 'load_train.log'))


    logger = logging.getLogger()

    port, env = 8097, params.model_version
    columnnames, rownames = list(range(1, params.model_args["num_class"] + 1)), list(
        range(1, params.model_args["num_class"] + 1))

    # log all params
    d_args = vars(args)
    for k in d_args.keys():
        logging.info('{0}: {1}'.format(k, d_args[k]))
    d_params = vars(params)
    for k in d_params.keys():
        logger.info('{0}: {1}'.format(k, d_params[k]))

    if 'HCN' in params.model_version:
        model = HCN.HCN(**params.model_args)
        if params.data_parallel:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(params.gpu_id)

        loss_fn = HCN.loss_fn
        metrics = HCN.metrics
    elif 'SGN' in params.model_version:
        model = SGN.SGN(num_classes=60, seg=20, batch_size=1)
        if params.data_parallel:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(params.gpu_id)

        loss_fn = HCN.loss_fn
        metrics = HCN.metrics
    elif '2sagcn' in params.model_version:
        model = TwoAGCN.TwoAGCN()
        if params.data_parallel:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(params.gpu_id)

        loss_fn = HCN.loss_fn
        metrics = HCN.metrics


    if params.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr, betas=(0.9, 0.999),
                               eps=1e-8,
                               weight_decay=params.weight_decay)

    elif params.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr, momentum=0.9,
                              nesterov=True, weight_decay=params.weight_decay)

    logger.info(model)
    # Create the input data pipeline
    logger.info("Loading the datasets...")
    # fetch dataloaders
    train_dl = data_loader.fetch_dataloader('train', params)
    test_dl = data_loader.fetch_dataloader('test', params)
    logger.info("- done.")

    if params.mode == 'train' or params.mode == 'load_train':
        # Train the model
        logger.info("Starting training for {} epoch(s)".format(params.num_epochs))
        print(params.dict.keys())
        if "extra_training_params" in params.dict.keys():
            train_and_evaluate(model, train_dl, test_dl, optimizer, loss_fn, metrics, params,
                               args.model_dir, logger, params.restore_file,
                               add_noise=params.extra_training_params['add_noise'],
                               noise_sigma=params.extra_training_params['sigma'])
        else:
            train_and_evaluate(model, train_dl, test_dl, optimizer, loss_fn, metrics, params,
                               args.model_dir, logger, params.restore_file)
    elif params.mode == 'test':
        test_only(model, train_dl, test_dl, optimizer,
                  loss_fn, metrics, params, args.model_dir, logger, params.restore_file)
    elif params.mode == 'test_one':
        test_one_file(model)
    else:
        print('mode input error!', flush=True)
