import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test # import test.py to get mAP after each epoch
from models import *
from utils.datasets import * #three channel images
from utils.utils_mb import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'



def infi_loop(dl):
    while True:
        for (imgs, targets, paths, _) in dl:
            yield imgs, targets, paths


def train():
    # Empty Cuda First
    torch.cuda.empty_cache()

    cfg = opt.cfg
    data = opt.data
    img_size, img_size_test = opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2  # train, test sizes
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    supplement_batch_size = opt.supplement_batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    # experiment_name = opt.experiment_name

    experiment_name = hyp['experiment_name']
    print(experiment_name)

    # Initialize
    init_seeds()
    if opt.multi_scale:
        img_sz_min = round(img_size / 32 / 1.5)
        img_sz_max = round(img_size / 32 * 1.5)
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    synth_path = data_dict['supplement']
    test_path = data_dict['valid']
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    image_number = data_dict['nImages'] # number of images in dataset
    loop_count = int(image_number) // batch_size

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg, arc=opt.arc).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    # https://github.com/alphadl/lookahead.pytorch
    # optimizer = torch_utils.Lookahead(optimizer, k=5, alpha=0.5)

    start_epoch = 0
    best_fitness = 0.0
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif weights.endswith('.pth'):
        model.load_state_dict(torch.load(weights), strict=False)

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    scheduler = lr_scheduler.StepLR(optimizer, hyp['step'], gamma=hyp['step_size'], last_epoch=-1) # step schedule
    scheduler.last_epoch = start_epoch - 1

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  cache_labels=True,
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)
    
    
    
    synth_dataset = LoadImagesAndLabels(synth_path, img_size, batch_size,
                                       augment=True,
                                       hyp=hyp, 
                                       rect=False, 
                                       cache_labels=True, 
                                       cache_images=opt.cache_images,
                                       single_cls=opt.single_cls)

    
    batch_size = min(batch_size, len(dataset))
    nw = 8  # number of workers
    # Calculate Correct Batch Size for Testing
    batch_size_test = 0
    if batch_size == 1:
        batch_size_test = 32
    else:
        batch_size_test = batch_size * 2
    
    test_dataset = LoadImagesAndLabels(test_path, img_size_test, batch_size_test,
                                                                 hyp=hyp,
                                                                 rect=False,
                                                                 cache_labels=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls)
    
    # Dataloader
    # Proceed to Create Dataloaders
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size-supplement_batch_size,
                                             num_workers=nw,
                                             shuffle=True,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)
    
    # coco dataloader
    synth_dataloader = torch.utils.data.DataLoader(synth_dataset,
                                                  batch_size=supplement_batch_size,
                                                  num_workers=nw, 
                                                  shuffle=True, 
                                                  pin_memory=True, 
                                                  collate_fn=coco_dataset.collate_fn)

    # Testloader
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size_test,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=test_dataset.collate_fn)

    # Start training
    nb = loop_count
    prebias = start_epoch == 0
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------
        model.train()
        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        if dataloader:
            gen_ir_data = infi_loop(dataloader)
        if synth_dataloader:
            gen_synth_data = infi_loop(coco_dataloader)

        pbar = tqdm(range(0, nb), total=nb)
        for i in pbar:
            if gen_ir_data and gen_synth_data:
                imgs_ir, targets_ir, paths_ir = next(gen_ir_data)
                imgs_synth, targets_synth, paths_synth = next(gen_synth_data)
                mixed_batch_size = batch_size - supplement_batch_size

                for si in reversed(range(mixed_batch_size)):
                    targets_ir[targets_ir[:,0] == si, 0] = supplement_batch_size + si

                imgs = torch.cat([imgs_synth, imgs_ir], dim=0)
                targets = torch.cat([targets_synth, targets_ir],dim=0)
                paths = paths_synth + paths_ir

            ni = i + nb * epoch  # number integrated batches (since train start)
            # imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            imgs = imgs.to(device).float()  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            # print(imgs.shape)

            # Multi-Scale training
            if opt.multi_scale:
                # Change Was made Here it used to be 10. I trained the grayscale weight son 10.
                if ni / accumulate % 1 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes
            if ni < 5:
                fname = 'train_mixedbatch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model, not prebias)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets), img_size)
            # print(s)
            pbar.set_description(s)

        # end batch ------------------------------------------------------------------------------------------------
        # Update scheduler
        scheduler.step()

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        results, maps = test.test(cfg,
                                  data,
                                  hyp,
                                  batch_size=32,
                                  img_size=img_size_test,
                                  model=model,
                                  conf_thres=0.1,
                                  iou_thres=0.5,
                                  save_json=True,
                                  single_cls=opt.single_cls,
                                  dataloader=testloader)



        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results)
            titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
                      'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification']
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save training results
        save = True
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)

            # Save best checkpoint
            if best_fitness == fi:
                torch.save(chkpt, best)

            # Save backup every 5 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, wdir + experiment_name + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, 'last%s.pt' % n, 'best%s.pt' % n
        os.rename('results.txt', fresults)
        os.rename(wdir + 'last.pt', wdir + flast) if os.path.exists(wdir + 'last.pt') else None
        os.rename(wdir + 'best.pt', wdir + fbest) if os.path.exists(wdir + 'best.pt') else None
        if opt.bucket:  # save to cloud
            os.system('gsutil cp %s gs://%s/results' % (fresults, opt.bucket))
            os.system('gsutil cp %s gs://%s/weights' % (wdir + flast, opt.bucket))
            # os.system('gsutil cp %s gs://%s/weights' % (wdir + fbest, opt.bucket))

    if not opt.evolve:
        plot_results()  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=4)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--supplement-batch-size', type=int, default=1)  #number of images in a batch that come from supplementary dataset
    parser.add_argument('--accumulate', type=int, default=1, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='/yolov3/cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='/yolov3/data/X.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640], help='train and test image-sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='/yolov3/weights/X', help='initial weights')
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--experiment-name', type=str, default='name', help='experiment name')
    parser.add_argument('--lr0', type=float, default=1E-4, help='train as single-class dataset')
    parser.add_argument('--lrf', type=float, default=1E-8, help='train as single-class dataset')
    parser.add_argument('--step', type=int, default=20, help='train as single-class dataset')
    parser.add_argument('--step-size', type=float, default=1E-1, help='train as single-class dataset')
    opt = parser.parse_args()

    # Hyperparameters (results68: 59.9 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310
    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)py
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': 0.2,  # iou training threshold
           'lr0': opt.lr0,
           'lrf': opt.lrf,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 0.000484,  # optimizer weight decay
           'fl_gamma': 0.5,  # focal loss gamma
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98,  # image rotation (+/- deg)
           'translate': 0.05,  # image translation (+/- fraction)
           'scale': 0.05,  # image scale (+/- gain)
           'shear': 0.641, # image shear (+/- deg)
           'experiment_name': opt.experiment_name + expName,
           'step': opt.step,
           'step_size': opt.step_size}

    # Overwrite hyp with hyp*.txt (optional)
    f = glob.glob('hyp*.txt')
    if f:
        print('Using %s' % f[0])
        for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
            hyp[k] = v

    # torch.manual_seed(0)
    opt.weights = last if opt.resume else opt.weights
    print(opt)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False


    tb_writer = None
    if not opt.evolve:  # Train normally
        try:
            # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter()
        except:
            pass

        train()  # train normally
