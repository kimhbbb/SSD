import torch
import torch.nn as nn
from model import *
from dataset import PascalVOCDataset
import argparse
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

data_folder = './SSD'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

num_classes = 21
batch_size = 8  # batch size
iterations = 10000  # number of iterations to train
num_workers = 4  # number of workers for loading data in the DataLoader
num_epochs = 15
lr = 1e-3
weight_decay = 5e-4
# grad_clip_norm = 0.1
momentum = 0.9



decay_lr_at = [80000, 100000] 
decay_lr_to = 0.1
grad_clip = None
print_freq = 200


def train():
    global label_map, epoch, decay_lr_at

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("My device: {}".format(device))

    model = SSD(num_classes)
    model = model.to(device)
    model.train()

    train_dataset = PascalVOCDataset(data_folder, split='train',
                                     keep_difficult = keep_difficult)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle = True,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               collate_fn = train_dataset.collate_fn) 

    criterion = MultiBoxLoss(priors_cxcy = model.priors_cxcy).to(device)

    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        # .bias로 끝나면 bias, .bias가 아니면 weight
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    # 보통은 model.parameters()를 넣지만, 지금은 bias의 학습률을 2배로 하기 위해서 이렇게 설정함.
    # momentum: 이전 기울기 정보를 활용하여 경사 방향을 안정적으로 만듦.
    # weight_decay: weight가 일정 수준 이상 커지지 않도록 함.
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]


    for epoch in range(num_epochs):
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1} / {num_epochs}")
        
        losses = AverageMeter()

        for i, (images, boxes, labels, _) in loop:
            # print(f"Batch {itertaion}: {images.shape}") # torch.Size([8, 3, 300, 300])

            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            predicted_locs, predicted_scores = model(images)

            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            optimizer.zero_grad()  
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            # Clip gradients, if necessary
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)


            optimizer.step()

            losses.update(loss.item(), images.size(0))

            loop.set_postfix(loss=loss.item())

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses))


            loop.update(1)

        # scheduler.step()
        # print(f"Epoch {epoch + 1}: Learning rate = {scheduler.get_last_lr()}")

        save_checkpoint(epoch, model, optimizer)

if __name__ == '__main__':
    train()