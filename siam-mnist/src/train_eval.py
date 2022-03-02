import torch
from batch import *
import tqdm

def train_val(model, device, scaler, optimizer, criterion, num_epochs, w, h, c, x_train, y_train, x_test, y_test, batch_size_train, batch_size_test):
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}
    batch_size = {'train': batch_size_train, 'val': batch_size_test}

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loop = tqdm(range(batch_size_train))
            else:
                loop = range(batch_size_test)
                model.eval()

            # Iterate over data
            for i in enumerate(loop):
                [x_anchors, x_positives, x_negatives, labels] = create_batch(x_train, y_train, x_test, y_test, w, h, c, batch_size=batch_size[phase], split=phase)

                x_anchors = x_anchors.to(device)
                x_positives = x_positives.to(device)
                x_negatives = x_negatives.to(device)
                labels = labels.to(device)

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(): # to use float16 training
                        outputsA = model(x_anchors)
                        outputsP = model(x_positives)
                        outputsN = model(x_negatives)

                        loss = criterion(outputsA, outputsP, outputsN)

                    # find the running loss and accuracy
                    running_loss = loss.item()

                    losses[phase].append(running_loss)

                    # backward
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        loop.set_postfix(loss=running_loss)
        
    print( "saving model...")
    checkpoint = {
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
