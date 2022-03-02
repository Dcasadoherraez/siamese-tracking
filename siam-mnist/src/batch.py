import torch
import random
import numpy as np

def create_batch(x_train, y_train, x_test, y_test, w, h, c, batch_size=256, split = "train"):
    x_anchors = torch.zeros((batch_size, h, w, c))
    x_positives = torch.zeros((batch_size, h, w, c))
    x_negatives = torch.zeros((batch_size, h, w, c))
    ys = torch.zeros((batch_size))
    
    if split =="train":
        data = x_train
        data_y = y_train
    else:
        data = x_test
        data_y = y_test
    
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, data.shape[0] - 1)
        x_anchor = data[random_index]
        y = data_y[random_index]
        
        indices_for_pos, _ = torch.where(data_y == y)
        indices_for_neg, _ = torch.where(data_y != y)

        random_pos_sample = indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]
        x_positive = data[random_pos_sample]
        random_neg_sample = indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]
        x_negative = data[random_neg_sample]
        
        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        ys[i] = y
        
    return [x_anchors.permute(0,3,1,2), x_positives.permute(0,3,1,2), x_negatives.permute(0,3,1,2), ys]


def create_hard_batch(model, x_train, y_train, x_test, y_test, w, h, c, num_hard, batch_size=256, split = "train"):
    
    x_anchors = np.zeros((batch_size, w, h, c))
    x_positives = np.zeros((batch_size, w, h, c))
    x_negatives = np.zeros((batch_size, w, h, c))
    
    if split =="train":
        data = x_train
        data_y = y_train
    else:
        data = x_test
        data_y = y_test
    
    # Generate num_hard number of hard examples:
    hard_batches = [] 
    batch_losses = []
    
    rand_batches = []
    
    # Get some random batches
    for i in range(0, batch_size):
        hard_batches.append(create_batch(x_train, y_train, x_test, y_test, w, h, c, batch_size=1, split=split))
        
        A_emb = model.predict(hard_batches[i][0])
        P_emb = model.predict(hard_batches[i][1])
        N_emb = model.predict(hard_batches[i][2])
        
        # Compute d(A, P) - d(A, N) for each selected batch
        batch_losses.append(np.sum(np.square(A_emb-P_emb),axis=1) - np.sum(np.square(A_emb-N_emb),axis=1))
    
    # Sort batch_loss by distance, highest first, and keep num_hard of them
    hard_batch_selections = [x for _, x in sorted(zip(batch_losses,hard_batches), key=lambda x: x[0])]
    hard_batches = hard_batch_selections[:num_hard]
    
    # Get batch_size - num_hard number of random examples
    num_rand = batch_size - num_hard
    for i in range(0, num_rand):
        rand_batch = create_batch(x_train, y_train, x_test, y_test, w, h, c, batch_size=1, split=split)
        rand_batches.append(rand_batch)
    
    selections = hard_batches + rand_batches
    
    for i in range(0, len(selections)):
        x_anchors[i] = selections[i][0]
        x_positives[i] = selections[i][1]
        x_negatives[i] = selections[i][2]
        
    return [x_anchors, x_positives, x_negatives]
    