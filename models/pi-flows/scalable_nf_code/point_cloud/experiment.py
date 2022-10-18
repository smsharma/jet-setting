import logging
import time
import data
from copy import deepcopy
import numpy as np
import stribor as st
import torch

def get_exact_model(dim, hidden_dims, latent_dim):
    transforms = []
    for _ in range(12):
        transforms.append(st.ContinuousNormalizingFlow(dim, net=st.net.DiffeqExactTraceDeepSet(dim, hidden_dims, dim, d_h=latent_dim),
                                            divergence='exact', solver='dopri5', atol=1e-4))
    model = st.Flow(st.Normal(torch.zeros(dim), torch.ones(dim)), transforms)
    return model

def get_approx_model(dim, hidden_dims, *args):
    transforms = []
    for _ in range(12):
        transforms.append(st.ContinuousNormalizingFlow(dim, net=st.net.DiffeqDeepset(dim, hidden_dims, dim),
                                            divergence='approximate', solver='dopri5', atol=1e-4))
    model = st.Flow(st.Normal(torch.zeros(dim), torch.ones(dim)), transforms)
    return model

if __name__ == '__main__':
    dataset = 'airplane' # airplane or chair or mnist
    model_name = 'approximate' # exact or approximate
    batch_size = 128
    learning_rate = 1e-3
    learning_rate_decay = 0.5
    learning_rate_decay_step = 50
    weight_decay = 1e-4
    epochs = 1
    display_step = 1
    patience = 20
    latent_dim = 0
    hidden_dims = [128, 128, 128]

    # Load data
    dset = data.load_dataset(dataset)
    trainset, valset, testset = dset.split_train_val_test()
    dim = trainset.dim

    collate = data.collate
    dl_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dl_val = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    dl_test = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Load model
    if model_name == 'exact':
        model = get_exact_model(dim, hidden_dims, latent_dim)
    elif model_name == 'approximate':
        model = get_approx_model(dim, hidden_dims)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, learning_rate_decay_step, learning_rate_decay)

    ## Training
    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    training_val_losses, epoch_durations, num_evals = [], [], []

    for epoch in range(epochs):
        # Optimization
        model.train()
        start_time = time.time()
        for batch in dl_train:
            optimizer.zero_grad()

            x, _ = batch
            log_prob = model.log_prob(x)
            loss = -log_prob.sum(-1).mean()

            num_evals.append(sum([t.num_evals() for t in model.transforms]))

            loss.backward()
            optimizer.step()
        epoch_durations.append(time.time() - start_time)

        # Validation
        model.train()
        loss_val = 0
        for _, batch in enumerate(dl_val):
            x, _ = batch
            log_prob = model.log_prob(x)
            loss_val -= log_prob.sum(-1).mean() / len(dl_val)
        training_val_losses.append(loss_val.item())

        scheduler.step()

        # Early stopping
        if (best_loss - loss_val) < 1e-4:
            impatient += 1
            if loss_val < best_loss:
                best_loss = loss_val.item()
                best_model = deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'models/{model_name}_{dataset}_{latent_dim}_new.model')
        else:
            best_loss = loss_val.item()
            best_model = deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'models/{model_name}_{dataset}_{latent_dim}_new.model')
            impatient = 0

        if impatient >= patience:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break
        if torch.isnan(loss_val):
            print(f'Breaking due to nan loss at epoch {epoch}')
            break

        if (epoch + 1) % display_step == 0:
            print(f"Epoch {epoch+1:4d}, loss_train = {loss:.4f}, loss_val = {loss_val:.4f}")

    ## Test model
    model.load_state_dict(best_model)
    model.eval()
    start_time = time.time()

    test_loss = 0
    for _, batch in enumerate(dl_test):
        x, m = batch
        log_prob = model.log_prob(x, mask=m)
        test_loss -= (log_prob * m).sum() / m.sum() / len(dl_test)
    test_time = time.time() - start_time
