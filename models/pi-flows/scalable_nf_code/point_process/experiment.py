import logging
import numpy as np
import time
import torch
import data
import stribor as st
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KDTree
from copy import deepcopy

if __name__ == '__main__':
    dataset = 'mixture_normal_64'
    model = 'DiffeqZeroTraceDeepSet'
    batch_size = 64
    learning_rate = 1e-3
    learning_rate_decay = 0.9
    learning_rate_decay_step = 50
    weight_decay = 1e-4
    epochs = 2
    display_step = 1
    patience = 50

    # params
    hidden_dims = [64, 64]
    num_layers = 1
    solver = 'dopri5'
    n_heads = 4
    pooling = 'max'
    latent_dim = 8

    ## Load data
    dset = data.load_dataset(dataset)
    trainset, valset, testset = dset.split_train_val_test()

    collate = data.collate
    dl_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    dl_val = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    dl_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate)

    ## Define model
    dim = trainset.dim
    transforms = [st.Logit()]
    for _ in range(num_layers):
        transforms.append(
            st.ContinuousNormalizingFlow(
                dim,
                net=getattr(st.net, model)(dim, hidden_dims, dim, d_h=latent_dim),
                divergence='exact' if 'exact' in model.lower() or 'zero' in model.lower() else 'approximate',
                solver=solver,
                set_data=True
            )
        )
    transforms.append(st.Sigmoid())

    model = st.Flow(st.Uniform(torch.zeros(dim), torch.ones(dim)), transforms)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, learning_rate_decay_step, learning_rate_decay)

    ## Training
    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    training_val_losses, epoch_durations = [], []

    for epoch in range(epochs):
        # Optimization
        model.train()
        start_time = time.time()
        for batch in dl_train:
            optimizer.zero_grad()

            x, m = batch
            log_prob = model.log_prob(x, mask=m)
            loss = -(log_prob * m).sum() / m.sum()

            loss.backward()
            optimizer.step()

        epoch_durations.append(time.time() - start_time)

        # Validation
        model.train()
        loss_val = 0
        for _, batch in enumerate(dl_val):
            x, m = batch
            log_prob = model.log_prob(x, mask=m)
            loss_val -= (log_prob * m).sum() / m.sum() / len(dl_val)
        training_val_losses.append(loss_val.item())

        scheduler.step()

        # Early stopping
        if (best_loss - loss_val) < 1e-4:
            impatient += 1
            if loss_val < best_loss:
                best_loss = loss_val.item()
                best_model = deepcopy(model.state_dict())
        else:
            best_loss = loss_val.item()
            best_model = deepcopy(model.state_dict())
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

    ## Sampling quality -- Wasserstein score
    dist_test, dist_samples = [], []
    for x in testset:
        if len(x[0]) > 2:
            dist_test.append(KDTree(x[0]).query(x[0], k=2)[0][:,1])
            samples = model.sample(len(x[0])).detach().cpu().numpy()
            dist_samples.append(KDTree(samples).query(samples, k=2)[0][:,1])

    dist_test = np.concatenate(dist_test, 0)
    dist_samples = np.concatenate(dist_samples, 0)

    wasserstein = float(wasserstein_distance(dist_test, dist_samples))
