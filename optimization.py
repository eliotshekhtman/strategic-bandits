import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

def l1norm_gragent(X, phi, b, u):
    m, d = phi.shape
    one_m = torch.ones(m)
    X_hat = torch.rand(d)
    X_hat.requires_grad_(True)
    # optimizer = torch.optim.SGD([X_hat], lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam([X_hat], lr=0.0001)
    for _ in range(300):
        optimizer.zero_grad()
        action = phi @ X_hat + b
        reward = u @ action / (one_m @ action)
        cost = torch.sum((X - X_hat) ** 2)
        loss = - (reward - 0.5 * cost)
        loss.backward()
        optimizer.step()
        # print(loss)
    return X_hat

def softmax_gragent(X, phi, b, u):
    m, d = phi.shape
    one_m = torch.ones(m)
    X_hat = torch.rand(d)
    X_hat.requires_grad_(True)
    softmax = nn.Softmax()
    optimizer = torch.optim.SGD([X_hat], lr=0.01, momentum=0.9)
    for _ in range(300):
        optimizer.zero_grad()
        reward = u @ softmax(phi @ X_hat + b)
        cost = torch.sum((X - X_hat) ** 2)
        loss = - (reward - 0.5 * cost)
        loss.backward()
        optimizer.step()
        # print(loss)
    return X_hat

def get_xW(X_hat, Rwd, phi, b):
    one_m = torch.ones(m)
    xW = torch.rand(d)
    xW.requires_grad_(True)
    action = (phi @ X_hat + b) / (one_m @ (phi @ X_hat + b))
    optimizer = torch.optim.Adam([xW], lr=0.0001)
    for _ in range(1000):
        optimizer.zero_grad()
        action = (phi @ X_hat + b) / (one_m @ (phi @ X_hat + b))
        loss = (Rwd - xW @ action) ** 2 + (1 - torch.norm(xW)) ** 2
        loss.backward()
        optimizer.step()
    return xW 

def old_dm_reward(X, Xhat, W, phi, b):
    one_m = torch.ones(m)
    action = (phi @ Xhat + b) / (one_m @ (phi @ Xhat + b))
    return (X @ W) @ action

def ag_reward(Xhat, phi, b, u):
    one_m = torch.ones(m)
    action = (phi @ Xhat + b) / (one_m @ (phi @ Xhat + b))
    return u @ action

def opt_phi(Xhats, xWs, u): # no additional penalty for finding x?
    one_m = torch.ones(m)
    phi = torch.rand((m, d))
    phi.requires_grad_(True)
    b = torch.zeros(m)
    # b.requires_grad_(True)
    optimizer = torch.optim.Adam([phi], lr=0.0001)
    relu = nn.ReLU()
    # optimizer = torch.optim.SGD([phi], lr=0.01, momentum=0.9)
    for _ in range(10000):
        optimizer.zero_grad()
        norm_phi = relu(phi) / torch.max(phi) # + torch.rand((m, d)) * 0.05
        actions = (norm_phi @ Xhats.T + b.reshape(-1, 1)) / (one_m @ (norm_phi @ Xhats.T + b.reshape(-1, 1))) # (md @ dn + m) / (m @ (md @ dn + m)) --> mn
        loss = - torch.mean(xWs @ actions) # + (1 - torch.norm(phi)) ** 2 # mean(nm @ mn) --> 1
        # print(torch.mean(xWs @ actions).item(), loss.item())
        loss.backward()
        optimizer.step()
    return (relu(phi) / torch.max(phi)).detach(), b




def softmax(v, temp=1.):
    sm = nn.Softmax(dim=0)
    return sm(v / temp)

def classify(x, phi, temp=0.02):
    return softmax( phi @ x.T , temp=temp)

def cost(x, x_hat):
    return torch.norm(x - x_hat) ** 2

def agent_reward(u, phi, x, x_hat, temp):
    return u @ classify(x_hat, phi, temp=temp) - cost(x, x_hat)

def gragent(X, phi, u, temp=0.02, attempts=5, iters=300):
    m, d = phi.shape
    best_X_hat = X.clone()
    # best_X_hat = torch.concat((torch.rand(d - 1), torch.ones(1)))
    best_reward = agent_reward(u, phi, x=X, x_hat=best_X_hat, temp=temp)
    X_hat = X[:-1].clone()
    X_hat.requires_grad_(True)
    optimizer = torch.optim.SGD([X_hat], lr=0.01, momentum=0.9)
    for _ in range(iters):
        X_hat_aug = torch.concat((X_hat, torch.ones(1)))
        optimizer.zero_grad()
        loss = -agent_reward(u, phi, X_hat_aug, X_hat_aug, temp)
        loss.backward()
        # print(loss)
        optimizer.step()
    for _ in range(iters):
        X_hat_aug = torch.concat((X_hat, torch.ones(1)))
        optimizer.zero_grad()
        loss = -agent_reward(u, phi, X, X_hat_aug, temp)
        loss.backward()
        # print(loss)
        optimizer.step()
    if loss < -best_reward:
        best_reward = -loss.detach()
        best_X_hat = torch.concat((X_hat.detach(), torch.ones(1)))
    for _ in range(attempts - 1):
        X_hat = torch.rand(d - 1)
        X_hat.requires_grad_(True)
        optimizer = torch.optim.SGD([X_hat], lr=0.01, momentum=0.9)
        for _ in range(iters):
            X_hat_aug = torch.concat((X_hat, torch.ones(1)))
            optimizer.zero_grad()
            loss = -agent_reward(u, phi, X_hat_aug, X_hat_aug, temp)
            loss.backward()
            # print(loss)
            optimizer.step()
        for _ in range(iters):
            X_hat_aug = torch.concat((X_hat, torch.ones(1)))
            optimizer.zero_grad()
            loss = -agent_reward(u, phi, X, X_hat_aug, temp)
            loss.backward()
            # print(loss)
            optimizer.step()
        if loss < -best_reward:
            best_reward = -loss.detach()
            best_X_hat = torch.concat((X_hat.detach(), torch.ones(1)))
    return best_X_hat


def dm_reward(xWs, x_hats, phi, temp=0.02):
    actions = classify(x=x_hats, phi=phi, temp=temp)
    if len(x_hats.shape) > 1:
        return torch.mean(torch.diag(xWs @ actions))
    else:
        return xWs @ actions

def phi_regularization(phi):
    return torch.norm(torch.norm(phi, dim=1) - 1)

def gr_phi(Xhats, xWs, u, prev_phi=None, 
           temp=0.02, attempts=5, iters=300): # no additional penalty for finding x?
    n, d = Xhats.shape
    m = u.shape[0]
    if prev_phi is None:
        phi = torch.rand((m, d))
        phi.requires_grad_(True)
    else:
        phi = torch.clone(prev_phi)
        phi.requires_grad_(True)
    best_phi = phi.clone()
    best_reward = dm_reward(xWs=xWs, x_hats=Xhats, phi=phi, temp=temp)
    for _ in range(attempts):
        optimizer = torch.optim.Adam([phi], lr=0.0001)
        for _ in tqdm(range(iters), position=0, leave=True):
            optimizer.zero_grad()
            loss = - dm_reward(xWs=xWs, x_hats=Xhats, phi=phi, temp=temp) # + phi_regularization(phi)
            loss.backward()
            optimizer.step()
            # print(loss)
        if loss < -best_reward:
            best_reward = -loss.detach()
            best_phi = phi.detach()
        phi = torch.rand((m, d))
        phi.requires_grad_(True)
    return best_phi

def gr_xW(reward, x_hat, phi, temp=0.02, attempts=5, iters=1_000):
    m, d = phi.shape
    xW = torch.rand(m)
    xW.requires_grad_(True)
    best_xW = xW.clone()
    best_loss = torch.norm(reward - dm_reward(xWs=xW, x_hats=x_hat, phi=phi, temp=temp))
    for _ in range(attempts):
        optimizer = torch.optim.Adam([xW], lr=0.0001)
        for _ in range(iters):
            optimizer.zero_grad()
            loss = torch.norm(reward - dm_reward(xWs=xW, x_hats=x_hat, phi=phi, temp=temp))
            loss.backward()
            optimizer.step()
        if loss < best_loss:
            best_loss = loss.detach()
            best_xW = xW.detach()
        xW = torch.rand(m)
        xW.requires_grad_(True)
    return best_xW

