import torch

def energy_wrapper(nenergy):
    '''
    Wrapper to facilitate flexible energy function sign
    '''
    energy = - nenergy
    return energy

def hamiltonian(x, v, model):
    energy = 0.5 * torch.pow(v, 2).sum(dim=1) + energy_wrapper(model.forward(x).squeeze())
    return energy

def leapfrog_step(x, v, model, step_size, num_steps, sample=False, mh=FLAGS['MH']):
    x0 = x
    v0 = v
    
    x.requires_grad_(requires_grad=True)
    energy = energy_wrapper(model.forward(x))
    x_grad = torch.autograd.grad([energy.sum()], [x])[0]
    v = v - 0.5 * step_size * x_grad
    x_negs = []
    
    for i in range(num_steps):
        x.requires_grad_(requires_grad=True)
        energy = energy_wrapper(model.forward(x))

        if i == num_steps - 1:
            x_grad = torch.autograd.grad([energy.sum()], [x], create_graph=True)[0]
            v = v - step_size * x_grad
            x = x + step_size * v
            v = v.detach()
        else:
            x_grad = torch.autograd.grad([energy.sum()], [x])[0]
            v = v - step_size * x_grad
            x = x + step_size * v
            x = x.detach()
            v = v.detach()

        if sample:
            x_negs.append(x)

        if i % 10 == 0:
            print(i, hamiltonian(torch.sigmoid(x), v, model).mean(), torch.abs(v).mean(), torch.abs(x_grad).mean())

    if mh:
        accept = MH_accept(model, x0, x)
        x = accept * x + (1 - accept) * x0
        v = accept * v + (1 - accept) * v0
        x_grad = accept * x_grad
    if sample:
        return x, torch.stack(x_negs, dim=0), v, x_grad
    else:
        return x, v, x_grad

def gen_hmc_samples(model, x_neg, num_steps, step_size, sample=False):

    v = 0.001 * torch.randn_like(x_neg)

    if sample:
        x_neg, x_negs, v, x_grad = leapfrog_step(x_neg, v, model, step_size, num_steps, sample=sample)
        return x_neg, x_negs, x_grad, v
    else:
        x_neg, v, x_grad = leapfrog_step(x_neg, v, model, step_size, num_steps, sample=sample)
        return x_neg, x_grad, v 
    
def MH_accept(model, x0, x1):
    '''
    Add a Metropolis-Hastings step after HMC to move around the energy landscape
    '''
    energy0 = energy_wrapper(model.forward(x0))
    energy1 = energy_wrapper(model.forward(x1))
    likelihood_ratio = torch.exp(-energy1 + energy0)
    u = torch.rand_like(likelihood_ratio)
    accept = ((u - likelihood_ratio) < 0).float()
    return accept