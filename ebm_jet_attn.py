#!/usr/bin/env python

from ebm_preamble import *

groomer = None
dump_number_of_nodes = False 

FLAGS = {
        'max_len': 10000,
        'new_sample_rate': 0.05,
        'singlestep': False, # for KL improved training, only back-prop through the last LD step
        'MH': True,  # Metropolis-Hastings step for HMC
        'val_steps': 128,
        'scaled': False # Input feature scaling
        }

def kl_divergence():
    kl = F.kl_div(a, b)

def random_sample(n_sample, n_consti):
    if FLAGS['scaled']:
        rand_logpt = torch.normal(0.0, 1.0, (n_sample, n_consti, 1))
        rand_eta = torch.normal(0.0, 1.0, (n_sample, n_consti, 1)) 
        rand_phi = torch.normal(0.0, 1.0, (n_sample, n_consti, 1))
    else:
        rand_logpt = torch.normal(2.0, 1.0, (n_sample, n_consti, 1))
        rand_eta = torch.normal(0.0, 0.1, (n_sample, n_consti, 1))
        rand_phi = torch.normal(0.0, 0.2, (n_sample, n_consti, 1))
    
    rand_jets = torch.cat([rand_logpt, rand_eta, rand_phi], dim=-1)
    rand_jets = rand_jets.view(n_sample, n_consti*3)
    
    return rand_jets

class Sampler:

    def __init__(self, model, jet_shape, sample_size, max_len=FLAGS['max_len'], kl=False, hmc=False, epsilon=0.005, return_grad=False):
        super().__init__()
        self.model = model
        self.jet_shape = jet_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.kl = kl
        self.hmc = hmc
        self.epsilon = epsilon
        self.return_grad = return_grad
        self.examples = [random_sample(1, jet_shape[0] // 3) for _ in range(sample_size)]

    def sample_new_exmps(self, steps=60, step_size=10):
        n_new = np.random.binomial(self.sample_size, FLAGS['new_sample_rate'])
        n_consti = self.jet_shape[0] // 3
        rand_jets = random_sample(n_new, n_consti)
                
        old_jets = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
        inp_jets = torch.cat([rand_jets, old_jets], dim=0).detach().to(device)

        if self.hmc:
            inp_jets, x_grad, v = Sampler.generate_samples(self.model, inp_jets, steps=steps, step_size=step_size, hmc=True)
            self.examples = list(inp_jets.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
            self.examples = self.examples[:self.max_len]
            return inp_jets, x_grad, v
        else:
            inp_jets, inp_jets_kl, grad_norm = Sampler.generate_samples(self.model, inp_jets, steps=steps, step_size=step_size, kl=self.kl, epsilon=self.epsilon, return_grad=self.return_grad)
            self.examples = list(inp_jets.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
            self.examples = self.examples[:self.max_len]
            return inp_jets, inp_jets_kl, grad_norm
        
    @staticmethod
    def generate_samples(model, inp_jets, steps=60, step_size=10, return_jet_per_step=False, return_grad=False, kl=False, hmc=False, epsilon=0.005):
        if hmc:
            if return_jet_per_step:
                im_neg, im_samples, x_grad, v = gen_hmc_samples(model, inp_jets, steps, step_size, sample=True)
                return im_samples, v
            else:
                im_neg, x_grad, v = gen_hmc_samples(model, inp_jets, steps, step_size)
                return im_neg, x_grad, v
        else:
            is_training = model.training
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
                
            inp_jets.requires_grad = True

            had_gradients_enabled = torch.is_grad_enabled()
            torch.set_grad_enabled(True)

            noise = torch.randn(inp_jets.shape, device=inp_jets.device)

            grad_norm = 0.0
            jets_per_step = []

            for i in range(steps):
                if i == steps - 1:
                    inp_jets_orig = inp_jets

                noise.normal_(0, epsilon)
                inp_jets.data.add_(noise.data)

                out_jets = - model(inp_jets.float())
                
                if FLAGS['singlestep']:
                    out_jets.sum().backward()
                    
                    inp_jets.data.add_(-step_size * inp_jets.grad.data)
                    
                    grad_norm += inp_jets.grad.data.norm(dim=1)
                    
                    inp_jets.grad.detach_()
                    inp_jets.grad.zero_()
                else:
                    if kl:
                        x_grad = torch.autograd.grad([out_jets.sum()], [inp_jets], create_graph=True)[0]
                    else:
                        x_grad = torch.autograd.grad([out_jets.sum()], [inp_jets])[0]   
                    inp_jets = inp_jets - step_size * x_grad
                    
                    grad_norm += x_grad.norm(dim=1)

                if return_jet_per_step:
                    jets_per_step.append(inp_jets.clone().detach())

                if i == steps - 1:
                    if kl:
                        inp_jets_kl = inp_jets_orig
                        energy = - model.forward(inp_jets_kl)
                        x_grad = torch.autograd.grad([energy.sum()], [inp_jets_kl], create_graph=True)[0]
                        inp_jets_kl = inp_jets_kl - step_size * x_grad
                    else:
                        inp_jets_kl = torch.zeros_like(inp_jets)  
                
            inp_jets = inp_jets.detach()
            
            for p in model.parameters():
                p.requires_grad = True
            model.train(is_training)

            torch.set_grad_enabled(had_gradients_enabled)

            if return_grad:
                grad_norm = grad_norm / steps
            else:
                grad_norm = 0.0

            if return_jet_per_step:
                return torch.stack(jets_per_step, dim=0), grad_norm
            else:
                return inp_jets, inp_jets_kl, grad_norm

    
class DeepEnergyModel(pl.LightningModule):

    def __init__(self, jet_shape, batch_size, steps=60, step_size=10, kl=False, repel_im=False, hmc=False, epsilon=0.005, alpha=0.1, lr=1e-4, beta1=0.0, **net_args):
        super().__init__()
        self.save_hyperparameters()
        self.save_hyperparameters("step_size")
        
        self.jet_shape = jet_shape
        self.batch_size = batch_size
        self.hmc = hmc
        self.epsilon = epsilon

        self.net = Transformer(**net_args)
        self.sampler = Sampler(self.net, jet_shape=jet_shape, sample_size=batch_size, kl=kl, hmc=hmc, epsilon=epsilon, return_grad=True)

    def forward(self, x):
        z = self.net(x)
        return z

    def configure_optimizers(self): 
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        real_jets = batch
        small_noise = torch.randn_like(real_jets) * self.epsilon
        real_jets.add_(small_noise)
        
        if self.hparams.hmc:
            fake_jets, x_grad, v = self.sampler.sample_new_exmps(steps=self.hparams.steps, step_size=self.hparams.step_size)
        else:
            fake_jets, fake_jets_kl, v = self.sampler.sample_new_exmps(steps=self.hparams.steps, step_size=self.hparams.step_size)

        inp_jets = torch.cat([real_jets, fake_jets], dim=0)
        real_out, fake_out = self.net(inp_jets.float()).chunk(2, dim=0)

        reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
        cdiv_loss =  fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        if self.hparams.hmc:
            v_flat = v.view(v.size(0), -1)
            x_grad_flat = x_grad.view(x_grad.size(0), -1)
            dot_product = F.normalize(v_flat, dim=1) * F.normalize(x_grad_flat, dim=1)
            loss_hmc = torch.abs(dot_product.sum(dim=1)).mean()
            loss = loss + 0.1 * loss_hmc
            v = v.norm(dim=1)
        else:
            loss_hmc = torch.zeros(1)
            
        if self.hparams.kl:
            self.net.requires_grad_(False)
            loss_kl = - self.net.forward(fake_jets_kl)
            self.net.requires_grad_(True)
            loss = loss +  loss_kl.mean()

            if self.hparams.repel_im:
                start = datetime.datetime.now()
                
                bs = fake_jets_kl.size(0)

                im_flat = fake_jets_kl.view(bs, -1)

                if len(self.sampler.examples) > 1000:
                    compare_batch = torch.cat(random.choices(self.sampler.examples, k=100), dim=0)
                    
                    compare_batch = torch.Tensor(compare_batch).cuda(0)
                    compare_flat = compare_batch.view(100, -1)

                    dist_matrix = torch.norm(im_flat[:, None, :] - compare_flat[None, :, :], p=2, dim=-1)
                    loss_repel = torch.log(dist_matrix.min(dim=1)[0]).mean()
                    
                    loss = loss - 0.3 * loss_repel
                else:
                    loss_repel = torch.zeros(1)

                end = datetime.datetime.now()
            else:
                loss_repel = torch.zeros(1)   
        else:
            loss_kl = torch.zeros(1)
        
        self.log('loss', loss)
        self.log('loss_reg', reg_loss)
        self.log('loss_cd', cdiv_loss, prog_bar=True)
        self.log('loss_kl', loss_kl.mean(), prog_bar=True)
        self.log('loss_repel', loss_repel)
        self.log('loss_hmc', loss_hmc.mean(), prog_bar=True)
        self.log('nenergy_real', real_out.mean())
        self.log('nenergy_sample', fake_out.mean())
        self.log('train_average_v', v.mean())
        
        return loss

    def validation_step(self, batch, batch_idx):
        jets, labels = batch
        batch_size = len(labels)
        
        qcd = jets[labels==0]
        signal = jets[labels==1]
        
        jets = torch.cat([qcd, signal], dim=0)
        qcd_out, signal_out = self.net(jets.float()).chunk(2, dim=0)
        cdiv_top = signal_out.mean() - qcd_out.mean()
        
        y_pred = np.concatenate((-qcd_out.cpu(), -signal_out.cpu()))
        y_true = np.concatenate((np.zeros_like(qcd_out.cpu()), np.ones_like(signal_out.cpu())))
        auc = roc_auc_score(y_true, y_pred)    
        
        n_consti = self.jet_shape[0] // 3
        random_jets = random_sample(batch_size, n_consti).to(device)
        
        random_out = self.net(random_jets.float())
        cdiv_random = random_out.mean() - qcd_out.mean()
        
        self.log('val_cd_top', cdiv_top, prog_bar=True)
        self.log('val_cd_random', cdiv_random, prog_bar=True)
        self.log('val_nenergy_top', signal_out.mean())
        self.log('val_nenergy_qcd', qcd_out.mean())
        self.log('val_auc_top', auc, prog_bar=True)
        self.log('hp_metric', auc)
        
        self.eval()
        n_consti = self.hparams["jet_shape"][0] // 3
        init_samples = random_sample(batch_size, n_consti).to(self.device)
        
        torch.set_grad_enabled(True)
        if self.hparams.hmc:
            gen_samples, x_grad, v = self.sampler.generate_samples(self.net, init_samples, steps=FLAGS['val_steps'], step_size=self.hparams.step_size, hmc=True)

        else:
            gen_samples, _, _ =  self.sampler.generate_samples(self.net, init_samples, steps=FLAGS['val_steps'], step_size=self.hparams.step_size, kl=False, hmc=False) # turn off KL for saving memory and faster generation
        torch.set_grad_enabled(False)
        
        self.train()
                
        gen_out = self.net(gen_samples)
        cdiv_gen = gen_out.mean() - qcd_out.mean()
        self.log('val_cd_gen', cdiv_gen, prog_bar=True)
        
        gen_samples = jet_from_ptetaphi(gen_samples.cpu(), scaled=FLAGS['scaled'])
        qcd = jet_from_ptetaphi(qcd.cpu(), scaled=FLAGS['scaled'])

        gen_pts = list(map(jet_pt, gen_samples))
        gen_pts = torch.tensor(gen_pts)

        real_pts = list(map(jet_pt, qcd))
        real_pts = torch.tensor(real_pts)
        
        prob_gen_pts = torch.histc(gen_pts, bins=50, min=200, max=1000)
        prob_gen_pts = prob_gen_pts / prob_gen_pts.sum()
        prob_real_pts = torch.histc(real_pts, bins=50, min=200, max=1000)
        prob_real_pts = prob_real_pts / prob_real_pts.sum()

        prob_mean = (prob_real_pts + prob_gen_pts) / 2.0
        kl_pt = (F.kl_div(torch.log(prob_mean), prob_real_pts) + F.kl_div(torch.log(prob_mean), prob_gen_pts)) / 2.0
        
        self.log('val_KL_pt', kl_pt, prog_bar=True)
        
        gen_ms = list(map(jet_mass, gen_samples))
        gen_ms = torch.tensor(gen_ms)

        real_ms = list(map(jet_mass, qcd))
        real_ms = torch.tensor(real_ms)
        
        prob_gen_ms = torch.histc(gen_ms, bins=50, min=0, max=500)
        prob_gen_ms = prob_gen_ms / prob_gen_ms.sum()
        prob_real_ms = torch.histc(real_ms, bins=50, min=0, max=500)
        prob_real_ms = prob_real_ms / prob_real_ms.sum()

        prob_mean = (prob_real_ms + prob_gen_ms) / 2.0
        kl_m = (F.kl_div(torch.log(prob_mean), prob_real_ms) + F.kl_div(torch.log(prob_mean), prob_gen_ms)) / 2.0
        
        self.log('val_KL_m', kl_m, prog_bar=True)

        self.log('val_KL_avg', (kl_pt + kl_m)/2.0, prog_bar=True)
        
    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict   

def train_model(train_loader, val_loader, model_name, epochs, **kwargs):
    
    default_path =  os.path.join(CHECKPOINT_PATH, "attn", datetime.datetime.now().strftime("%m%d-%H%M%S")+"_"+model_name)
    tb_logger = TensorBoardLogger(default_path, name=None, version=None)
    
    trainer = pl.Trainer(logger=tb_logger,
                         gpus=-1 if str(device).startswith("cuda") else 0, # set gpus=-1 to use all available gpus
                         #accelerator="ddp",
                         max_epochs=epochs,
                         gradient_clip_val=0.1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_KL_avg'),
                                    PeriodicCheckpoint(interval=len(train_loader), save_weights_only=True),
                                    LitProgressBar(),
                                    LearningRateMonitor("epoch")
                                   ])
    
    #pl.seed_everything(42)
    model = DeepEnergyModel(**kwargs)
    
    # Multiple GPUs
    #model= nn.DataParallel(model)
    #model.to(device)
    
    trainer.fit(model, train_loader, val_loader)
    model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    return model           

def eval_ood(model, train, test):
    from sklearn.metrics import roc_auc_score
    with torch.no_grad():
        model.to(device)
        train = torch.Tensor(train).to(device)
        train_out = model.net(train)
    
        test = torch.Tensor(test).to(device)
        test_out = model.net(test)
        y_true = torch.cat((torch.zeros_like(train), torch.ones_like(test)))
        y_pred = torch.cat((-train_out, -test_out))
        auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())        
        print(f"Test AUC: {auc:4.3f}")
        return auc

def main():
    parser = argparse.ArgumentParser()
    
    # Inputs
    parser.add_argument('--input_dim', type=int, default=160)
    parser.add_argument('--input_scaler', action='store_true')

    # MCMC 
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--step_size', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.005)
    parser.add_argument('--kl', action='store_true') # Add KL term for improved training or not
    parser.add_argument('--repel_im', action='store_true')
    parser.add_argument('--hmc', action='store_true')

    # Training
    parser.add_argument('--n_train', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--topref', action='store_true')
    
    # Saving models
    parser.add_argument('--mode', default="train")
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--tag', default=None)

    args = parser.parse_args()
    e_func = "attnv3"
    
    train_set, scaler = load_attn_train(n_train=args.n_train, input_dim=args.input_dim, scale=args.input_scaler, topref=args.topref)
    
    test_fn = os.environ['VAE_DIR'] + 'h3_m174_h80_01_preprocessed.h5'
    test_set = load_attn_test(scaler, test_fn, input_dim=args.input_dim, scale=args.input_scaler)
    
    val_X, val_y = load_attn_val(scaler, n_val=10000, input_dim=args.input_dim, scale=args.input_scaler, topref=args.topref)
        
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = data.DataLoader([[val_X[i], val_y[i]] for i in range(len(val_X))], batch_size=args.batch_size, shuffle=False,  drop_last=True, num_workers=2, pin_memory=True)
    
    test_loader  = data.DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    if args.mode == "train":
            
        if args.model_name is None:
            model_path = 'models/{}_n{}k_d{}_stp{}_ss{}_eps{}_bs{}_e{}_l{}'.format(e_func, int(args.n_train / 1000.), args.input_dim, args.steps, args.step_size, args.epsilon, args.batch_size, args.epochs, args.lr)
            model_path += "_kl" if args.kl else "" 
            model_path += "_hmc" if args.hmc else ""
            model_path += "_scale" if args.input_scaler else ""
            model_path += "_{}".format(args.tag) if args.tag else ""

        else:
            model_path = "models/" + args.model_name
        
        model = train_model(train_loader,
                            val_loader,
                            os.path.basename(model_path),
                            epochs=args.epochs,
                            jet_shape=(args.input_dim // 4 * 3,),
                            batch_size=train_loader.batch_size,
                            lr=args.lr,
                            beta1=0.0,
                            steps=args.steps,
                            step_size=args.step_size,
                            num_layers=8,
                            d_model=128,
                            num_heads=16,
                            dff=1024,
                            rate=0.1,
                            kl=args.kl,
                            repel_im=args.repel_im,
                            hmc=args.hmc,
                            epsilon=args.epsilon
                           )
        
        torch.save(model.state_dict(), model_path)

        test_data = data(input_dim=160)
        _, bkg, sig1, sig2 = test_data.load()
        
        eval_ood(model, bkg, sig1)
        eval_ood(model, bkg, sig2)
                
    elif args.mode == "test":
        assert model_name != None
        model = torch.load(model_name)

        test_data = data(input_dim=160)
        _, bkg, sig1, sig2 = test_data.load()
        
        eval_ood(model, bkg, sig1)
        eval_ood(model, bkg, sig2)


if __name__ == "__main__":
    main()
