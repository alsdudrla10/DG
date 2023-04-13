import click
import os
import classifier_lib
import torch
import numpy as np
import glob
import dnnlib
import torchvision.transforms as transforms
import torch.utils.data as data

def npz_concat(filenames):
    for file in filenames:
        samples = np.load(file)['samples']
        try:
            data = np.concatenate((data, samples))
        except:
            data = samples
    return data

def npz_concat_cond(filenames):
    for file in filenames:
        samples = np.load(file)['samples']
        label = np.load(file)['label']
        try:
            data = np.concatenate((data, samples))
            data_label = np.concatenate((data_label, label))
        except:
            data = samples
            data_label = label
    return data, data_label

class BasicDataset(data.Dataset):
  def __init__(self, x_np, y_np, transform=transforms.ToTensor()):
    super(BasicDataset, self).__init__()

    self.x = x_np
    self.y = y_np
    self.transform = transform

  def __getitem__(self, index):
    return self.transform(self.x[index]), self.y[index]

  def __len__(self):
    return len(self.x)

class BasicDatasetCond(data.Dataset):
  def __init__(self, x_np, y_np, cond_np, transform=transforms.ToTensor()):
    super(BasicDatasetCond, self).__init__()

    self.x = x_np
    self.y = y_np
    self.cond = cond_np
    self.transform = transform

  def __getitem__(self, index):
    return self.transform(self.x[index]), self.y[index], self.cond[index]

  def __len__(self):
    return len(self.x)

@click.command()
@click.option('--savedir',                     help='Save directory',          metavar='PATH',    type=str, required=True,     default="/checkpoints/discriminator/cifar_uncond")
@click.option('--gendir',                      help='Fake sample directory',   metavar='PATH',    type=str, required=True,     default="/samples/cifar_uncond_vanilla")
@click.option('--datadir',                     help='Real sample directory',   metavar='PATH',    type=str, required=True,     default="/data/true_data.npz")
@click.option('--img_resolution',              help='Image resolution',        metavar='INT',     type=click.IntRange(min=1),  default=32)
@click.option('--cond',                        help='Is it conditional?',      metavar='INT',     type=click.IntRange(min=0),  default=0)

@click.option('--pretrained_classifier_ckpt',  help='Path of classifier',      metavar='STR',     type=str,                    default='/checkpoints/ADM_classifier/32x32_classifier.pt')
@click.option('--num_data',                    help='Num samples',             metavar='INT',     type=click.IntRange(min=1),  default=50000)
@click.option('--batch_size',                  help='Num samples',             metavar='INT',     type=click.IntRange(min=1),  default=128)
@click.option('--epoch',                       help='Num samples',             metavar='INT',     type=click.IntRange(min=1),  default=50)
@click.option('--lr',                          help='Learning rate',           metavar='FLOAT',   type=click.FloatRange(min=0),default=3e-4)
@click.option('--device',                      help='Device',                  metavar='STR',     type=str,                    default='cuda:0')

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    gendir = os.getcwd() + opts.gendir
    savedir = os.getcwd() + opts.savedir
    datadir = os.getcwd() + opts.datadir
    os.makedirs(savedir, exist_ok=True)

    ## Prepare real data
    if not opts.cond:
        real_data = np.load(datadir)['arr_0']
    else:
        real_data  = np.load(datadir)['samples']
        real_label = np.load(datadir)['label']
        real_label = np.eye(10)[real_label]

    ## Prepare fake data
    if not opts.cond:
        if not os.path.exists(os.path.join(gendir, 'gen_data_for_discriminator_training.npz')):
            filenames = np.sort(glob.glob(os.path.join(gendir, 'sample*.npz')))
            gen_data = npz_concat(filenames)
            np.savez_compressed(os.path.join(gendir, 'gen_data_for_discriminator_training.npz'), samples=gen_data)
        else:
            gen_data = np.load(os.path.join(gendir, 'gen_data_for_discriminator_training.npz'))['samples']
    else:
        if not os.path.exists(os.path.join(gendir, 'gen_data_for_discriminator_training.npz')):
            filenames = np.sort(glob.glob(os.path.join(gendir, 'sample*.npz')))
            gen_data, gen_label = npz_concat_cond(filenames)
            np.savez_compressed(os.path.join(gendir, 'gen_data_for_discriminator_training.npz'), samples=gen_data, label=gen_label)
        else:
            gen_data = np.load(os.path.join(gendir, 'gen_data_for_discriminator_training.npz'))['samples']
            gen_label = np.load(os.path.join(gendir, 'gen_data_for_discriminator_training.npz'))['label']
            gen_label = gen_label[:opts.num_data]

    ## Combine the fake / real
    real_data = real_data[:opts.num_data]
    gen_data = gen_data[:opts.num_data]
    train_data = np.concatenate((real_data, gen_data))
    train_label = torch.zeros(train_data.shape[0])
    train_label[:real_data.shape[0]] = 1.
    transform = transforms.Compose([transforms.ToTensor()])
    if not opts.cond:
        train_dataset = BasicDataset(train_data, train_label, transform)
    else:

        condition_label = np.concatenate((real_label, gen_label))
        train_dataset = BasicDatasetCond(train_data, train_label, condition_label, transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=0, shuffle=True, drop_last=True)

    ## Extractor & Disciminator
    pretrained_classifier = classifier_lib.load_classifier(opts.pretrained_classifier_ckpt, opts.img_resolution, opts.device, eval=False)
    discriminator = classifier_lib.load_discriminator(None, opts.device, opts.cond, eval=False)

    ## Prepare training
    vpsde = classifier_lib.vpsde()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, weight_decay=1e-7)
    loss = torch.nn.BCELoss()
    scaler = lambda x: 2. * x - 1.

    ## Training
    for i in range(opts.epoch):
        outs = []
        cors = []
        num_data = 0
        for data in train_loader:
            optimizer.zero_grad()
            if not opts.cond:
                inputs, labels = data
            else:
                inputs, labels, cond = data
                cond = cond.to(opts.device)
            inputs = inputs.to(opts.device)
            labels = labels.to(opts.device)
            inputs = scaler(inputs)

            ## Data perturbation
            t, _ = vpsde.get_diffusion_time(inputs.shape[0], inputs.device)
            mean, std = vpsde.marginal_prob(t)
            z = torch.randn_like(inputs)
            perturbed_inputs = mean[:, None, None, None] * inputs + std[:, None, None, None] * z

            ## Forward
            with torch.no_grad():
                pretrained_feature = pretrained_classifier(perturbed_inputs, timesteps=t, feature=True)
            if not opts.cond:
                label_prediction = discriminator(pretrained_feature, t, sigmoid=True).view(-1)
            else:
                label_prediction = discriminator(pretrained_feature, t, sigmoid=True, condition=cond).view(-1)
            ## Backward
            out = loss(label_prediction, labels)
            out.backward()
            optimizer.step()

            ## Report
            cor = ((label_prediction > 0.5).float() == labels).float().mean()
            outs.append(out.item())
            cors.append(cor.item())
            num_data += inputs.shape[0]
            print(f"{i}-th epoch BCE loss: {np.mean(outs)}, correction rate: {np.mean(cors)}")

        ## Save
        torch.save(discriminator.state_dict(), savedir + f"/discriminator_{i+1}.pt")

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------