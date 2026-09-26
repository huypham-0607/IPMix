"""Corruption-error (mCE) eval for the IPMIX reproduction."""
import argparse, os, csv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms, datasets

CORRUPTIONS = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur',
    'glass_blur','motion_blur','zoom_blur','snow','frost','fog','brightness',
    'contrast','elastic_transform','pixelate','jpeg_compression']

def _wi(m):
    if isinstance(m,(nn.Linear,nn.Conv2d)): init.kaiming_normal_(m.weight)
class Lambda(nn.Module):
    def __init__(s,l): super().__init__(); s.l=l
    def forward(s,x): return s.l(x)
class BasicBlock(nn.Module):
    expansion=1
    def __init__(s,inp,planes,stride=1,option='A'):
        super().__init__()
        s.conv1=nn.Conv2d(inp,planes,3,stride,1,bias=False); s.bn1=nn.BatchNorm2d(planes)
        s.conv2=nn.Conv2d(planes,planes,3,1,1,bias=False);   s.bn2=nn.BatchNorm2d(planes)
        s.shortcut=nn.Sequential()
        if stride!=1 or inp!=planes:
            if option=='A':
                s.shortcut=Lambda(lambda x: F.pad(x[:,:,::2,::2],(0,0,0,0,planes//4,planes//4),"constant",0))
            else:
                s.shortcut=nn.Sequential(nn.Conv2d(inp,planes,1,stride,bias=False),nn.BatchNorm2d(planes))
    def forward(s,x):
        out=F.relu(s.bn1(s.conv1(x))); out=s.bn2(s.conv2(out)); out+=s.shortcut(x); return F.relu(out)
class CifarResNet(nn.Module):
    def __init__(s,block,nb,num_classes=10):
        super().__init__(); s.in_planes=16
        s.conv1=nn.Conv2d(3,16,3,1,1,bias=False); s.bn1=nn.BatchNorm2d(16)
        s.layer1=s._ml(block,16,nb[0],1); s.layer2=s._ml(block,32,nb[1],2); s.layer3=s._ml(block,64,nb[2],2)
        s.linear=nn.Linear(64,num_classes); s.apply(_wi)
    def _ml(s,block,planes,n,stride):
        layers=[]
        for st in [stride]+[1]*(n-1):
            layers.append(block(s.in_planes,planes,st)); s.in_planes=planes*block.expansion
        return nn.Sequential(*layers)
    def forward(s,x):
        out=F.relu(s.bn1(s.conv1(x))); out=s.layer1(out); out=s.layer2(out); out=s.layer3(out)
        out=F.avg_pool2d(out,out.size()[3]).view(out.size(0),-1); return s.linear(out)
def resnet20(num_classes=10): return CifarResNet(BasicBlock,[3,3,3],num_classes)

def build(arch,nc,layers,widen):
    if arch=='resnet20': return resnet20(nc)
    if arch=='resnet18':
        from models.ResNet.resnet import resnet18; return resnet18(num_classes=nc)
    if arch=='resnext29':
        from models.ResNeXt_DenseNet.models.resnext import resnext29; return resnext29(num_classes=nc)
    if arch=='densenet':
        from models.ResNeXt_DenseNet.models.densenet import densenet; return densenet(num_classes=nc)
    if arch=='wrn':
        from models.WideResNet_pytorch.wideresnet import WideResNet; return WideResNet(layers,nc,widen,0.3)
    raise ValueError(arch)

def load_ckpt(model,path,dev):
    ck=torch.load(path,map_location=dev,weights_only=False)
    if isinstance(ck,dict) and 'state_dict' in ck:
        sd=ck['state_dict']
        if 'best_acc' in ck: print('  ckpt best_acc:',ck['best_acc'])
        if 'epoch' in ck: print('  ckpt epoch:',ck['epoch'])
    else: sd=ck
    sd={k[7:] if k.startswith('module.') else k:v for k,v in sd.items()}
    miss,unexp=model.load_state_dict(sd,strict=False)
    if miss: print('  [warn] missing:',len(miss),miss[:4])
    if unexp: print('  [warn] unexpected:',len(unexp),unexp[:4])
    return model

class NumpyDS(torch.utils.data.Dataset):
    def __init__(s,data,tgt,tf): s.data,s.tgt,s.tf=data,tgt,tf
    def __len__(s): return len(s.data)
    def __getitem__(s,i): return s.tf(Image.fromarray(s.data[i])), int(s.tgt[i])

@torch.no_grad()
def acc(net,loader,dev):
    net.eval(); c=t=0
    for x,y in loader:
        x,y=x.to(dev),y.to(dev); p=net(x).argmax(1); c+=(p==y).sum().item(); t+=y.size(0)
    return c/t

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arch',required=True,choices=['resnet20','resnet18','resnext29','wrn','densenet'])
    p.add_argument('--dataset',required=True,choices=['cifar10','cifar100'])
    p.add_argument('--ckpt',required=True); p.add_argument('--cifar-c-dir',required=True)
    p.add_argument('--eval-batch-size',type=int,default=1000); p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--layers',type=int,default=40); p.add_argument('--widen-factor',type=int,default=4)
    p.add_argument('--check-clean',action='store_true'); p.add_argument('--clean-data',default='./data')
    p.add_argument('--out',default=None); a=p.parse_args()
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    nc=100 if a.dataset=='cifar100' else 10
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)])
    print(f'== {a.arch} / {a.dataset} on {dev} ==')
    net=build(a.arch,nc,a.layers,a.widen_factor).to(dev); load_ckpt(net,a.ckpt,dev)
    if a.check_clean:
        DS=datasets.CIFAR100 if a.dataset=='cifar100' else datasets.CIFAR10
        cl=DS(a.clean_data,train=False,transform=tf,download=True)
        ld=torch.utils.data.DataLoader(cl,batch_size=a.eval_batch_size,shuffle=False,num_workers=a.num_workers)
        print(f'CLEAN test error: {100-100*acc(net,ld,dev):.2f}%  (compare to training log)')
    labels=np.load(os.path.join(a.cifar_c_dir,'labels.npy')); rows=[]; accs=[]
    for c in CORRUPTIONS:
        data=np.load(os.path.join(a.cifar_c_dir,c+'.npy'))
        ld=torch.utils.data.DataLoader(NumpyDS(data,labels,tf),batch_size=a.eval_batch_size,shuffle=False,num_workers=a.num_workers)
        ac=acc(net,ld,dev); accs.append(ac); e=100-100*ac; rows.append((c,round(e,3)))
        print(f'{c:20s} error {e:6.3f}%')
    mce=100-100*float(np.mean(accs))
    print(f'\nMean Corruption Error ({a.arch}, {a.dataset}): {mce:.3f}%')
    if a.out:
        with open(a.out,'w',newline='') as f:
            w=csv.writer(f); w.writerow(['corruption','error_pct']); w.writerows(rows); w.writerow(['mCE',round(mce,3)])
        print('wrote',a.out)

if __name__=='__main__': main()




