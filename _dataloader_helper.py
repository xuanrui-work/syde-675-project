import torch
import torchvision
import torchvision.transforms as transforms

class LoaderHelperBase:
    def __init__(self, batch_size=128, val_split=0.2):
        self.batch_size = batch_size
        self.val_split = val_split

        self.src_train = self.src_val = self.src_test = None
        self.tgt_train = self.tgt_val = self.tgt_test = None
    
    def get_src_loaders(self):
        src_train_loader = torch.utils.data.DataLoader(
            self.src_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        src_val_loader = torch.utils.data.DataLoader(
            self.src_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        src_test_loader = torch.utils.data.DataLoader(
            self.src_test,
            batch_size=self.batch_size,
            shuffle=False
        )
        return (src_train_loader, src_val_loader, src_test_loader)
    
    def get_tgt_loaders(self):
        tgt_train_loader = torch.utils.data.DataLoader(
            self.tgt_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        tgt_val_loader = torch.utils.data.DataLoader(
            self.tgt_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        tgt_test_loader = torch.utils.data.DataLoader(
            self.tgt_test,
            batch_size=self.batch_size,
            shuffle=False
        )
        return (tgt_train_loader, tgt_val_loader, tgt_test_loader)

class MNIST2USPS(LoaderHelperBase):
    def __init__(
        self,
        reverse=False,
        image_size=(32, 32),
        mnist_save_dir='./dataset/mnist', usps_save_dir='./dataset/usps',
        **kwargs
    ):
        super().__init__(**kwargs)

        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        usps_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        mnist_train = torchvision.datasets.MNIST(
            root=mnist_save_dir,
            train=True,
            transform=mnist_transform,
            download=True
        )
        mnist_test = torchvision.datasets.MNIST(
            root=mnist_save_dir,
            train=False,
            transform=mnist_transform,
            download=True
        )
        mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [1-self.val_split, self.val_split])

        usps_train = torchvision.datasets.USPS(
            root=usps_save_dir,
            train=True,
            transform=usps_transform,
            download=True
        )
        usps_test = torchvision.datasets.USPS(
            root=usps_save_dir,
            train=False,
            transform=usps_transform,
            download=True
        )
        usps_train, usps_val = torch.utils.data.random_split(usps_train, [1-self.val_split, self.val_split])

        if not reverse:
            self.src_train, self.src_val, self.src_test = mnist_train, mnist_val, mnist_test
            self.tgt_train, self.tgt_val, self.tgt_test = usps_train, usps_val, usps_test
        else:
            self.src_train, self.src_val, self.src_test = usps_train, usps_val, usps_test
            self.tgt_train, self.tgt_val, self.tgt_test = mnist_train, mnist_val, mnist_test
