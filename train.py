# all import statements remain unchanged
# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.


# The program train.py trains a base classifier with Gaussian data augmentation:

import argparse  # Importa il modulo per analizzare gli argomenti della riga di comando
import os  # Importa il modulo per le operazioni di sistema
import torch  # Importa il modulo PyTorch
from torch.nn import CrossEntropyLoss  # Importa la funzione di perdita Cross Entropy
from torch.utils.data import DataLoader  # Importa DataLoader per il caricamento dei dati
from datasets import get_dataset, DATASETS  # Importa funzioni e costanti per i dataset
from architectures import ARCHITECTURES, get_architecture  # Importa le architetture dei modelli
from torch.optim import SGD, Optimizer  # Importa l'ottimizzatore SGD
from torch.optim.lr_scheduler import StepLR  # Importa il scheduler per la riduzione del learning rate
import time  # Importa il modulo per gestire il tempo
import datetime  # Importa il modulo per gestire data e ora
from train_utils import AverageMeter, accuracy, init_logfile, log  # Importa utilità per il training
from robustbench import load_model
from robustbench.data import load_cifar10

# Definisce l'analizzatore degli argomenti della riga di comando
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("dataset", type=str, choices=DATASETS)  # Aggiunge l'argomento per il dataset
parser.add_argument("arch", type=str, choices=ARCHITECTURES)  # Aggiunge l'argomento per l'architettura del modello
parser.add_argument("outdir", type=str, help="folder to save model and training log)")  # Aggiunge l'argomento per la cartella di output
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
    )  # Aggiunge l'argomento per il numero di worker per il caricamento dei dati
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run" 
    )  # Aggiunge l'argomento per il numero di epoche
parser.add_argument(
    "--batch", default=256, type=int, metavar="N", help="batchsize (default: 256)"
    )  # Aggiunge l'argomento per la dimensione del batch
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    help="initial learning rate",
    dest="lr",
)  # Aggiunge l'argomento per il learning rate iniziale
parser.add_argument(
    "--lr_step_size",
    type=int,
    default=30,
    help="How often to decrease learning by gamma.",
)  # Aggiunge l'argomento per la frequenza di riduzione del learning rate
parser.add_argument(
    "--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule."
)  # Aggiunge l'argomento per il fattore di riduzione del learning rate
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")  # Aggiunge l'argomento per il momentum
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)  # Aggiunge l'argomento per il weight decay
parser.add_argument(
    "--noise_sd",
    default=0.0,
    type=float,
    help="standard deviation of Gaussian noise for data augmentation",
)  # Aggiunge l'argomento per la deviazione standard del rumore Gaussiano per l'augmentation
parser.add_argument(
    "--gpu", default=None, type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)  # Aggiunge l'argomento per gli ID delle GPU
parser.add_argument(
    "--print-freq",
    default=1,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)  # Aggiunge l'argomento per la frequenza di stampa delle informazioni


args = parser.parse_args()  # Analizza gli argomenti della riga di comando

# Funzione principale
def main():
    
    #f args.gpu:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # Imposta le GPU visibili

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)  # Crea la cartella di output se non esiste

    train_dataset = get_dataset(args.dataset, "train")  # Ottiene il dataset di training
    test_dataset = get_dataset(args.dataset, "test")  # Ottiene il dataset di test
    pin_memory = args.dataset == "imagenet"  # Imposta pin_memory per ImageNet
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )  # Crea il DataLoader per il training
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )  # Crea il DataLoader per il test
    
    print("Hello")

    #maura said the model should be loaded with robust bench OKAYYY
    x_test, y_test = load_cifar10(n_examples=50)
    model = load_model(model_name='Sehwag2021Proxy_R18', dataset='cifar10', threat_model='L2')

    print("bye")
  # Ottiene l'architettura del modello

    logfilename = os.path.join(args.outdir, "log.txt")  # Imposta il nome del file di log
    init_logfile(
        logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttest loss\ttest acc"
    )  # Inizializza il file di log

    criterion = CrossEntropyLoss()  # Imposta la funzione di perdita
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )  # Imposta l'ottimizzatore
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)  # Imposta il scheduler

    print("Before training")
    for epoch in range(args.epochs):  # Ciclo sulle epoche
        before = time.time()  # Tempo iniziale
        print("time setted")
        train_loss, train_acc = train(
            train_loader, model, criterion, optimizer, epoch, args.noise_sd, scheduler
        )  # Esegue il training
        print("ends the training")             
        test_loss, test_acc = test(test_loader, model, criterion, args.noise_sd)  # Esegue il test
        after = time.time()  # Tempo finale
        print("ends the testing")

        log(
            logfilename,
            "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch,
                str(datetime.timedelta(seconds=(after - before))),
                scheduler.get_last_lr()[0],  # Usa get_last_lr()
                train_loss,
                train_acc,
                test_loss,
                test_acc,
            ),
        )  # Registra le informazioni sul log

        torch.save(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(args.outdir, "checkpoint.pth.tar"),
        )  # Salva il checkpoint del modello


# Funzione di training
def train(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion,
    optimizer: Optimizer,
    epoch: int,
    noise_sd: float,
    scheduler,
):
    batch_time = AverageMeter()  # Inizializza il misuratore del tempo per batch
    data_time = AverageMeter()  # Inizializza il misuratore del tempo per dati
    losses = AverageMeter()  # Inizializza il misuratore delle perdite
    top1 = AverageMeter()  # Inizializza il misuratore dell'accuratezza top-1
    top5 = AverageMeter()  # Inizializza il misuratore dell'accuratezza top-5
    end = time.time()  # Tempo iniziale

    model.train()  # Imposta il modello in modalità training

    for i, (inputs, targets) in enumerate(loader):  # Ciclo sui batch
        data_time.update(time.time() - end)  # Aggiorna il tempo per i dati

        inputs = inputs + torch.randn_like(inputs, device="cpu") * noise_sd  # Aggiunge rumore Gaussiano ai dati
        print("Inputs shape:", inputs.shape)
        print("post inputs")
        outputs = model(inputs)  # Esegue il forward pass
        print("Outputs shape:", outputs.shape)
        loss = criterion(outputs, targets)  # Calcola la perdita
        
        print("Targets shape:", targets.shape)
        print("Loss:", loss.item())
        

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))  # Calcola l'accuratezza top-1 e top-5
        print("Acc1:", acc1.item())
        print("Acc5:", acc5.item())
        losses.update(loss.item(), inputs.size(0))  # Aggiorna il misuratore delle perdite
        top1.update(acc1.item(), inputs.size(0))  # Aggiorna il misuratore dell'accuratezza top-1
        print("Updating losses, top1, top5")

        top5.update(acc5.item(), inputs.size(0))  # Aggiorna il misuratore dell'accuratezza top-5

        print("pre optimizer")
        optimizer.zero_grad()  # Azzera i gradienti
        loss.backward()  # Calcola i gradienti
        optimizer.step()  # Aggiorna i pesi del modello
        scheduler.step()  # Aggiorna il scheduler

        batch_time.update(time.time() - end)  # Aggiorna il tempo per batch
        end = time.time()  # Tempo finale
        print("PRE STAMPA")
        if i % args.print_freq == 0:  # Controlla la frequenza di stampa
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i+1,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )  # Stampa le informazioni sul training

    return (losses.avg, top1.avg)  # Ritorna la perdita media e l'accuratezza top-1 media


# Funzione di test
def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()  # Inizializza il misuratore del tempo per batch
    data_time = AverageMeter()  # Inizializza il misuratore del tempo per dati
    losses = AverageMeter()  # Inizializza il misuratore delle perdite
    top1 = AverageMeter()  # Inizializza il misuratore dell'accuratezza top-1
    top5 = AverageMeter()  # Inizializza il misuratore dell'accuratezza top-5
    end = time.time()  # Tempo iniziale

    model.eval()  # Imposta il modello in modalità evaluation

    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for i, (inputs, targets) in enumerate(loader):  # Ciclo sui batch
            data_time.update(time.time() - end)  # Aggiorna il tempo per i dati

            inputs = inputs + torch.randn_like(inputs, device="cpu") * noise_sd  # Aggiunge rumore Gaussiano ai dati

            outputs = model(inputs)  # Esegue il forward pass
            loss = criterion(outputs, targets)  # Calcola la perdita

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))  # Calcola l'accuratezza top-1 e top-5
            losses.update(loss.item(), inputs.size(0))  # Aggiorna il misuratore delle perdite
            top1.update(acc1.item(), inputs.size(0))  # Aggiorna il misuratore dell'accuratezza top-1
            top5.update(acc5.item(), inputs.size(0))  # Aggiorna il misuratore dell'accuratezza top-5

            batch_time.update(time.time() - end)  # Aggiorna il tempo per batch
            end = time.time()  # Tempo finale

            if i % args.print_freq == 0:  # Controlla la frequenza di stampa
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Acc@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i+1,
                        len(loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )  # Stampa le informazioni sul test

    return (losses.avg, top1.avg)  # Ritorna la perdita media e l'accuratezza top-1 media


if __name__ == "__main__":
    main()  # Esegue la funzione principale
