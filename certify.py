# evaluate a smoothed classifier on a dataset
#The program certify.py certifies the robustness of g on bunch of inputs. For example,
# evaluate a smoothed classifier on a dataset
# The program certify.py certifies the robustness of g on bunch of inputs. For example,
# evaluate a smoothed classifier on a dataset
# The program certify.py certifies the robustness of g on bunch of inputs. For example,
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from robustbench.data import load_cifar10
from robustbench.utils import load_model
import foolbox as fb

parser = argparse.ArgumentParser(description="Certify many examples")
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument(
    "base_classifier", type=str, help="path to saved pytorch model of base classifier"
)
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument(
    "--split", choices=["train", "test"], default="test", help="train or test set"
)
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    device = torch.device("cpu")
    x_test, y_test = load_cifar10(n_examples=50)
    checkpoint = torch.load("/home/giuliavanzato/Desktop/smoothing-mlsec-master/smoothing-mlsec-master/model_output_dir/checkpoint.pth.tar", map_location=device)  # pass the path of the weights saved with the other script
    base_classifier = load_model(model_name='Sehwag2021Proxy_R18', dataset='cifar10', threat_model='L2')
    
    # Ensure the base classifier is on the correct device
    base_classifier.load_state_dict(checkpoint["state_dict"])
    base_classifier = base_classifier.to(device)
    base_classifier.eval()  # Set the model to evaluation mode

    # Wrap the model with Foolbox
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)
    fmodel = fb.PyTorchModel(smoothed_classifier, bounds=(0, 1), device = device)

    # Load and prepare data
    x_test, y_test = load_cifar10(n_examples=50)  # Adjust n_examples as needed
    x_test = torch.tensor(x_test, device=device)
    y_test = torch.tensor(y_test, device=device)

    # Run the attack
    attack = fb.attacks.LinfPGD()
    out = attack(fmodel, x_test, y_test, epsilons=[8/255])
    print('stampaaa')

    # prepare output file
    f = open(args.outfile, "w")
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.to(device)
        label = torch.tensor(label, device=device)

        before_time = time()
        # certify the prediction of g around x
        prediction, radius = smoothed_classifier.certify(
            x, args.N0, args.N, args.alpha, args.batch
        )
        after_time = time()
        correct = int(prediction == label)
        print("iTERATIONNNNNNNNNNNNNNNNNNNN---------------------")
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print(
            "{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed
            ),
            file=f,
            flush=True,
        )

    f.close()
