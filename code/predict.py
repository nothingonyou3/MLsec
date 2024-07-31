"""This script loads a base classifier and then runs PREDICT on many examples from a dataset."""
#The program predict.py makes predictions using g on a bunch of inputs. For example,

import argparse
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
from architectures import get_architecture
import datetime

parser = argparse.ArgumentParser(description="Predict on many examples")
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
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    #model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
    #NEL CHECKPOINT DEVO PASSARE IL PATH DEI WEIGHTS CALCOLATI CON L'ALTRO SCRIPT, PROBABILMENTE train.py
    checkpoint = torch.load(args.base_classifier) #qui dovrei caricare il mio modello con quello scritto su ---the weights with the other script are the ones we obtain with train.py? How can I save them?
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint["state_dict"])

    # create the smoothed classifier g
    smoothed_classifier = Smooth(
        base_classifier, get_num_classes(args.dataset), args.sigma #calcolo lo smoothing di quel classifier
    )
    
    
    """
    Evaluate its robutness
    !pip install -q foolbox
    import foolbox as fb
    qua forse al posto di model gli devo passare lo smoothed classifier
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    _, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[8/255])
    print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))
    """
    
    """
    Valutazione con auto attack
    # autoattack is installed as a dependency of robustbench so there is not need to install it separately
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test, y_test)
    
    """

    # prepare output file
    f = open(args.outfile, "w")
    print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        before_time = time()

        # make the prediction
        prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)

        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print(
            "{}\t{}\t{}\t{}\t{}".format(i, label, prediction, correct, time_elapsed),
            file=f,
            flush=True,
        )

    f.close()