import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=200, help="epochs of training")
    parser.add_argument('--batch_size', type=int, default=16, help="size of the batch")
    parser.add_argument('--test_batch', type=int, default=16, help="size of the test batch")
    parser.add_argument('--lr', type=float, default=0.0003, help="learning rate")
    parser.add_argument('--num_samples', type=int, default=70, help="number of lines per sample")
    parser.add_argument('--model', type=str, default='TCN', help="the model for classifier, TCN, CNN or LSTM")
    parser.add_argument('--transform', type=bool, default=True, help="whether datasets will transform tensors into 3d")
    parser.add_argument('--record_path', type=str, default='Record/record.txt', help="the path to store training record")
    args = parser.parse_args()
    return args