import argparse

DEFAULT_T = 3
DEFAULT_C = 100 # all
DEFAULT_E = 10
DEFAULT_B = 256

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('-T', type=int, default=DEFAULT_T, help="rounds of training T")
    parser.add_argument('-C', type=int, default=DEFAULT_C, help="number of clients C")
    parser.add_argument('-E', type=int, default=DEFAULT_E, help="the number of local epochs E")
    parser.add_argument('-B', type=int, default=DEFAULT_B, help="local batch size B")

    # other arguments
    parser.add_argument('-cls', type=str, default='mc', help="binary (b) or multclass (mc) classification")

    args = parser.parse_args()
    return args