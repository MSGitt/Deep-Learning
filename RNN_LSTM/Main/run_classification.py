import torch
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from time import time

from utils import *

def main(args):
    MODEL_NAME = args.model_name
    
    assert MODEL_NAME in ["RNN_classifier", "LSTM_classifier"], "model_name must be either 'RNN_classifier' or 'LSTM_classifier'"

    EMBED_DIM = args.embed_dim
    HIDDEN_DIM = args.hidden_dim
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    LEARNING_RATE = args.learning_rate
    VALID_RATIO = args.valid_ratio
    RANDOM_STATE = args.random_state
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("... RUNNING ...")
    print(f"> MODEL_NAME: {MODEL_NAME}")
    print(f"> EMBED_DIM: {EMBED_DIM}")
    print(f"> HIDDEN_DIM: {HIDDEN_DIM}")
    print(f"> NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"> BATCH_SIZE: {BATCH_SIZE}")
    print(f"> TEST_BATCH_SIZE: {TEST_BATCH_SIZE}")
    print(f"> LEARNING_RATE: {LEARNING_RATE}")
    print(f"> VALID_RATIO: {VALID_RATIO}")
    print(f"> RANDOM_STATE: {RANDOM_STATE}\n")

    torch.random.manual_seed(RANDOM_STATE)

    MODEL_CONFIG=(DEVICE, EMBED_DIM, HIDDEN_DIM)

    ## PREPARE DATA FOR CLASSIFICATION
    train = pd.read_csv("data/YNAT/train_data.csv", sep=',').dropna()
    test = pd.read_csv("data/YNAT/test_data.csv", sep=',').dropna()
    X, X_test, y, y_test = YNAT_preprocess(train, test)

    # DEFINE MODEL
    import model
    MODEL_CLASS = getattr(model, MODEL_NAME)
    model = MODEL_CLASS(*MODEL_CONFIG)

    # SPLIT DATA INTO TRAIN AND VALID
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=VALID_RATIO, random_state=RANDOM_STATE)

    # START TRAINING
    print("... TRAINING ...\n")
    s = time()
    model.train_model(X_train, X_valid, y_train, y_valid, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)

    print(f"\n... DONE ... TOTAL TRAINING TIME: {time()-s:.2f}s\n")
    
    # START TEST
    topic_dict = {0: "IT과학", 1: "경제", 2: "사회", 3: "생활문화", 4: "세계", 5: "스포츠", 6: "정치"}

    with torch.no_grad():
        test_accuracy, _, preds = model.predict(X_test, y_test, TEST_BATCH_SIZE, return_preds=True)
        for i, (sentence, label, pred) in enumerate(zip(X_test, y_test, preds)):
            if i == 5:
                break
            print(f'{sentence} ---->  {topic_dict[pred]} (True: {topic_dict[label]})')
                    
        print(f'\n>>>> Test accuracy: {test_accuracy*100:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default="RNN_classifier")
    parser.add_argument('--embed_dim', type=int, default=16, help='Dimension of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Dimension of hidden vector')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help="Number of data per a batch")
    parser.add_argument('--test_batch_size', type=int, default=1024, help="Number of test data per a batch")
    parser.add_argument('--learning_rate', type=float, default=0, help="Learning rate")
    parser.add_argument('--valid_ratio', type=float, default=0.1, help="Ratio of validation data")
    parser.add_argument('--random_state', type=float, default=506, help="Random state")

    args = parser.parse_args()

    main(args)