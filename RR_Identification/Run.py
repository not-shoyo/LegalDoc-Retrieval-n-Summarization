import argparse
import json
import sys
import pandas as pd

from Model.Hierarchical_BiLSTM_CRF_Classifier import *
from Prepare_Data import *
from Train import *

def main():

    # Manually handle the -f argument
    if '-f' in sys.argv:
        sys.argv.remove('-f')

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--pretrained', default = True, type = bool, help = 'Whether the model uses pretrained sentence embeddings or not')
    parser.add_argument('--data_path', default = './All_Data/Generated_Data/RR_Embeddings/LegalBERT/', type = str, help = 'Folder to store the annotated text files')
    parser.add_argument('--save_path', default = 'Saved_Models/RR_Model/', type = str, help = 'Folder where predictions and models will be saved')
    parser.add_argument('--saved_json', default = 'RR_Identification/Saved/', type = str, help = 'Folder where json will be saved')
    parser.add_argument('--cat_path', default = 'RR_Identification/categories.txt', type = str, help = 'Path to file containing category details')     
    # =================================================================================================
    # parser.add_argument('--dataset_size', default = 245, type = int, help = 'Total no. of docs')
    parser.add_argument('--dataset_size', default = 7030, type = int, help = 'Total no. of docs')
    # =================================================================================================
    parser.add_argument('--num_folds', default = 5, type = int, help = 'No. of folds to divide the dataset into')
    parser.add_argument('--device', default = 'cuda', type = str, help = 'cuda / cpu')
    
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--print_every', default = 20, type = int, help = 'Epoch interval after which validation micro f1 and loss will be printed')
    parser.add_argument('--lr', default = 0.01, type = float, help = 'Learning Rate')
    parser.add_argument('--reg', default = 0, type = float, help = 'L2 Regularization')
    # =================================================================================================
    parser.add_argument('--emb_dim', default = 512, type = int, help = 'Sentence embedding dimension') # ????
    # =================================================================================================
    parser.add_argument('--word_emb_dim', default = 384, type = int, help = 'Word embedding dimension, applicable only if pretrained = False')
    parser.add_argument('--epochs', default = 300, type = int)

    parser.add_argument('--val_fold', default = '1', type = str, help = 'Fold number to be used as validation, use cross for num_folds cross validation')
    # parser.add_argument('-f')

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    print('\nPreparing data ...', end = ' ')
    idx_order = prepare_folds(args)
    x, y, word2idx, tag2idx = prepare_data(idx_order, args)
    print('Done')

    print('Vocabulary size:', len(word2idx))
    print('#Tags:', len(tag2idx))

    # Dump word2idx and tag2idx
    with open(args.saved_json + 'word2idx.json', 'w') as fp:
        json.dump(word2idx, fp)
    with open(args.saved_json + 'tag2idx.json', 'w') as fp:
        json.dump(tag2idx, fp)

    if args.val_fold == 'cross':
        print('\nCross-validation\n')
        for f in range(args.num_folds):

            print('\nInitializing model ...', end = ' ')  
            model = Hier_LSTM_CRF_Classifier(len(tag2idx), args.emb_dim, tag2idx['<start>'], tag2idx['<end>'], tag2idx['<pad>'], vocab_size = len(word2idx), word_emb_dim = args.word_emb_dim, pretrained = args.pretrained, device = args.device).to(args.device)
            print('Done')
            
            print('\nEvaluating on fold', f, '...')        
            learn(model, x, y, tag2idx, f, args)

    else:

        print('\nInitializing model ...', end = ' ')   
        model = Hier_LSTM_CRF_Classifier(len(tag2idx), args.emb_dim, tag2idx['<start>'], tag2idx['<end>'], tag2idx['<pad>'], vocab_size = len(word2idx), word_emb_dim = args.word_emb_dim, pretrained = args.pretrained, device = args.device).to(args.device)
        print('Done')

        print('\nEvaluating on fold', args.val_fold, '...')        
        learn(model, x, y, tag2idx, int(args.val_fold), args)
        


def get_input_docs_text():
    import os

    folder_path = "All_Data/Generated_Data/Paper_Data/Documents"
    file_contents_dict = {}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a text file
        if filename.endswith(".txt"):
            try:
                # Extract the integer value from the file name
                file_number = int(filename.split('.')[0].split('c')[1])
                
                with open(file_path, "r", encoding="utf-8") as file:
                    # Read the contents of the file and store in the dictionary
                    content = file.read().split('\n')
                    file_contents_dict[file_number] = content
            except ValueError:
                print(f"Skipping file {filename} as it does not have a valid integer in its name.")

    # Now, file_contents_dict contains the contents of all text files, organized by integer value of the file name
    return file_contents_dict



def label_input_docs():

    # Manually handle the -f argument
    if '-f' in sys.argv:
        sys.argv.remove('-f')

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--data_path', default = 'All_Data/Generated_Data/Paper_Data/Generated_Document_Embeddings/', type = str, help = 'Folder which has input embeddings')
    parser.add_argument('--dataset_size', default = 100, type = int, help = 'Total no. of test docs')
    parser.add_argument('--batch_size', default = 32, type = int) # ??? why 32
    parser.add_argument('--emb_dim', default = 512, type = int, help = 'Sentence embedding dimension') # ???? why 512
    parser.add_argument('--pretrained', default = True, type = bool, help = 'Whether the model uses pretrained sentence embeddings or not')
    parser.add_argument('--device', default = 'cuda', type = str, help = 'cuda / cpu')
    # parser.add_argument('-f', default = '')

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # Load the model checkpoint for the specified fold
    checkpoint_path = f"{args.save_path}/model_state1.tar"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(args.device))

    # Extract necessary information from the checkpoint
    model_state = checkpoint['state_dict']

    # Read the saved word2idx from file
    with open('RR_Identification/Saved/word2idx.json', 'r') as file:
        word2idx = json.load(file)
    
    # Read the saved tag2idx from file
    with open('RR_Identification/Saved/tag2idx.json', 'r') as file:
        tag2idx = json.load(file)
        idx2tag = {v: k for k, v in tag2idx.items()}

    # Initialize the model with the same architecture used during training
    model = Hier_LSTM_CRF_Classifier(
        n_tags=len(tag2idx),
        sent_emb_dim=args.emb_dim,
        sos_tag_idx=tag2idx['<start>'],
        eos_tag_idx=tag2idx['<end>'],
        pad_tag_idx=tag2idx['<pad>'],
        vocab_size=len(word2idx),
        # word_emb_dim=args.word_emb_dim,
        pretrained=args.pretrained,
        device=args.device
    )

    # Load the model state
    model.load_state_dict(model_state)

    # Set the model to evaluation mode
    model.eval()

    # Now you can use the loaded model for inference or other tasks
    # For example:
    # output = model(x)

    file_contents_dict = get_input_docs_text()

    idx_order = prepare_test_folds(args)
    x = prepare_test_data(idx_order, args)

    test_idx, test_pred = predict_labels(model, x, word2idx, tag2idx, args)

    respective_idx_pred_emb_doc = list(zip(test_idx, test_pred, 
                                          [x[i] for i in test_idx], 
                                          [file_contents_dict[i+1] for i in test_idx]))

    for i, pred, emb, doc in respective_idx_pred_emb_doc[:1]:
        print(f"============== i: {i}")
        print(f"============== docName: c{i+1}.txt")
        print(f"============== len(pred): {len(pred)}")
        print(f"============== len(doc): {len(doc)}")
        print(f"============== pred: {pred}")
        print(f"============== doc: {doc}")
        print(f"============== emb: {emb}")

        for sent, label in zip(doc, pred):
            print(f"{sent[:100]} -----> {label} -----> {idx2tag[label]}")

    df = pd.DataFrame(respective_idx_pred_emb_doc, columns=['i', 'pred', 'emb', 'doc'])

    # Save the DataFrame to a CSV file
    df.to_csv('test_output.csv', index=False)


if __name__ == '__main__':
    # main()
    label_input_docs()
