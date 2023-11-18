import argparse

from Model.Hierarchical_BiLSTM_CRF_Classifier import *
from Prepare_Data import *
from Train import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained', default = True, type = bool, help = 'Whether the model uses pretrained sentence embeddings or not')
    parser.add_argument('--data_path', default = './All_Data/Generated_Data/RR_Embeddings/Sent2Vec/', type = str, help = 'Folder to store the annotated text files')
    parser.add_argument('--save_path', default = 'RR_Identification/Saved/', type = str, help = 'Folder where predictions and models will be saved')
    parser.add_argument('--saved_json', default = 'RR_Identification/Saved/', type = str, help = 'Folder where json will be saved')
    parser.add_argument('--cat_path', default = 'RR_Identification/categories.txt', type = str, help = 'Path to file containing category details')     
    parser.add_argument('--dataset_size', default = 245, type = int, help = 'Total no. of docs')
    parser.add_argument('--num_folds', default = 5, type = int, help = 'No. of folds to divide the dataset into')
    parser.add_argument('--device', default = 'cuda', type = str, help = 'cuda / cpu')
    
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--print_every', default = 20, type = int, help = 'Epoch interval after which validation micro f1 and loss will be printed')
    parser.add_argument('--lr', default = 0.01, type = float, help = 'Learning Rate')
    parser.add_argument('--reg', default = 0, type = float, help = 'L2 Regularization')
    parser.add_argument('--emb_dim', default = 512, type = int, help = 'Sentence embedding dimension')     
    parser.add_argument('--word_emb_dim', default = 384, type = int, help = 'Word embedding dimension, applicable only if pretrained = False')
    parser.add_argument('--epochs', default = 300, type = int)

    parser.add_argument('--val_fold', default = '1', type = str, help = 'Fold number to be used as validation, use cross for num_folds cross validation')
    parser.add_argument('-f')
    args = parser.parse_args()

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
        

if __name__ == '__main__':
    main()