#Author: Farzad Khaleghi

# ***The program below is the standard version of NBC***


import pandas as pd
import math
import time



def cross_validation(data, folds):
    
    '''In this function we are taking in the training set and the number of folds to perform a k-fold
    cross-validation. What this function is doing is taking the data splitting it into k folds and then
    generating a list that gives information about which k-1 pieces are for training and which 1 piece
    is for testing, and obviously we repeat this process until all pieces have been used as the test set
    once. After sending the splits into their respective function we then compare their performance 
    on the classes they guessed vs what the actual class is, this allows us to be able to see the
    percentage of the correctly guessed classes, which then we take the average across all k times this
    process was repeated.'''
    
    
    rows = data.shape[0]
    
    split_size = int(rows / folds)
    
    splits = []
    
    for k in range(folds):
        s_range = []
        
        s_range.append((k * split_size))
        s_range.append((k+1)*split_size)
        
        splits.append(s_range)
    
    crsval_data = []
    
    for i in range(folds):
        order = []
        for b in range(folds):
            if i == b:
                continue
            else:
                order.append(b)
            
        order.append(i)
        crsval_data.append(order)
    
    accuracy = []
    for i in range(folds):
        train_pieces = []
        test_data = None
        
        for j in crsval_data[i]:
            
            if crsval_data[i].index(j) == folds - 1:
                
                test_data = data[splits[j][0]:splits[j][1]]
        
            else:
                train_pieces.append(data[splits[j][0]:splits[j][1]])
        
        concat_data = pd.concat(train_pieces)
        
        
        v = []
        learn_nb(concat_data, v)
        predictions = []
        classify(test_data, pv, v_prob, vocab, v, predictions)
        count = 0
        for i in range(split_size):
            if test_data.iloc[i,1] == predictions[i]:
                count += 1
            else:
                continue
        
        accuracy.append(count/split_size)
    
    print("The accuracies are:",accuracy)
    mean_acc = (sum(accuracy) / folds) * 100
    print("The mean accuracy => ", mean_acc,"%", sep='')
            
        
        
                    
def learn_nb(data, v):
    
    '''In this function we are essentially taking in our training dataset and making the vocabulary
    and using each word in this vocabulary to see the probability of the given word appearing in each
    class, we store all these probabilities in their own lists for easy access when we are trying to
    make decisions in the classify function.'''
    
    
    for i in (data["class"]):
        if i not in v:
            v.append(i)
    
    global vocab
    vocab = {}
    count = 0
    
    for b in (data["abstract"]):
        line = b.split()
        for c in line:
            if c not in vocab:
                vocab[c] = count
                count += 1
                
    global pv
    pv = {}
    global v_prob
    v_prob = []
    
    for vj in v:
        doc_vj = []
        ind = v.index(vj)
        vj_prob = [0]*len(vocab)
        
        sentences = data.loc[data["class"] == vj]
        
        for index, row in sentences.iterrows():
            doc_vj.append(row['abstract'])
        
        p_vj = len(doc_vj) / data.shape[0]
        
        pv[vj] = p_vj
        
        text_j = " ".join(doc_vj)
        text_j = " ".join(text_j.split())
        
        num = text_j.split()
        
        dis_words = set()
        
        
        for char in num:
            if char not in dis_words:
                dis_words.add(char)
        
        n = len(dis_words)
        
        for wk in vocab:
            
            if wk not in dis_words:
                nk = 0
            else:
                nk = num.count(wk)
            
            p_wkvj = (nk + 1) / (n + len(vocab))
            
            vj_prob[vocab.get(wk)] = p_wkvj
        
        
        v_prob += [vj_prob]
        
        
    
def classify(t_data, prob_v, prob_voc, voc, v, pred):
    
    '''This function essentially reads the abstract column from the test data and checks whether each word
    word appears in the predefined vocabulary, taking these distinct words it then calculates how much 
    the abstract might belong to each class and then it just takes the highest value indicating the most 
    probable.'''
    
    for line in t_data["abstract"]:
        sen = line.split()
        pos = []
        for s in sen:
            if s in voc:
                pos.append(s)
        
        dis_pos = set()
        for p in pos:
            if p not in dis_pos:
                dis_pos.add(p)
                
        
        
        pos_pred = {}
        
        
        for el in v:
            p_el = prob_v[el]
            allProb = 0

            for w in dis_pos:
                calculation = pos.count(w) * math.log(prob_voc[v.index(el)][voc.get(w)])

                allProb += calculation
           
            result = math.log(p_el) + allProb
            
            pos_pred[el] = result
        
        pred.append(max(pos_pred, key=pos_pred.get))   


#The code below runs the the training data on the train function and the test data on the test function
              
file_trg = pd.read_csv("trg.csv") # "trg.csv" is file that has the training data
v = []
learn_nb(file_trg, v)

test_data = pd.read_csv("tst.csv") # "tst.csv" is file that has the test data
predictions = []
classify(test_data, pv, v_prob, vocab, v, predictions)


#code below is to test the classifier using a 10 fold cross-validation

file = pd.read_csv("trg.csv")
cross_validation(file, 10)

