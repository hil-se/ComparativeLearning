import nltk
import pandas as pd
pd.set_option('display.max_columns', None)
# import fasttext as ft
import os
import re
# from nltk.corpus import stopwords
import random
from transformers import GPT2Tokenizer
import torch
# nltk.download('stopwords')
to_remove = [",", ".", "<", ">", "?", "/", ";", ":", "'", "!", "#", "$", "%", "^", "~",
             "*", "(", ")", "{", "}", "[", "]", "\\", "-", "_", "\n", "\t" "@", "&", "`"]

# en_stopwords = stopwords.words('english')

en_stopwords = ["a", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaaa", "about",
                "above", "across", "after", "again", "against", "all", "almost", "alone",
                "along", "already", "also", "although", "always", "am", "among", "an", "and",
                "another", "any", "anybody", "anyone", "anything", "anywhere", "are",  "aren't",
                "around", "as", "ask", "asked", "asking", "asks", "at", "away", "b", "back", "be",
                "became", "because", "become", "becomes", "been", "before", "began", "behind", "being",
                "beings", "below", "best", "better", "between", "big", "both", "but", "by", "c", "came",
                "can", "cannot", "can't", "case", "cases", "certain", "certainly", "clear", "clearly",
                "come", "could", "couldn't", "d", "did", "didn't", "differ", "different", "differently",
                "do", "does", "doesn't", "doing", "done", "don't", "down", "downed", "downing", "downs",
                "during", "e", "each", "early", "either", "end", "ended", "ending", "ends", "enough", "even",
                "evenly", "ever", "every", "everybody", "everyone", "everything", "everywhere", "f", "face",
                "faces", "fact", "facts", "far", "felt", "few", "find", "finds", "first", "for", "four",
                "from", "full", "fully", "further", "furthered", "furthering", "furthers", "g", "gave",
                "general", "generally", "get", "gets", "give", "given", "gives", "go", "going", "good",
                "goods", "got", "great", "greater", "greatest", "group", "grouped", "grouping", "groups",
                "h", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll",
                "her", "here", "here's", "hers", "herself", "he's", "high", "higher", "highest", "him",
                "himself", "his", "how", "however", "how's", "i", "i'd", "if", "i'll", "i'm", "important",
                "in", "interest", "interested", "interesting", "interests", "into", "is", "isn't", "it",
                "its", "it's", "itself", "i've", "j", "just", "k", "keep", "keeps", "kind", "knew", "know",
                "known", "knows", "l", "large", "largely", "last", "later", "latest", "least", "less", "let",
                "lets", "let's", "like", "likely", "long", "longer", "longest", "m", "made", "make", "making",
                "man", "many", "may", "me", "member", "members", "men", "might", "more", "most", "mostly", "mr",
                "mrs", "much", "must", "mustn't", "my", "myself", "n", "necessary", "need", "needed", "needing",
                "needs", "never", "new", "newer", "newest", "next", "no", "nobody", "non", "noone", "nor", "not",
                "nothing", "now", "nowhere", "number", "numbers", "o", "of", "off", "often", "old", "older", "oldest",
                "on", "once", "one", "only", "open", "opened", "opening", "opens", "or", "order", "ordered", "ordering",
                "orders", "other", "others", "ought", "our", "ours", "ourselves", "out", "over", "own", "p", "part",
                "parted", "parting", "parts", "per", "perhaps", "place", "places", "point", "pointed", "pointing",
                "possible", "q", "quite", "r", "rather", "really", "right",  "s", "said", "same", "saw", "say", "says",
                "see", "seem", "seemed", "seeming", "seems", "sees",  "shall", "shan't", "she", "she'd", "she'll",
                "she's", "should", "shouldn't",  "since", "small",  "so", "some", "somebody", "someone", "something",
                "somewhere", "state", "states", "still", "such", "sure", "t", "take", "taken", "than", "that", "that's",
                "the", "their", "theirs", "them", "themselves", "then", "there", "therefore", "there's", "these", "they",
                "they'd", "they'll", "they're", "they've", "thing", "things", "think", "thinks", "this", "those", "though",
                "thought", "thoughts", "three", "through", "thus", "to", "today", "together", "too", "took", "toward",
                "two", "u", "under", "until", "up", "upon", "us", "use", "used", "uses", "v", "very","via", "w", "want",
                "wanted", "wanting", "wants", "was", "wasn't", "way", "ways", "we", "we'd", "well", "we'll",  "went", "were",
                "we're", "weren't", "we've", "what", "what's", "when", "when's", "where", "where's", "whether", "which", "while",
                "who", "whole", "whom", "who's", "whose", "why", "why's", "will", "with", "within", "without", "won't", "work",
                "worked", "working", "works", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "your",
                "you're", "yours", "yourself", "yourselves", "you've", "z"]

SEQUENCE_LEN = 50

def filter(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[\W_]', '', text)
    text = text.lower()
    for word in en_stopwords:
        text = text.replace(word.lower(), " ")
    while "  " in text:
      text = text.replace("  ", " ")
    text = ' '.join(word for word in text.split())
    return text

def loadData(dataName="appceleratorstudio", datatype="train"):
    path = "../../Data/GPT2SP Data/Split/"
    df = pd.read_csv(path+dataName+"_"+datatype+".csv")

    # df = df.sample(frac=1)
    # df = df.head(min(1200, len(df.index)))

    return df

def GPTencode(tokenizer, sentence):
    # sentence = filter(sentence)
    encoded = tokenizer.encode(sentence)
    if len(encoded) > SEQUENCE_LEN:
        encoded = encoded[:SEQUENCE_LEN]
    elif len(encoded) < SEQUENCE_LEN:
        padding = SEQUENCE_LEN - len(encoded)
        for _ in range(padding):
            encoded.append(3)
    return encoded
    # return torch.Tensor(encoded).to(torch.int)

def process(dataName="appceleratorstudio", labelName="Storypoint", LM="FastText", num_to_add=1, num_of_labels=2):
    if LM=="FastText":
        # model = ft.load_model("../../../cc.en.300.bin")
        print()
    elif LM=="GPT2":
        model = GPT2Tokenizer.from_pretrained('gpt2')
        model.pad_token = '[PAD]'
    train = loadData(dataName=dataName, datatype="train")
    valid = loadData(dataName=dataName, datatype="val")
    test = loadData(dataName=dataName, datatype="test")

    # num_to_add = 1

    res_tr = []
    res_v = []
    res_ts = []

    train_len = len(train.index)
    test_len = len(test.index)
    val_len = len(valid.index)

    tr_indices = []
    ts_indices = []
    vl_indices = []

    lim = min(num_to_add*100, 1000)

    print("\nGenerating training data...")
    for indexA, rowA in train.iterrows():
        added = 0
        for i in range(num_to_add):
            foundOne = False
            cnter = 0
            while foundOne==False:
                if cnter>=lim:
                    foundOne=True
                cnter+=1
                indexB = random.randint(0, train_len - 1)
                while indexA==indexB:
                    indexB = random.randint(0, train_len - 1)
                rowB = train.iloc[indexB]
                varA = rowA[labelName]
                varB = rowB[labelName]
                label = 0
                if varA > varB:
                    label = 1
                elif varA < varB:
                    label = -1
                if num_of_labels==2:
                    if label!=0:
                        if (indexA, indexB) not in tr_indices and (indexB, indexA) not in tr_indices:
                            tr_indices.append((indexA, indexB))
                            added+=1
                else:
                    if (indexA, indexB) not in tr_indices and (indexB, indexA) not in tr_indices:
                        tr_indices.append((indexA, indexB))
                        added+=1
                if added>=num_to_add:
                    foundOne=True

    for pair in tr_indices:
        indexA = pair[0]
        indexB = pair[1]
        rowA = train.iloc[indexA]
        rowB = train.iloc[indexB]
        varA = rowA[labelName]
        varB = rowB[labelName]
        label = 0
        if varA > varB:
            label = 1
        elif varA < varB:
            label = -1
        if num_of_labels==2:
            if label != 0:
                textA = rowA["Issue"]
                textB = rowB["Issue"]
                if LM == "FastText":
                    print()
                    # embA = (model.get_sentence_vector(textA)).tolist()
                    # embB = (model.get_sentence_vector(textB)).tolist()
                elif LM == "GPT2":
                    embA = GPTencode(model, textA)
                    embB = GPTencode(model, textB)
                res_tr.append({"A": embA, "B": embB, "Label": label})
        else:
            textA = rowA["Issue"]
            textB = rowB["Issue"]
            if LM == "FastText":
                print()
                # embA = (model.get_sentence_vector(textA)).tolist()
                # embB = (model.get_sentence_vector(textB)).tolist()
            elif LM == "GPT2":
                embA = GPTencode(model, textA)
                embB = GPTencode(model, textB)
            res_tr.append({"A": embA, "B": embB, "Label": label})

    data_tr = pd.DataFrame(res_tr)
    data_tr.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_train.csv")

    print("\nGenerating validation data...")
    for indexA, rowA in valid.iterrows():
        added = 0
        for i in range(num_to_add):
            foundOne = False
            while foundOne == False:
                indexB = random.randint(0, val_len - 1)
                while indexA == indexB:
                    indexB = random.randint(0, val_len - 1)
                rowB = valid.iloc[indexB]
                varA = rowA[labelName]
                varB = rowB[labelName]
                label = 0
                if varA > varB:
                    label = 1
                elif varA < varB:
                    label = -1
                if num_of_labels == 2:
                    if label != 0:
                        if (indexA, indexB) not in vl_indices and (indexB, indexA) not in vl_indices:
                            vl_indices.append((indexA, indexB))
                            added += 1
                else:
                    if (indexA, indexB) not in vl_indices and (indexB, indexA) not in vl_indices:
                        vl_indices.append((indexA, indexB))
                        added += 1
                if added >= num_to_add:
                    foundOne = True

    for pair in vl_indices:
        indexA = pair[0]
        indexB = pair[1]
        rowA = valid.iloc[indexA]
        rowB = valid.iloc[indexB]
        varA = rowA[labelName]
        varB = rowB[labelName]
        label = 0
        if varA > varB:
            label = 1
        elif varA < varB:
            label = -1
        if num_of_labels == 2:
            if label != 0:
                textA = rowA["Issue"]
                textB = rowB["Issue"]
                if LM == "FastText":
                    print()
                    # embA = (model.get_sentence_vector(textA)).tolist()
                    # embB = (model.get_sentence_vector(textB)).tolist()
                elif LM == "GPT2":
                    embA = GPTencode(model, textA)
                    embB = GPTencode(model, textB)
                res_v.append({"A": embA, "B": embB, "Label": label})
        else:
            textA = rowA["Issue"]
            textB = rowB["Issue"]
            if LM == "FastText":
                print()
                # embA = (model.get_sentence_vector(textA)).tolist()
                # embB = (model.get_sentence_vector(textB)).tolist()
            elif LM == "GPT2":
                embA = GPTencode(model, textA)
                embB = GPTencode(model, textB)
            res_v.append({"A": embA, "B": embB, "Label": label})
    data_v = pd.DataFrame(res_v)
    data_v.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_val.csv")

    print("\nGenerating testing data...")
    for indexA, rowA in test.iterrows():
        added = 0
        for i in range(num_to_add):
            foundOne = False
            while foundOne == False:
                indexB = random.randint(0, test_len - 1)
                while indexA == indexB:
                    indexB = random.randint(0, test_len - 1)
                rowB = test.iloc[indexB]
                varA = rowA[labelName]
                varB = rowB[labelName]
                label = 0
                if varA > varB:
                    label = 1
                elif varA < varB:
                    label = -1
                if num_of_labels == 2:
                    if label != 0:
                        if (indexA, indexB) not in ts_indices and (indexB, indexA) not in ts_indices:
                            ts_indices.append((indexA, indexB))
                            added += 1
                else:
                    if (indexA, indexB) not in ts_indices and (indexB, indexA) not in ts_indices:
                        ts_indices.append((indexA, indexB))
                        added += 1
                if added >= num_to_add:
                    foundOne = True

    for pair in ts_indices:
        indexA = pair[0]
        indexB = pair[1]
        rowA = test.iloc[indexA]
        rowB = test.iloc[indexB]
        varA = rowA[labelName]
        varB = rowB[labelName]
        label = 0
        if varA > varB:
            label = 1
        elif varA < varB:
            label = -1
        if num_of_labels == 2:
            if label != 0:
                textA = rowA["Issue"]
                textB = rowB["Issue"]
                if LM == "FastText":
                    print()
                    # embA = (model.get_sentence_vector(textA)).tolist()
                    # embB = (model.get_sentence_vector(textB)).tolist()
                elif LM == "GPT2":
                    embA = GPTencode(model, textA)
                    embB = GPTencode(model, textB)
                res_ts.append({"A": embA, "B": embB, "Label": label})
        else:
            textA = rowA["Issue"]
            textB = rowB["Issue"]
            if LM == "FastText":
                print()
                # embA = (model.get_sentence_vector(textA)).tolist()
                # embB = (model.get_sentence_vector(textB)).tolist()
            elif LM == "GPT2":
                embA = GPTencode(model, textA)
                embB = GPTencode(model, textB)
            res_ts.append({"A": embA, "B": embB, "Label": label})
    data_ts = pd.DataFrame(res_ts)
    data_ts.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_test.csv")

    print("Done.")
    print("Training data size:", len(data_tr.index))
    print("Validation data size:", len(data_v.index))
    print("Testing data size:", len(data_ts.index))

    test_list = []
    for indexA, rowA in test.iterrows():
        varA = rowA[labelName]
        ind = indexA
        textA = rowA["Issue"]
        if LM=="FastText":
            print()
            # embA = model.get_sentence_vector(textA)
        elif LM=="GPT2":
            embA = GPTencode(model, textA)
        toAdd = {"indexA": ind, "A": embA, "Text": rowA["Issue"], "Score": varA}
        test_list.append(toAdd)

    test_list = pd.DataFrame(test_list)

    # print(data_tr.head(), "\n")
    # print(data_v.head(), "\n")
    # print(data_ts.head(), "\n")

    return data_tr, data_v, data_ts, test_list

def processRegression(dataName="appceleratorstudio", labelName="Storypoint", LM="GPT2"):
    if LM=="FastText":
        # model = ft.load_model("../../../cc.en.300.bin")
        print()
    elif LM=="GPT2":
        model = GPT2Tokenizer.from_pretrained('gpt2')
        model.pad_token = '[PAD]'
    train = loadData(dataName=dataName, datatype="train")
    # print(train)
    valid = loadData(dataName=dataName, datatype="val")
    test = loadData(dataName=dataName, datatype="test")

    # total_data = train._append(valid)
    # total_data = total_data._append(test)
    # total_data = total_data.sample(frac=1)
    # total_data.reset_index(inplace=True)
    # threshold = total_data[labelName].describe()["std"]
    # threshold = round(threshold.item(), 3)

    res_tr = []
    res_v = []
    res_ts = []

    train_len = len(train.index)
    test_len = len(test.index)
    val_len = len(valid.index)

    print("\nGenerating training data...")
    for indexA, rowA in train.iterrows():
        varA = rowA[labelName]
        textA = rowA["Issue"]
        # textA = filter(textA)
        # print("Filtered:")
        # print(textA,"\n")
        embA = GPTencode(model, textA)
        res_tr.append({"A": embA, "Label": varA})
    data_tr = pd.DataFrame(res_tr)
    data_tr.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_train_Regression.csv")

    print("\nGenerating validation data...")
    for indexA, rowA in valid.iterrows():
        varA = rowA[labelName]
        textA = rowA["Issue"]
        # textA = filter(textA)
        embA = GPTencode(model, textA)
        res_v.append({"A": embA, "Label": varA})
    data_v = pd.DataFrame(res_v)
    data_v.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_val_Regression.csv")

    print("\nGenerating testing data...")
    for indexA, rowA in test.iterrows():
        varA = rowA[labelName]
        textA = rowA["Issue"]
        # textA = filter(textA)
        embA = GPTencode(model, textA)
        res_ts.append({"A": embA, "Label": varA})
    data_ts = pd.DataFrame(res_ts)
    data_ts.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_test_Regression.csv")

    print("Done.")
    print("Training data size:", len(data_tr.index))
    print("Validation data size:", len(data_v.index))
    print("Testing data size:", len(data_ts.index))

    return data_tr, data_v, data_ts

# def process2labels(dataName="appceleratorstudio", labelName="Storypoint", LM="GPT2"):
#     if LM=="FastText":
#         # model = ft.load_model("../../../cc.en.300.bin")
#         print()
#     elif LM=="GPT2":
#         model = GPT2Tokenizer.from_pretrained('gpt2')
#         model.pad_token = '[PAD]'
#     train = loadData(dataName=dataName, datatype="train")
#     valid = loadData(dataName=dataName, datatype="val")
#     test = loadData(dataName=dataName, datatype="test")
#
#     # total_data = train._append(valid)
#     # total_data = total_data._append(test)
#     # total_data = total_data.sample(frac=1)
#     # total_data.reset_index(inplace=True)
#     # threshold = total_data[labelName].describe()["std"]
#     # threshold = round(threshold.item(), 3)
#
#     res_tr = []
#     res_v = []
#     res_ts = []
#
#     train_len = len(train.index)
#     test_len = len(test.index)
#     val_len = len(valid.index)
#
#     print("\nGenerating training data...")
#     for indexA, rowA in train.iterrows():
#         for indexb, rowB in train.iterrows():
#             varA = rowA[labelName]
#             varB = rowB[labelName]
#             label = 0
#             if varA > varB:
#                 label = 1
#             elif varA < varB:
#                 label = -1
#             if label!=0:
#                 textA = rowA["Issue"]
#                 textB = rowB["Issue"]
#                 # textA = filter(textA)
#                 # textB = filter(textB)
#                 embA = GPTencode(model, textA)
#                 embB = GPTencode(model, textB)
#                 res_tr.append({"A": embA, "B": embB, "Label": label})
#     data_tr = pd.DataFrame(res_tr)
#     data_tr.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_train.csv")
#
#     print("\nGenerating validation data...")
#     for indexA, rowA in valid.iterrows():
#         for indexb, rowB in valid.iterrows():
#             varA = rowA[labelName]
#             varB = rowB[labelName]
#             label = 0
#             if varA > varB:
#                 label = 1
#             elif varA < varB:
#                 label = -1
#             if label!=0:
#                 textA = rowA["Issue"]
#                 textB = rowB["Issue"]
#                 # textA = filter(textA)
#                 # textB = filter(textB)
#                 embA = GPTencode(model, textA)
#                 embB = GPTencode(model, textB)
#                 res_v.append({"A": embA, "B": embB, "Label": label})
#     data_v = pd.DataFrame(res_v)
#     data_v.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_val.csv")
#
#     print("\nGenerating testing data...")
#     for indexA, rowA in test.iterrows():
#         for indexb, rowB in test.iterrows():
#             varA = rowA[labelName]
#             varB = rowB[labelName]
#             label = 0
#             if varA > varB:
#                 label = 1
#             elif varA < varB:
#                 label = -1
#             if label!=0:
#                 textA = rowA["Issue"]
#                 textB = rowB["Issue"]
#                 # textA = filter(textA)
#                 # textB = filter(textB)
#                 embA = GPTencode(model, textA)
#                 embB = GPTencode(model, textB)
#                 res_ts.append({"A": embA, "B": embB, "Label": label})
#     data_ts = pd.DataFrame(res_ts)
#     data_ts.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_test.csv")
#
#     print("Done.")
#     print("Training data size:", len(data_tr.index))
#     print("Validation data size:", len(data_v.index))
#     print("Testing data size:", len(data_ts.index))
#
#     test_list = []
#     for indexA, rowA in test.iterrows():
#         varA = rowA[labelName]
#         ind = indexA
#         textA = rowA["Issue"]
#         # textA = filter(textA)
#         if LM=="FastText":
#             print()
#             # embA = model.get_sentence_vector(textA)
#         elif LM=="GPT2":
#             embA = GPTencode(model, textA)
#         toAdd = {"indexA": ind, "A": embA, "Text": rowA["Issue"], "Score": varA}
#         test_list.append(toAdd)
#
#     test_list = pd.DataFrame(test_list)
#
#     # print(data_tr.head(), "\n")
#     # print(data_v.head(), "\n")
#     # print(data_ts.head(), "\n")
#
#     return data_tr, data_v, data_ts, test_list

# def process3labels(dataName="appceleratorstudio", labelName="Storypoint", LM="GPT2"):
#     if LM=="FastText":
#         # model = ft.load_model("../../../cc.en.300.bin")
#         print()
#     elif LM=="GPT2":
#         model = GPT2Tokenizer.from_pretrained('gpt2')
#         model.pad_token = '[PAD]'
#     train = loadData(dataName=dataName, datatype="train")
#     valid = loadData(dataName=dataName, datatype="val")
#     test = loadData(dataName=dataName, datatype="test")
#
#     res_tr = []
#     res_v = []
#     res_ts = []
#
#     train_len = len(train.index)
#     test_len = len(test.index)
#     val_len = len(valid.index)
#
#     print("\nGenerating training data...")
#     for indexA, rowA in train.iterrows():
#         for indexb, rowB in train.iterrows():
#             varA = rowA[labelName]
#             varB = rowB[labelName]
#             label = 0
#             if varA > varB:
#                 label = 1
#             elif varA < varB:
#                 label = -1
#             textA = rowA["Issue"]
#             textB = rowB["Issue"]
#             embA = GPTencode(model, textA)
#             embB = GPTencode(model, textB)
#             res_tr.append({"A": embA, "B": embB, "Label": label})
#     data_tr = pd.DataFrame(res_tr)
#     data_tr.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_train.csv")
#
#     print("\nGenerating validation data...")
#     for indexA, rowA in valid.iterrows():
#         for indexb, rowB in valid.iterrows():
#             varA = rowA[labelName]
#             varB = rowB[labelName]
#             label = 0
#             if varA > varB:
#                 label = 1
#             elif varA < varB:
#                 label = -1
#             textA = rowA["Issue"]
#             textB = rowB["Issue"]
#             embA = GPTencode(model, textA)
#             embB = GPTencode(model, textB)
#             res_v.append({"A": embA, "B": embB, "Label": label})
#     data_v = pd.DataFrame(res_v)
#     data_v.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_val.csv")
#
#     print("\nGenerating testing data...")
#     for indexA, rowA in test.iterrows():
#         for indexb, rowB in test.iterrows():
#             varA = rowA[labelName]
#             varB = rowB[labelName]
#             label = 0
#             if varA > varB:
#                 label = 1
#             elif varA < varB:
#                 label = -1
#             textA = rowA["Issue"]
#             textB = rowB["Issue"]
#             embA = GPTencode(model, textA)
#             embB = GPTencode(model, textB)
#             res_ts.append({"A": embA, "B": embB, "Label": label})
#     data_ts = pd.DataFrame(res_ts)
#     data_ts.to_csv("../../Data/GPT2SP Data/Embeddings/" + dataName + "_test.csv")
#
#     print("Done.")
#     print("Training data size:", len(data_tr.index))
#     print("Validation data size:", len(data_v.index))
#     print("Testing data size:", len(data_ts.index))
#
#     test_list = []
#     for indexA, rowA in test.iterrows():
#         varA = rowA[labelName]
#         ind = indexA
#         textA = rowA["Issue"]
#         if LM=="FastText":
#             print()
#             # embA = model.get_sentence_vector(textA)
#         elif LM=="GPT2":
#             embA = GPTencode(model, textA)
#         toAdd = {"indexA": ind, "A": embA, "Text": rowA["Issue"], "Score": varA}
#         test_list.append(toAdd)
#
#     test_list = pd.DataFrame(test_list)
#
#     # print(data_tr.head(), "\n")
#     # print(data_v.head(), "\n")
#     # print(data_ts.head(), "\n")
#
#     return data_tr, data_v, data_ts, test_list


def generateData(dataName, labelName, LM, data_type="pairwise", labels=2, num_to_add=1):
    if data_type=="pairwise":
        return process(dataName=dataName, labelName=labelName, LM=LM, num_to_add=num_to_add, num_of_labels=labels)
    elif data_type=="regression":
        return processRegression(dataName=dataName, labelName=labelName, LM=LM)

# _, _, _, _ = generateData("clover", "Storypoint", "GPT2", "regression")
