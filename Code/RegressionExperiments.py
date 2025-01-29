import pandas as pd
from transformers import GPT2Tokenizer, GPT2PreTrainedModel, GPT2Config
import re
from GPT2SPModel import GPT2SPModel
import torch
import numpy as np
import scipy
import DatasetCreator as dc
import math
# from torchmetrics.classification import BinaryHingeLoss
import gc

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def loadData(dataset="clover", num_to_add=1):
    print("Loading data...")
    return dc.generateData(dataName=dataset, labelName="Storypoint", LM="GPT2", data_type="regression", num_to_add=num_to_add)

def custom_loss(predictions, labels):
    # loss = torch.mean(max(abs(predictions)-1, 0))
    loss = torch.mean(torch.where(abs(predictions)-1 > 0, abs(predictions)-1, 0))
    return loss

def trainModel(dataname, train, val, test):
    config = GPT2Config(num_labels=1, pad_token_id=50256)
    # model = GPT2SPModel(config)
    model = GPT2SPModel.from_pretrained('gpt2', config=config)

    # model.load_state_dict(torch.load("../../Data/GPT2SP Data/Trained models/"+dataname+".pkl", weights_only=True), strict=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    model = model.to(DEVICE)

    # train = train.to(DEVICE)
    # val = val.to(DEVICE)

    test_sp = test["Label"].tolist()
    test_len = len(test_sp)
    test = torch.Tensor(test["A"].tolist()).to(torch.int)
    test = test.to(DEVICE)

    train_A = train["A"].tolist()
    train_label = train["Label"].tolist()
    train_len = len(train.index)

    val_A = val["A"].tolist()
    val_label = val["Label"].tolist()
    val_len = len(val.index)

    model_path = "../../Data/GPT2SP Data/Models/"+dataname+"_Regression.pth"

    loss_fn = torch.nn.L1Loss()
    # loss_fn = torch.nn.HingeEmbeddingLoss()
    # loss_fn = torch.nn.MarginRankingLoss()

    batch_size = 16
    epochs_per_round = 1
    batch_start = 0
    ind_tr = 0
    ind_v = 0
    ind_ts = 0
    batch_end = (batch_size*ind_tr)+batch_size

    epochs = epochs_per_round*int(train_len/batch_size)
    # epochs = 200
    best_loss = 100
    best_loss_v = 100
    best_loss_avg = 100
    best_epoch = 0
    epochs_since_decrease = 0
    early_stopping_epochs = batch_size*5
    # early_stopping_epochs = 15

    print("Training...")
    print("Max epochs:", epochs)

    # test_pred = model(test)
    # test_pred = test_pred.to(torch.device("cpu"))
    # test_pred = test_pred.tolist()
    # test_pred.sort()
    # test_sp.sort()
    # spearmans = scipy.stats.spearmanr(test_sp, test_pred).statistic
    # _, _, _, spearmans = testModel(model, test)
    # print("Epoch: 0, Spearman's coefficient:", spearmans)

    for epoch in range(epochs):
        # Batching
        batch_start = ind_tr*batch_size
        batch_end = batch_start + batch_size
        if batch_end>=train_len:
            batch_end = train_len
            ind_tr = 0
        train_A_batch = train_A[batch_start:batch_end]
        train_label_batch = train_label[batch_start:batch_end]
        ind_tr+=1

        batch_start = ind_v * batch_size
        batch_end = batch_start + batch_size
        if batch_end >= val_len:
            batch_end = val_len
            ind_v = 0
        val_A_batch = val_A[batch_start:batch_end]
        val_label_batch = val_label[batch_start:batch_end]
        ind_v += 1

        # Formatting
        train_A_batch = torch.Tensor(train_A_batch).to(torch.int)
        train_label_batch = torch.Tensor(train_label_batch)
        train_label_batch = train_label_batch.to(DEVICE)
        val_A_batch = torch.Tensor(val_A_batch).to(torch.int)
        val_label_batch = torch.Tensor(val_label_batch)
        val_label_batch = val_label_batch.to(DEVICE)

        train_A_batch = train_A_batch.to(DEVICE)
        val_A_batch = val_A_batch.to(DEVICE)

        # Training
        train_A_pred = model(train_A_batch)
        train_A_pred = train_A_pred.to(DEVICE)
        # train_pred = torch.sub(train_A_pred, train_B_pred, alpha=1)

        val_A_pred = model(val_A_batch)
        val_A_pred = val_A_pred.to(DEVICE)
        # val_pred = torch.sub(val_A_pred, val_B_pred, alpha=1)

        # train_pred = train_pred.to(DEVICE)
        # val_pred = val_pred.to(DEVICE)

        # Loss
        loss_tr = loss_fn(train_A_pred, train_label_batch)
        loss_v = loss_fn(val_A_pred, val_label_batch)
        # loss_tr = custom_loss(train_pred, train_label_batch)
        # loss_v = custom_loss(val_pred, val_label_batch)
        # loss_tr = loss_fn(train_A_pred, train_B_pred, train_label_batch)
        # loss_v = loss_fn(val_A_pred, val_B_pred, val_label_batch)
        avg_loss = (abs(loss_tr.item())+abs(loss_v.item()))/2


        # del train_pred
        # del val_pred
        # gc.collect()
        # torch.cuda.empty_cache()
        # test_pred = model(test)
        # test_pred = test_pred.to(torch.device("cpu"))
        # test_pred = test_pred.tolist()
        # test_pred.sort()
        # test_sp.sort()
        # spearmans = scipy.stats.spearmanr(test_sp, test_pred).statistic

        # loss_tr = max(0.0, 1.0 - (train_pred * train_label_batch))
        # loss_v = max(0.0, 1.0 - (val_pred * val_label_batch))

        # loss_tr = 0
        # loss_v = 0
        #
        # train_pred = train_pred.cpu().numpy().tolist()
        # val_pred = val_pred.cpu().numpy().tolist()
        # train_label_batch = train_label_batch.cpu().numpy().tolist()
        # val_label_batch = val_label_batch.cpu().numpy().tolist()
        #
        # for i in range(len(train_pred)):
        #     loss_tr+=(max(0.0, 1.0 - (train_pred[i] * train_label_batch[i])))
        # for i in range(len(val_pred)):
        #     loss_v+=(max(0.0, 1.0 - (val_pred[i] * val_label_batch[i])))
        #
        # loss_tr/=len(train_pred)
        # loss_v/=len(val_pred)

        # _, _, _, spearmans = testModel(model, test_list)
        # print("Epoch:", epoch+1, ", Training Loss:", loss_tr.item(), ", Val Loss:", loss_v.item(), " Avg. Loss:", avg_loss, ", Spearman's coefficient:", spearmans)
        # torch.save(model.state_dict(), model_path)
        print("Epoch:", epoch + 1, ", Training Loss:", loss_tr.item(), ", Val Loss:", loss_v.item(), " Avg. Loss:", avg_loss)
        # print("Epoch:", epoch + 1, ", Loss:", loss_tr, ", Val Loss:", loss_v)

        # if loss_tr.item()<best_loss and loss_v.item()<best_loss_v:
        # if loss_tr < best_loss and loss_v < best_loss_v:
        # if loss_v.item() < best_loss_v:
        if avg_loss < best_loss_avg:
            best_loss = loss_tr.item()
            best_loss_v = loss_v.item()
            best_loss_avg = avg_loss
            # best_loss = loss_tr
            # best_loss_v = loss_v
            best_epoch = epoch+1
            epochs_since_decrease = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_since_decrease+=1
            if epochs_since_decrease>=early_stopping_epochs:
                print("\nEarly stopping limit reached.")
                print("Best loss:", best_loss, ", Best validation loss:", best_loss_v, "Best avg loss:", best_loss_avg)
                print("Loading best weights from epoch:", best_epoch)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.eval()
                return model
        loss_tr.backward()
        optimizer.step()
    # model.load_state_dict(torch.load(model_path, weights_only=True))
    # model.eval()
    return model

# def testModel(model, test_list):
#     print("\nTesting...")
#     test = torch.Tensor(test_list["A"].tolist()).to(torch.int)
#     # test = test.to(DEVICE)
#     test_sp = test_list["Score"].tolist()
#
#     batch_size = 16
#     batch_start = 0
#     ind_ts = 0
#     batch_end = (batch_size * ind_ts) + batch_size
#
#     all_test_pred = []
#     test_len = len(test_sp)
#     itrs = math.ceil(test_len/batch_size)
#
#     test_batch = torch.Tensor(test).to(torch.int)
#     test_batch = test_batch.to(DEVICE)
#
#     test_pred = model(test_batch)
#
#     test_pred = test_pred.to(torch.device("cpu"))
#     test_pred = test_pred.tolist()
#
#     MAEs = []
#     for i in range(len(test_sp)):
#         MAEs.append(abs(test_sp[i]-test_pred[i]))
#     MAE = sum(MAEs)/len(MAEs)
#     MdAE = np.median(MAEs)
#
#     pearsons = scipy.stats.pearsonr(test_pred, test_sp)[0]
#
#     test_pred.sort()
#     test_sp.sort()
#     spearmans = scipy.stats.spearmanr(test_sp, test_pred).statistic
#
#     return MAE, MdAE, pearsons, spearmans


def testModel(model, test_list):
    # print("\nTesting...")
    test = torch.Tensor(test_list["A"].tolist()).to(torch.int)
    # test = test.to(DEVICE)
    test_sp = test_list["Label"].tolist()

    batch_size = 16
    batch_start = 0
    ind_ts = 0
    batch_end = (batch_size * ind_ts) + batch_size

    all_test_pred = []
    test_len = len(test_sp)
    itrs = math.ceil(test_len/batch_size)

    for i in range(itrs):
        if batch_end==test_len:
            break
        batch_start = ind_ts * batch_size
        batch_end = batch_start + batch_size
        if batch_end >= test_len:
            batch_end = test_len
            ind_ts = 0
        test_batch = test[batch_start:batch_end]
        ind_ts += 1

        # Formatting
        test_batch = torch.Tensor(test_batch).to(torch.int)
        test_batch = test_batch.to(DEVICE)

        test_pred = model(test_batch)

        test_pred = test_pred.to(torch.device("cpu"))
        test_pred = test_pred.tolist()
        if type(test_pred) is list:
            all_test_pred.extend(test_pred)
        else:
            all_test_pred.append(test_pred)

    MAEs = []
    for i in range(len(test_sp)):
        MAEs.append(abs(test_sp[i]-all_test_pred[i]))
    MAE = sum(MAEs)/len(MAEs)
    MdAE = np.median(MAEs)

    pearsons = scipy.stats.pearsonr(all_test_pred, test_sp)[0]

    all_test_pred.sort()
    test_sp.sort()
    all_test_pred[all_test_pred==np.inf]=0
    # all_test_pred[np.isnan(all_test_pred)]=0
    spearmans = scipy.stats.spearmanr(test_sp, all_test_pred).statistic

    return MAE, MdAE, pearsons, spearmans

def experiment(dataset):
    print(dataset)
    train, val, test = loadData(dataset, num_to_add=1)
    model = trainModel(dataset, train, val, test)
    print("\n\nTesting...")
    MAE, MdAE, pearsons, spearmans = testModel(model, test)
    print(dataset, MAE, MdAE, pearsons, spearmans)
    print("\n\n")
    return {"Data": dataset, "Pearson's coefficient": pearsons, "Spearman's coefficient": spearmans, "MAE": MAE, "MdAE": MdAE}


datas = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement", "duracloud", "jirasoftware",
         "mesos", "moodle", "mule", "mulestudio", "springxd", "talenddataquality", "talendesb", "titanium", "usergrid"]

# experiment("clover")
results = []
for d in datas:
    results.append(experiment(d))
print("\n\n")
results = pd.DataFrame(results)
results.to_csv("../../Results/ISP_GPT2_model_Regression.csv", index=False)
print(results)
print("Average results:")
print("Pearson's coefficient:", sum(results["Pearson's coefficient"].tolist())/len(results["Pearson's coefficient"]))
print("Spearman's coefficient:", sum(results["Spearman's coefficient"].tolist())/len(results["Spearman's coefficient"]))
print("MAE:", sum(results["MAE"].tolist())/len(results["MAE"]))
print("MdAE:", sum(results["MdAE"].tolist())/len(results["MdAE"]))












