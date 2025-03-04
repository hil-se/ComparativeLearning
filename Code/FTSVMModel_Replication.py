import numpy as np
import scipy
import fasttext as ft
from sklearn.svm import SVR
import pandas as pd

def loadData(dataName="appceleratorstudio", datatype="train"):
    path = "../Data/GPT2SP Data/Split/"
    df = pd.read_csv(path+dataName+"_"+datatype+".csv")
    return df

lmodel = ft.load_model("../../cc.en.300.bin")
print("Loading FastText model...")

def process(dataName="appceleratorstudio", labelName="Storypoint"):
    train = loadData(dataName=dataName, datatype="train")
    valid = loadData(dataName=dataName, datatype="val")
    test = loadData(dataName=dataName, datatype="test")

    train_len = len(train.index)
    test_len = len(test.index)
    val_len = len(valid.index)

    train_list_df = train.sample(frac=1)
    train_list = []

    for indexA, rowA in train_list_df.iterrows():
        text = rowA["Issue"]
        text = text.replace("\n", " ")
        text = lmodel.get_sentence_vector(text)
        train_list.append({"A": text, "Score": rowA[labelName]})
    train_list = pd.DataFrame(train_list)

    val_list_df = valid.sample(frac=1)
    val_list = []

    for indexA, rowA in val_list_df.iterrows():
        text = rowA["Issue"]
        text = text.replace("\n", " ")
        text = lmodel.get_sentence_vector(text)
        val_list.append({"A": text, "Score": rowA[labelName]})
    val_list = pd.DataFrame(val_list)

    test_list = []
    for indexA, rowA in test.iterrows():
        text = rowA["Issue"]
        text = text.replace("\n", " ")
        text = lmodel.get_sentence_vector(text)
        test_list.append({"A": text, "Score": rowA[labelName]})

    test_list = pd.DataFrame(test_list)

    return train_list, val_list, test_list

def train_and_test(dataname):
    model = SVR()
    # print(dataname)
    train_list, val_list, test_list = process(dataname, "Storypoint")
    train_list = pd.concat([train_list, val_list], axis=0)
    train_list = train_list.sample(frac=1.0)

    train_x = np.array(train_list["A"].tolist())
    train_y = np.array(train_list["Score"].tolist())
    test_x = np.array(test_list["A"].tolist())
    trues = test_list["Score"].tolist()

    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    preds = np.squeeze(preds).tolist()

    MAEs = []
    for i in range(len(trues)):
        MAEs.append(abs(trues[i] - preds[i]))
    MAE = sum(MAEs) / len(MAEs)
    MdAE = np.median(MAEs)

    pearsons = scipy.stats.pearsonr(preds, trues)[0]

    preds.sort()
    trues.sort()
    preds[preds == np.inf] = 0
    spearmans = scipy.stats.spearmanr(trues, preds).statistic

    return MAE, MdAE.item(), pearsons.item(), spearmans.item()

# print("Clover:", train_and_test("clover"))



datas = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement", "duracloud", "jirasoftware",
         "mesos", "moodle", "mule", "mulestudio", "springxd", "talenddataquality", "talendesb", "titanium", "usergrid"]

results = []
for d in datas:
    mae, mdae, p, sp = train_and_test(d)
    print(d, p, sp, mae, mdae)
    results.append({"Data": d, "Pearson's": p, "Spearman's": sp, "MAE": mae, "MdAE": mdae})
results = pd.DataFrame(results)
print(results)
results.to_csv("../Results/FTSVM.csv", index=False)


