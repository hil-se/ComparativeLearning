import numpy as np
import scipy
import os
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVR
import pandas as pd
import tensorflow as tf
from pdb import set_trace

def loadData(dataName="appceleratorstudio", datatype="train"):
    path = "../Data/GPT2SP Data/Split/"
    df = pd.read_csv(path+dataName+"_"+datatype+".csv")
    return df

model = SentenceTransformer("all-MiniLM-L6-v2")

def process(dataName="appceleratorstudio", labelName="Storypoint"):
    train = loadData(dataName=dataName, datatype="train")
    valid = loadData(dataName=dataName, datatype="val")
    test = loadData(dataName=dataName, datatype="test")

    train_list_df = train.sample(frac=1)

    # train_list = []
    # for indexA, rowA in train_list_df.iterrows():
    #     text = rowA["Issue"]
    #     text = text.replace("\n", " ")
    #     text = model.encode(text)
    #     train_list.append({"A": text, "Score": rowA[labelName]})
    # train_list = pd.DataFrame(train_list)
    embeddings = model.encode(train_list_df["Issue"])
    train_list = pd.DataFrame([{"A": embeddings[i], "Score": train_list_df[labelName][i]} for i in range(len(train))])
    # train_list = pd.DataFrame({"A": embeddings, "Score": train_list_df[labelName]})

    val_list_df = valid.sample(frac=1)

    embeddings = model.encode(val_list_df["Issue"])
    val_list = pd.DataFrame([{"A": embeddings[i], "Score": val_list_df[labelName][i]} for i in range(len(valid))])
    # val_list = pd.DataFrame({"A": embeddings, "Score": val_list_df[labelName]})

    test_list_df = test.sample(frac=1)
    embeddings = model.encode(test_list_df["Issue"])
    test_list = pd.DataFrame([{"A": embeddings[i], "Score": test_list_df[labelName][i]} for i in range(len(test))])
    # test_list = pd.DataFrame({"A": embeddings, "Score": test_list_df[labelName]})

    return train_list, val_list, test_list


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),

        # tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.3),
        #
        # tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.2),
        #
        # tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1, activation="linear")
    ])

    # initial_learning_rate = 0.001
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate, decay_steps=5000, alpha=0.0001
    # )
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer='adam',
        loss="mae",
        # loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )

    return model

def train_and_test(dataname):

    train_list, val_list, test_list = process(dataname, "Storypoint")

    train_x = np.array(train_list["A"].tolist())
    train_y = np.array(train_list["Score"].tolist())
    val_x = np.array(val_list["A"].tolist())
    val_y = np.array(val_list["Score"].tolist())
    test_x = np.array(test_list["A"].tolist())
    test_y = test_list["Score"].tolist()

    model = build_model((train_x.shape[1]))

    checkpoint_path = "checkpoint/STD.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=150, verbose=1,
    #                                                   restore_best_weights=True)

    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=32,
        epochs=1000,
        callbacks=[reduce_lr, checkpoint],
        verbose=1
    )
    print("\nLoading best checkpoint model...")
    model.load_weights(checkpoint_path)
    # model.fit(train_x, train_y)
    preds_test = model.predict(test_x).flatten()
    preds_train = model.predict(train_x).flatten()


    pearsons_train = scipy.stats.pearsonr(preds_train, train_y)[0]
    spearmans_train = scipy.stats.spearmanr(preds_train, train_y).statistic
    pearsons_test = scipy.stats.pearsonr(preds_test, test_y)[0]
    spearmans_test = scipy.stats.spearmanr(preds_test, test_y).statistic

    return pearsons_train.item(), spearmans_train.item(), pearsons_test.item(), spearmans_test.item()

# print("Clover:", train_and_test("clover"))



# datas = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement", "duracloud", "jirasoftware",
#          "mesos", "moodle", "mule", "mulestudio", "springxd", "talenddataquality", "talendesb", "titanium", "usergrid"]
datas = ["jirasoftware"]

results = []
for d in datas:
    for i in range(20):
        r_train, rs_train, r_test, rs_test = train_and_test(d)
        print(d, r_train, rs_train, r_test, rs_test)
        results.append({"Data": d, "Pearson Train": r_train, "Spearman Train": rs_train, "Pearson Test": r_test, "Spearman Test": rs_test})
results = pd.DataFrame(results)
print(results)
results.to_csv("../Results/STD.csv", index=False)


