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

        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1, activation=None)
    ])

    # initial_learning_rate = 0.001
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate, decay_steps=5000, alpha=0.0001
    # )
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    return model

class ComparativeModel(tf.keras.Model):
    def __init__(self, encoder, **kwargs):
        super(ComparativeModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, trainable=True):
        encodings_A = self.encoder(features["A"], training=trainable)
        encodings_B = self.encoder(features["B"], training=trainable)
        return tf.subtract(encodings_A, encodings_B)

    def compute_loss(self, y, diff):
        y = tf.cast(y, tf.float32)
        loss = tf.reduce_mean(tf.math.maximum(0.0, 1.0 - (y * tf.squeeze(diff))))
        return loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            diff = self(x)
            loss = self.compute_loss(y, diff)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # def test_step(self, features):
    #     encodings_A, encodings_B = self(features)
    #     loss = self.compute_loss(encodings_A, encodings_B, features["Label"])
    #     self.loss_tracker.update_state(loss)
    #     return {"loss": self.loss_tracker.result()}

    def predict(self, X):
        """Predicts preference between two items."""
        return np.array(self.encoder(np.array(X.tolist())))

def generate_comparative_judgments(train_list, N=1):
    m = len(train_list)
    features = {"A": [], "B": [], "Label": []}
    for i in range(m):
        n = 0
        while n < N:
            j = np.random.randint(0, m)
            if train_list["Score"][i] > train_list["Score"][j]:
                features["A"].append(train_list["A"][i])
                features["B"].append(train_list["A"][j])
                features["Label"].append(1.0)
                # features.append({"A": train_list["A"][i], "B": train_list["A"][j], "Label": 1.0})
                n += 1
            elif train_list["Score"][i] < train_list["Score"][j]:
                features["A"].append(train_list["A"][i])
                features["B"].append(train_list["A"][j])
                features["Label"].append(-1.0)
                # features.append({"A": train_list["A"][i], "B": train_list["A"][j], "Label": -1.0})
                n += 1
    features = {key: np.array(features[key]) for key in features}
    return features

def train_and_test(dataname, N=1):

    train_list, val_list, test_list = process(dataname, "Storypoint")
    train_list = pd.concat([train_list, val_list], axis=0)
    train_list = train_list.sample(frac=1.0)
    train_list.index = range(len(train_list))
    features = generate_comparative_judgments(train_list, N=N)

    train_x = np.array(train_list["A"].tolist())
    train_y = np.array(train_list["Score"].tolist())
    test_x = np.array(test_list["A"].tolist())
    test_y = test_list["Score"].tolist()

    encoder = build_model((train_x.shape[1]))
    de = ComparativeModel(encoder=encoder)

    checkpoint_path = "checkpoint/STD_comp.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    de.compile(optimizer="SGD", loss=tf.keras.losses.Hinge())
    # de.compile(optimizer="SGD")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1, )

    # Train model
    history = de.fit(features, features["Label"], epochs=500, batch_size=32, callbacks=[reduce_lr, checkpoint], verbose=1)
    print("\nLoading best checkpoint model...")
    de.load_weights(checkpoint_path)

    preds_test = de.predict(test_x).flatten()
    preds_train = de.predict(train_x).flatten()


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
    for n in [1,2,3,4,5,10]:
        for i in range(20):
            r_train, rs_train, r_test, rs_test = train_and_test(d, N=n)
            print(d, r_train, rs_train, r_test, rs_test)
            results.append({"Data": d, "N": n, "Pearson Train": r_train, "Spearman Train": rs_train, "Pearson Test": r_test, "Spearman Test": rs_test})
results = pd.DataFrame(results)
print(results)
results.to_csv("../Results/STD_comp_noval.csv", index=False)


