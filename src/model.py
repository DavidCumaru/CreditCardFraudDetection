import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class FraudDetectionModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print(self.data.head())

    def preprocess(self):
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']

        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

    def build_model(self):
        self.model = Sequential([
            Dense(32, input_dim=self.X_train.shape[1], activation='relu'),
            Dropout(0.5),
            Dense(16, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer=Adam(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[es],
            verbose=1
        )

    def evaluate(self):
        y_pred = (self.model.predict(self.X_test) > 0.5).astype('int32')

        print(classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        plt.show()