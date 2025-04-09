from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    def __init__(self, model):
        self.model = model
        self.history = None

    def train(self, X_train, y_train):
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            callbacks=[es],
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        y_pred = (self.model.predict(X_test) > 0.5).astype('int32')

        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        plt.show()

        # Plot do histórico
        if self.history:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Loss')
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy')
            plt.legend()

            if 'auc' in self.history.history:
                plt.subplot(1, 3, 3)
                plt.plot(self.history.history['auc'], label='Train AUC')
                plt.plot(self.history.history['val_auc'], label='Validation AUC')
                plt.title('AUC')
                plt.legend()

            plt.tight_layout()
            plt.show()
