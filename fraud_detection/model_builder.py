from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    def build(self, input_dim):
        model = Sequential([
            Dense(32, input_dim=input_dim, activation='relu'),
            Dropout(0.5),
            Dense(16, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])  # Pode adicionar 'AUC' aqui
        return model
