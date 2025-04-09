from fraud_detection.data_loader import DataLoader
from fraud_detection.preprocessing import Preprocessor
from fraud_detection.model_builder import ModelBuilder
from fraud_detection.trainer import Trainer

if __name__ == '__main__':
    file_path = 'data/processed/creditcard_balanced.csv'

    # Carregamento dos dados
    loader = DataLoader(file_path)
    data = loader.load()

    # Pré-processamento
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.transform(data)

    # Construção do modelo
    builder = ModelBuilder()
    model = builder.build(input_dim=X_train.shape[1])

    # Treinamento e avaliação
    trainer = Trainer(model)
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
