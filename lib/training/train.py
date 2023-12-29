from training import ModelTrainer

def main():
    # Create an instance of the ModelTrainer class
    trainer = ModelTrainer()

    # Load the training data
    trainer.load_data()

    # Preprocess the data
    trainer.preprocess_data()

    # Build the model
    trainer.build_model()

    # Train the model
    trainer.train_model()

    # Evaluate the model
    trainer.evaluate_model()

if __name__ == "__main__":
    main()
