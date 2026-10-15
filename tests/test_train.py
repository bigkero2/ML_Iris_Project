from src.train import train_model

def test_train_model():
    accuracy, confusion_matrix = train_model(
        test_size=0.2,
        random_state=42
    )
    
    # The model should achieve at least 90% accuracy
    assert accuracy >= 0.90
    
    # Iris has three classes, so the confusion matrix should be 3 x 3
    assert confusion_matrix.shape == (3,3)