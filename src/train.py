import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train_model(test_size, random_state):
    # Load the Iris dataset
    iris = load_iris()

    # Separate features and target
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Create the K-Nearest Neighbors model
    model = KNeighborsClassifier(n_neighbors=3)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create outputs directory if it does not exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Plot and save confusion matrix
    plt.figure()
    plt.imshow(cm)
    plt.title("Iris Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")
    return accuracy, cm


def main():
    parser = argparse.ArgumentParser(
        description="Train a KNN model on the Iris dataset."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the data used for testing."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    train_model(
        test_size=args.test_size,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()