import argparse
import pickle
import numpy as np


# Main function to load model, predict and evaluate performance
def predict_results(model_path, input_X_path, output_dir):
    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load training data
    train_X = np.loadtxt(input_X_path)

    # Make predictions
    proba = model.predict_proba(train_X)
    y_scores = np.max(proba, axis=1)  # set max confidence（score）
    y_preds = model.classes_[np.argmax(proba, axis=1)]  # get label（label）

    # concate confidence and label
    output = np.column_stack((y_preds, y_scores))

    # save the concated result into a txt file
    np.savetxt(f'{output_dir}/results.txt', output, fmt='%d\t%.6f', header='Score\tPred', comments='')

# Entry point: use command-line arguments to run the evaluation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained model.')

    parser.add_argument('--model_path', type=str, required=False, default='./weights/RandomForestClassifier_weight.pkl',
                        help='Path to the trained model file (.pkl)')
    parser.add_argument('--input_X', type=str, required=False, default='./data/test_X.txt',
                        help='Path to the scaled training feature data (.txt)')
    parser.add_argument('--output_dir', type=str, required=False, default='',
                        help='Path to the output the results')

    args = parser.parse_args()

    predict_results(
        model_path=args.model_path,
        input_X_path=args.input_X,
        output_dir=args.output_dir
    )
