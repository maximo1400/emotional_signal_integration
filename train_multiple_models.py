import itertools
import json
from pathlib import Path

from config_loader import get_config
from L2_emot_state_estimation.classifier import ClassifierManager
from L2_emot_state_estimation.run import _collect_training_data


def train_with_params(classifier_name: str, hyperparams: dict, output_dir: Path):
    config = get_config([
        "feather_file_path",
        "POW_COLUMNS",
        "num_classes",
        "class_balancing",
        "data_split_method",
        "data_split_parameters",
    ])

    feather_path = config["feather_file_path"]

    pow_vectors, labels, people = _collect_training_data(
        feather_path,
        config["POW_COLUMNS"],
        config["num_classes"],
    )

    classifier_input_len = len(pow_vectors[0])
    classifier_manager = ClassifierManager(classifier_input_len)

    model_filename = f"{classifier_name}.joblib"
    model_path = output_dir / model_filename

    print("\n==============================================")
    print(f"Training {classifier_name}")
    print(f"Params: {hyperparams}")
    print("==============================================")

    # Train the model
    classifier_manager.train(
        classifier_name,
        pow_vectors,
        labels,
        str(model_path),
        hyperparams,
        config["num_classes"],
        config["class_balancing"],
        config["data_split_method"],
        config["data_split_parameters"],
        people,
    )

    # Save hyperparams alongside the model
    params_path = output_dir / "hyperparams.json"
    with open(params_path, "w") as f:
        json.dump(hyperparams, f, indent=4)

    return model_path


def main():
    # Define a simple parameter grid
    param_grids = {
        "knn": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]},
        "random_forest": {"n_estimators": [50, 100], "max_depth": [5, 10]},
        "svm": {"C": [0.1, 1.0], "kernel": ["rbf", "linear"], "probability": [True]},
    }

    # The special folder where all these models will be saved
    special_folder = Path("output_data/grid_search_models")
    special_folder.mkdir(parents=True, exist_ok=True)

    for clf_name, grid in param_grids.items():
        keys, values = zip(*grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for idx, params in enumerate(combinations):
            # Create a dedicated sub-folder for this specific model configuration
            model_sub_folder = special_folder / f"{clf_name}_config_{idx}"
            model_sub_folder.mkdir(parents=True, exist_ok=True)

            try:
                train_with_params(clf_name, params, model_sub_folder)
                print(
                    f"Successfully saved {clf_name} (Config {idx}) to {model_sub_folder}"
                )
            except Exception as e:
                print(f"Error training {clf_name} (Config {idx}): {e}")


if __name__ == "__main__":
    main()
