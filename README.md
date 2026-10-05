# Machine learning and search algorithms from scratch

Classifiers and optimisation algorithms implemented with NumPy only, without ML libraries, as part of IN4050 (Introduction to Artificial Intelligence and Machine Learning) at the University of Oslo. The goal was to understand how the algorithms work, so everything from gradient descent to early stopping and genetic operators is written by hand. scikit-learn is used only to generate the synthetic dataset.

The repository has two parts:

| Folder | Topic | Algorithms |
|---|---|---|
| [`classifiers/`](classifiers/) | Supervised classification | Linear regression classifier, logistic regression, one-vs-rest, multinomial (softmax) regression |
| [`tsp/`](tsp/) | Travelling salesman problem | Exhaustive search, hill climbing, genetic algorithm |

## Classifiers

### What is implemented

- **Linear regression classifier** (`linear_regression.py`): least-squares loss trained with batch gradient descent, thresholded at 0.5.
- **Logistic regression** (`logistic_regression.py`): sigmoid output, binary cross-entropy loss, and early stopping on validation loss with a relative-improvement tolerance and patience.
- **One-vs-rest** (`one_vs_rest.py`): one logistic regression model per class, each tuned separately. The prediction is the class whose model gives the highest probability.
- **Multinomial logistic regression** (`multinomial.py`): softmax output and categorical cross-entropy, with the same early-stopping scheme.
- **Hyperparameter tuning**: grid search over learning rate, tolerance, and number of epochs.

### Data

A synthetic 2D dataset of 2,000 points in five overlapping clusters, split 1,000 / 500 / 500 into training, validation, and test sets. A binary version merges classes 0–2 and 3–4. Neither version is linearly separable, which is the point: it shows how far linear models get and where they stop.

### Results

Validation accuracy after tuning:

| Model | Task | Validation accuracy |
|---|---|---|
| Linear regression classifier | Binary | 0.760 |
| Logistic regression | Binary | 0.762 |
| One-vs-rest (logistic regression) | 5 classes | 0.748 |
| Multinomial logistic regression | 5 classes | 0.792 |

On the binary task, both linear models level off at about 0.76, which is roughly the ceiling for a linear decision boundary on this data. On the five-class task, the multinomial model outperformed one-vs-rest in this run. A likely reason is that the one-vs-rest models are trained independently, so their probability scores aren't calibrated against each other, while softmax normalises across all classes jointly.

The notebook also shows the effect of feature scaling (z-score normalisation) on the number of epochs gradient descent needs, plus training and validation loss curves.

## Travelling salesman problem

The task is to find the shortest round trip through up to 24 European cities, given a distance matrix in kilometres.

### Exhaustive search

The search checks every permutation and is exact, but its running time grows factorially. It finds the optimum for 6 cities (5,018.81 km) and 10 cities (7,486.31 km). For 24 cities there are about 1.3 × 10²² distinct tours, which rules out exhaustive search.

### Hill climbing

The search starts from a random tour and repeatedly applies the best-improving swap of two cities until no swap improves it. It is repeated with 20 random restarts.

### Genetic algorithm

- **Representation:** a permutation of city indices
- **Selection:** tournament selection (size 3)
- **Crossover:** order crossover (OX), which preserves valid permutations
- **Mutation:** inversion mutation (reverses a random segment), applied with 30 % probability
- **Elitism:** the five best individuals carry over to each new generation
- **Parameters:** population 300, 300 generations, 20 independent runs

### Results (24 cities, 20 runs each)

| Algorithm | Best (km) | Mean (km) | Worst (km) | Std. dev. (km) |
|---|---|---|---|---|
| Hill climbing | 12,587.56 | 14,819.73 | 17,312.15 | 1,213.78 |
| Genetic algorithm | 12,287.07 | 12,404.29 | 12,956.66 | 172.13 |

The genetic algorithm's average run beats hill climbing's best run, and its spread is about seven times smaller. Hill climbing gets stuck in local optima and depends heavily on the starting tour. The genetic algorithm escapes them by recombining good partial routes from different individuals. On 10 cities, hill climbing's best run matched the exhaustive optimum (7,486.31 km).

## Running the code

Requires Python 3.12 or newer, because the notebooks use f-string syntax introduced in 3.12.

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `classifiers/training.ipynb` or `tsp/tsp.ipynb`. Run the notebooks from inside their own folder, since they import local modules and read data files with relative paths.

## Limitations

- **Evaluation:** hyperparameters are selected on the validation set, and the accuracies above are measured on that same set, so they are optimistic. The test set is held out but not yet used for a final evaluation.
- **Reproducibility:** the TSP experiments use Python's `random` module, which isn't seeded, so exact numbers vary between runs.
- **Structure:** most of the TSP code lives in the notebook rather than in importable modules.

## Attribution

Developed as coursework for IN4050 at the University of Oslo. The course provided the dataset generation setup, the plotting helpers (`plot_decision_regions`, `plot_plan`), `add_bias`, the city coordinates, the distance matrix, and the map image. All model and algorithm implementations are my own.
