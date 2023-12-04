# Sequential RecSys: Open-Source Library Development and Algorithms Evaluation
Welcome to the Sequential RecSys project! This open-source library is dedicated to the implementation and evaluation of three recommendation algorithms on three diverse datasets: MovieLens 1M, Amazon Beauty, and Steam.


## Datasets
Dataset Variety: Our library supports three distinct datasets for algorithm evaluation:

- MovieLens 1M: A well-known movie rating dataset.
- Amazon Beauty: A dataset containing product reviews and ratings.
- Steam Dataset: Data from the popular gaming platform Steam, including user interactions with games.


## Three Recommendation Algorithms:
We have implemented three state-of-the-art recommendation algorithms, each designed to cater to different recommendation scenarios.

To train and run the algorithms, use the following commands:

- **SASRec Algorithm:**
  ```python
  python train_sasrec.py --dataset=path/to/the/dataset --train_dir=default
  ```

- **SR-GNN Algorithm:**
  ```python
  python train_srgnn.py --dataset_path=path/to/the/dataset
  ```

- **CASER Algorithm:**
  ```python
  python train_caser.py
  ```
  ## Contact

If you have any questions, suggestions, or need assistance, please feel free to contact us:

- Email : areddy21@gmu.edu
