## Image Classification Optical Front-End

# Project Description
To be updated.

# Folder Structure
This section will give basic descriptions of the software workflow and what is stored where.
References [this](https://theaisummer.com/best-practices-deep-learning-code/) article.
Further [best practices](https://neptune.ai/blog/how-to-organize-deep-learning-projects-best-practices).

- configs: in configs we define every single thing that can be configurable and can be changed in the future. Good examples are training hyperparameters, folder paths, the model architecture, metrics, flags.
- dataloader: is quite self-explanatory. All the data loading and data preprocessing classes and functions live here.
- evaluation: is a collection of code that aims to evaluate the performance and accuracy of our model.
- executor: in this folder, we usually have all the functions and scripts that train the model or use it to predict something in different environments. And by different environments I mean: executors for GPUs, executors for distributed systems. This package is our connection with the outer world and it’s what our “main.py” will use.
- model: contains the actual deep learning code (we talk about tensorflow, pytorch etc)
- utils: utilities functions that are used in more than one places and everything that don’t fall in on the above come here.

![This image](https://i0.wp.com/neptune.ai/wp-content/uploads/2022/10/DL-project-directory.png) also gives a good overview of what forms folder structure for deep learning code can take.