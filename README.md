# Exploring Automatic Differentiation with TensorFlow's GradientTape

## Problem Statement and Goal of Project

This project is a hands-on exploration of one of the most fundamental concepts in deep learning: automatic differentiation. The goal is to understand and demonstrate how TensorFlow's `tf.GradientTape` can be used to compute gradients—the cornerstone of training neural networks via backpropagation. This notebook serves as a practical guide to the mechanics of gradient computation, moving from simple mathematical functions to more complex scenarios involving multiple variables and persistent tapes.

## Solution Approach

The project systematically explores the functionalities of `tf.GradientTape` through a series of clear, commented examples:

1.  **Basic Gradient Calculation**: The notebook starts by computing the gradient of a simple function, $y = x^2$, and verifying that the result ($2x$) matches the analytical solution.
2.  **Trainable vs. Non-Trainable Tensors**: It demonstrates the difference between using `tf.Variable` (which is automatically "watched" by the tape) and `tf.constant` (which is not). It then shows how to explicitly use `tape.watch()` to compute gradients with respect to a constant tensor.
3.  **Complex Functions and Visualization**: The gradient of the sigmoid function is computed over a range of inputs, and the results are plotted using Matplotlib to visualize the relationship between the function and its derivative.
4.  **Gradients with Multiple Variables**: The project extends the concept to a simple linear model ($y = xW + b$), calculating the gradients of a loss function with respect to multiple trainable variables (weights `W` and bias `b`).
5.  **Persistent Tapes**: Finally, it showcases the use of a `persistent=True` GradientTape to compute multiple gradients from the same set of recorded operations, which is useful for more advanced models or for debugging purposes.

## Technologies & Libraries

  * **TensorFlow**: For all tensor operations and automatic differentiation using `tf.GradientTape`.
  * **Matplotlib**: For visualizing the sigmoid function and its gradient.
  * **Jupyter Notebook**: For interactive development and clear presentation of the concepts.

## Description about Dataset

This project does not use a formal dataset. Instead, it relies on programmatically generated `tf.Tensor` objects (variables and constants) to clearly and concisely demonstrate the mathematical principles of gradient computation without the overhead of data loading and preprocessing.

## Installation & Execution Guide

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install tensorflow matplotlib
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook gradients.ipynb
    ```

## Key Results / Performance

This project was focused on demonstrating a core machine learning concept rather than training a model for performance. The key results are the successful computations of gradients, which were verified against known analytical solutions.

For example, the gradient of $y = x^2$ at $x = 3.0$ was correctly computed as **6.0**. The notebook also successfully produced the characteristic bell-shaped curve for the derivative of the sigmoid function.

## Screenshots / Sample Output

**Visualization of the Sigmoid Function and its Gradient**

The plot below clearly illustrates the relationship between the sigmoid activation function and its derivative, computed using `tf.GradientTape`.

## Additional Learnings / Reflections

This project was a deep dive into the mechanics of backpropagation and provided a solid foundation for understanding how neural networks learn.

  * **Understanding `tf.GradientTape`**: I gained a practical and intuitive understanding of how the GradientTape API works, including the importance of "watching" variables and the flexibility offered by persistent tapes.
  * **Trainable Variables**: The distinction between `tf.Variable` and `tf.constant` in the context of automatic differentiation is now much clearer. This knowledge is essential for building custom layers and models where I need to define which parameters should be trained.
  * **Foundation for Custom Training**: This exercise is a direct prerequisite for creating custom training loops. By mastering gradient computation, I can now move beyond the high-level `model.fit()` API to implement more complex training logic for advanced architectures like GANs or Siamese Networks.

-----

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

## 👤 Author

## Mehran Asgari

## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).

## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

-----

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.