# Flexible Multiple Linear Regression

## Overview

This project explores the calculus foundations of machine learning using linear regression in JAX, but with gradient analytically derived and manually computed. The effects of advanced techniques are explored like *$L_p$-regularization*, *dropout*, *early stopping* and different weight-initialization methods like He, LeCun, random and zero. This project also exposes the need for advanced first-order gradient methods, like momentum, RMS propagation, Adagrad, Adam and so on.

## The Model 

The model of the data is assumed to be:

$$ 
y_i = \tilde{\beta_0} + \sum_{j=1}^{q-1}{x_{ij}\tilde{\beta_j}} + \epsilon_i, \quad i \in \{1,2,...,n\}
$$

Where: 
- $y_i$ is the label for the $i^{\text{th}}$ data point,
- $x_{i,j}$ is the $j^{\text{th}}$ feature for the $i^{\text{th}}$ data point,
- $\tilde{\beta_j}$ is the coefficient for the $j^{\text{th}}$ feature,
- and $\epsilon_i$ is assumed to be Normal and independent and identically distributed (i.i.d.) with mean $0$ and variance $\sigma^2$.<br>
This is because it is presumed that the data provided will come with noise and such noise is modelled using $\epsilon_i$ for each data point.

This model is identical to the following:

$$
\boldsymbol{Y} = \boldsymbol{X}\boldsymbol{\tilde{\beta}} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim N(\boldsymbol{0}, \Sigma)
$$

Where:
- $\boldsymbol{Y} \in \mathbb{R}^{n \times 1}$ is the response variable; 
- $\boldsymbol{X} \in \mathbb{R}^{n \times q}$ is the full-rank design matrix; 
- $\boldsymbol{\tilde{\beta}} \in \mathbb{R}^{q \times 1}$ is the vector of true coefficients, including $\beta_0$; 
- $\boldsymbol{\epsilon} \in \mathbb{R}^{n \times 1}$ is the vector of errors;
- and $\Sigma \in \mathbb{R}^{n \times n}$ is the variance-covariance matrix of $\boldsymbol{\epsilon}$.

It is worth noting that because $\epsilon_i$ is i.i.d., the covariance matrix $\Sigma$ is diagonal:

$$
(\Sigma)_{ij} = 
\begin{cases}
\sigma^2, \text{ if } i=j\\
0, \text{ if } i \ne j.
\end{cases}
$$

i.e. $\Sigma = \sigma^2\boldsymbol{I}_n$, where $\boldsymbol{I}_n$ is the $n\times n$ identity matrix.

Under these assumptions, predictions can be made using the following formula:
$$
\boldsymbol{\hat{Y}} = \boldsymbol{X\beta}
$$
- $\boldsymbol{\tilde{Y}} \in \mathbb{R}^{n \times 1}$ is the 'best' vector of fitted values.
- $\boldsymbol{\hat{\beta}} \in \mathbb{R}^{q \times 1}$ is the 'best' vector of coefficients.
The quotations marks are used to question if there really is a 'best' way of optimizing the model. However, under the assumptions of the model, it turns out there is actually a 'best' way to find the vector of coefficients and in turn, the best vector of fitted values.

## The Optimization

The optimizer step used is batch gradient descent (BGD).

In order to define the optimizer step, an appropriate loss is needed. The loss function chosen is the $L_2$ loss function (which is in this case defined to be $\text{RSS}(\boldsymbol{\beta})$, the *residual sum of squares*) with $L_p$ regularization, with regularization parameter $\lambda \ge 0$ and $p \ge 1$, defined to be $J_{\lambda}({\boldsymbol{\beta}})$. <br>
This is chosen over $L_1$ loss due the smoothness of the $L_2$ loss function and the findings of the experiments which will be discussed later.

The step is summarized as:
$$
\boldsymbol{\beta}_{k+1} :=\boldsymbol{\beta}_k - \alpha \boldsymbol{\nabla}_{\boldsymbol{\beta}_k} J_{\lambda}(\boldsymbol{\beta}_k), 
$$

$$
J_{\lambda}(\boldsymbol{\beta}) = \text{RSS}(\boldsymbol{\beta}) + \lambda\sum_{j=1}^{q-1}|\beta_j|^{p}
$$
with:
$$
\text{RSS}(\boldsymbol{\beta}) = (\boldsymbol{Y} - \boldsymbol{X\beta})^T(\boldsymbol{Y} - \boldsymbol{X\beta})
$$
The coefficient vector $\boldsymbol{\beta}_k$ is the coefficient vector at $k^{\text{th}}$ epoch as this step is ran each epoch to update the parameters and which will eventually converge to a neighbourhood near the true parameters.

The intuition behind this optimization step is that $-\boldsymbol{\nabla}_{\boldsymbol{\beta}} J_{\lambda}(\boldsymbol{\beta})$ represents the steepest descent at the point $\boldsymbol{\beta} \in \mathbb{R}^{q \times 1}$, and $|\boldsymbol{\nabla}_{\boldsymbol{\beta}} J_{\lambda}(\boldsymbol{\beta})|$ represents the jump along the descending slope. <br>
Therefore, $\alpha \ge 0$ regulates the aforementioned jump, hence called the learning rate.

Finding an explicit expression of $\boldsymbol{\nabla}_{\boldsymbol{\beta}} J_{\lambda}(\boldsymbol{\beta})$:

$$
\begin{align*}
J_{\lambda}(\boldsymbol{\beta}) &= (\boldsymbol{Y} - \boldsymbol{X\beta})^T(\boldsymbol{Y} - \boldsymbol{X\beta}) + (\boldsymbol{X\beta})^T\boldsymbol{X\beta} + \lambda\sum_{j=1}^{q-1}|\beta_j|^{p}\\

&= \boldsymbol{Y^TY} - (\boldsymbol{X\beta})^T\boldsymbol{Y} - \boldsymbol{Y}^T\boldsymbol{X\beta} + \boldsymbol{\beta}^T\boldsymbol{X}^T\boldsymbol{X\beta} + \lambda\sum_{j=1}^{q-1}|\beta_j|^{p}\\

&= \boldsymbol{Y^TY} - 2\boldsymbol{\beta}^T\boldsymbol{X}^T\boldsymbol{Y} + \boldsymbol{\beta}^T\boldsymbol{X}^T\boldsymbol{X\beta} + \lambda\sum_{j=1}^{q-1}|\beta_j|^{p}
\end{align*}
$$

By the symmetry of the dot product, $(\boldsymbol{X\beta})^T\boldsymbol{Y} = \boldsymbol{Y}^T\boldsymbol{X\beta}$, hence the final expression. 

Now using matrix calculus, we can find an expression of $\boldsymbol{\nabla_{\beta}}J_{\lambda}(\boldsymbol{\beta})$:

$$
\begin{align*}
\boldsymbol{\nabla_{\beta}}J_{\lambda}(\boldsymbol{\beta}) &= -2\boldsymbol{X}^T\boldsymbol{Y} + 2\boldsymbol{X}^T\boldsymbol{X\beta} + \lambda \boldsymbol{\nabla_{\beta}} \sum_{j=1}^{q-1}|\beta_j|^{p}
\end{align*}
$$

Now to find $\boldsymbol{\nabla_{\beta}} \sum_{j=1}^{q-1}{|\beta_j|^{p}}$:

$$
\begin{align*}
\boldsymbol{\nabla_{\beta}} \sum_{j=1}^{q-1}{|\beta_j|^p}_i &= \boldsymbol{\nabla_{\beta}}\sum_{j=1}^{q-1}{|\beta_j|^p} \\
&= p\sum_{j=1}^{q-1}{\text{sgn}(\beta_j)|\beta_j|^{p-1}\boldsymbol{\hat{\text{e}}}_{j+1}} \\
&= p\left(0, \text{sgn}(\beta_1)|\beta_1|^{p-1}, \text{sgn}(\beta_2)|\beta_2|^{p-1}, ..., \text{sgn}(\beta_{q-1})|\beta_{q-1}|^{p-1} \right)^T
\end{align*} 
$$

The function $\text{sgn}:\mathbb{R}\rightarrow \{-1, 0, 1\}$, takes an input $x$ and outputs $1$ if $x>0$, $0$ if $x=0$ and $-1$ if $x<0$.

The term $\boldsymbol{\hat{\text{e}}_{k}}$ is the $k^{\text{th}}$ unit vector in the standard orthonormal basis $B_q = \{\boldsymbol{\hat{\text{e}}}_{1}, \boldsymbol{\hat{\text{e}}}_{2}, ..., \boldsymbol{\hat{\text{e}}}_{q}\}$ of the Euclidean space $\mathbb{\R}^q$.

Therefore:

$$
\begin{align*}
\boldsymbol{\nabla_{\beta}}J_{\lambda}(\boldsymbol{\beta}) &= 2\boldsymbol{X}^T(\boldsymbol{X\beta}-\boldsymbol{Y}) + \lambda p\sum_{j=1}^{q-1}{\text{sgn}(\beta_j)|\beta_j|^{p-1}\boldsymbol{\hat{\text{e}}}_{j+1}} \\
&= 2\boldsymbol{X}^T(\boldsymbol{\hat{Y}}-\boldsymbol{Y}) + \lambda p\sum_{j=1}^{q-1}{\text{sgn}(\beta_j)|\beta_j|^{p-1}\boldsymbol{\hat{\text{e}}}_{j+1}} \\
\newline
\end{align*}
$$

Thus completing the expression need to completed need to implement the gradient descent on the coefficients. 

This should ideally converge to a neighbourhood near the best fitted vector of coefficients which, under the assumptions of our model, has a closed form as previously mentioned. <br>
The solution to finding that set of coefficients is through this formula:
$$
\boldsymbol{\hat{\beta}} = \underset{\boldsymbol{\beta}}{\text{argmin}} \ \text{RSS}(\boldsymbol{\beta})
$$
Now let us find this closed form. Using what was found by finding gradient vector of the loss function $J_{\lambda}(\boldsymbol{\beta})$ we can immediately start by looking at the gradient of the residual sum of squares and setting it equal to zero:
$$
\begin{align*}
\boldsymbol{\nabla}_{\boldsymbol{\beta}}\text{RSS}(\boldsymbol{\beta}) |_{\boldsymbol{\beta} = \boldsymbol{\hat{\beta}}}&= 2\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{{\hat\beta}} - 2\boldsymbol{X}^T\boldsymbol{Y} = 0 \\
\\
\implies \boldsymbol{\hat{\beta}} &= (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{Y}
\end{align*}
$$

The inversion of the matrix $\boldsymbol{X}^T\boldsymbol{X}$ is only possible if $\boldsymbol{X}$ is of full rank. This means that the rank of the design matrix must be $\text{min}(n,q)$.<br>
Reasonably, you would not have more parameters than observations because of overfitting becomes very likely and so the rank of the design matrix is often assumed to be $q$.

This is indeed a minimum point because $\text{RSS}(\boldsymbol{\beta})$ is a quadratic equation with positive quadratic terms, therefore it will only have one turning point under the axes $\{\beta_0, \beta_1,...,\beta_{q-1}\}$ which will be a global minimum. <br>
More rigorously, the Hessian matrix of $\text{RSS}(\boldsymbol{\beta})$ is equal to $2\boldsymbol{X}^T\boldsymbol{X}$, which is positive-definite if the design matrix is full rank, therefore $\boldsymbol{\hat{\beta}}$ is the global minimum of $\text{RSS}(\boldsymbol{\beta})$.

## Structuring the Model

I employ Object-Oriented Programming, to keep the syntax analogous to scikit-learn. However, it is more flexible as it allows many more parameters that can adjust many regularization techniques, both explicit and implicit. Regularization techniques are used to prevent overfitting, which is often a result of the model memorizing the training data, rather than learning generalized patterns. This leads to very high performance on the training data but poor performance on unseen data i.e. validation and test data. Therefore, the techniques solve this by reducing the complexity of the model artificially to avoid learning the noise and improving generalization.

The class is called `LinearRegressionModel` with contains the following adjustable initialized attributes:
- `weights_init` - A string that allows the options of picking zero, random, Xavier, LeCun or He initialization for $\boldsymbol{\beta}_0$. For more information, check the following website:
[A Gentle Introduction To Weight Initialization for Neural Networks](https://wandb.ai/sauravmaheshkar/initialization/reports/A-Gentle-Introduction-To-Weight-Initialization-for-Neural-Networks--Vmlldzo2ODExMTg).
- `epochs` - Number of epochs.
- `learning_rate` - the learning rate, $\alpha$.
- `p` - Indicates which $L_p$ norm used in the explicit regularization technique. Does not include $L_{\infty}$.
- `lambda_` - The regularization coefficient, $\lambda$.
- `max_patience` - The maximum amount of epochs that can run if the loss on the validation data does not improve significantly.
- `dropout` - The expected proportion of zeroed feature data points. It's modelled as a multivariate Bernoulli variable of size $n$ with probability $(1-$`dropout`$)$. <br>
This can be important for reducing 
- `random_state` - The random seed.

The class also have the following methods:
- `fit` - Takes in the training data and validation data (optional) and trains the parameters on the data. Nothing is returned.
- `predict` - Only takes in the testing feature data `X_test` to predict the test labels.
- `plot_losses` - No arguments. Outputs the graph of training losses and validation losses on the same graph using the trained parameters. Therefore, this method is best used after fitting the model using `fit`.

## Experimentation

I have used three datasets from Kaggle as part of the experiment:
- [Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
- [Multiple Linear Regression Dataset](https://www.kaggle.com/datasets/hussainnasirkhan/multiple-linear-regression-dataset)
- [US Health Insurance Dataset](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset)

### Data Preprocessing

Three steps were taken when preprocessing data:
- one-hot encoding,
- normalizing,
- and splitting the data.

One-hot encoding was done first to make sure the `object`-type columns are represented numerically before normalizing the data. Normalizing the data was done by subtracting each data point by the mean of the column then dividing by the standard deviation of the same column. This ensures each column is of mean 0 and variance 1. This is necessary because if not normalized, the optimizer step becomes more sensitive to changes in the hyperparameters. Such is a problem because it would either lead to divergence or very slow learning to ensure convergence. Splitting the data is the standard procedure in retrieving the training, validation and testing data. The split chosen was 72% training, 8% validation and 20% testing, but these are hyperparameters which are subject to personal preference. The dataset I used for evaluation, diagnosis and configure the hyperparameters in `config.py` is [Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression), and I ran the other .csv files to make sure that this code (hopefully) works on other unseen custom datasets. With the specific configurations of the hyperparameters in `config.py` I was able to beat the performance of scikit-learn's built-in `sklearn.linear_model.LinearRegression` model on the test set of the aforementioned custom dataset.

### Testing the Assumption of Proposed Model

To see if the assumption of the proposed model was reasonable we need to look at the claims made:
- *Linearity* - The expected value of $\boldsymbol{Y}$ given $\boldsymbol{X}$ is linear. More specifically: $\text{E}(\boldsymbol{Y|X})=\boldsymbol{X\beta}$.
- *Normality of errors* - For each data point, $\epsilon_i \sim N(0,\sigma^2)$.
- *Independence of errors* - All errors are assumed to be i.i.d. or equivalently: $\text{Cov}(\epsilon_i,\epsilon_j)=0$ if $i \ne j$.
- *Homoscedasticity* - The errors have constant variance $\sigma^2$.
- *Full-Rank Design Matrix* - The design matrix $\boldsymbol{X}$ is of full rank. It is reasonable to assume full rank because a design matrix that is not full rank would contain a feature of the data that is simply a reflection of another feature(s) of the data. Mathematically, it would mean that at least one of the columns of the design matrix is a linear combination of the other columns in that same matrix.

A useful way to check the above assumptions of the model, is by looking at the following diagnostic plots:

<img src='assets/diagnostic_plots.png' alt='Diagnostic Plots' style='max-width: 600px; height: auto;'><br>

***Residual vs Fitted Values***:

The plot shows whether the residuals, $\boldsymbol{e} \, (= \boldsymbol{Y} - \boldsymbol{\hat{Y}})$, are independent to the $\boldsymbol{\hat{Y}}$, fitted values.<br>
This plot checks to see if the following assumptions are valid:
- *Normality of errors*
- *Independence of errors*
- *Homoscedasticity*
- *Full-Rank Design Matrix*

Before we see why, let's point out the relationship between the best fitted values and the response matrix: 
$$
\begin{align*}
\boldsymbol{\hat{Y}} &= \boldsymbol{X\hat{\beta}} \\
&= \boldsymbol{X}(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{Y} \\
\implies \boldsymbol{\hat{Y}} &= \boldsymbol{P}\boldsymbol{Y}
\end{align*}
$$
Where $\boldsymbol{P}$ is called the *hat matrix*.<br>
The formation of the hat matrix requires the assumption of *Full-Rank Design Matrix*.

It has two properties:
- Idempotence: $\boldsymbol{P}^2 = \boldsymbol{P}$
- Symmetry: $\boldsymbol{P}^T = \boldsymbol{P}$

With this information, let's begin to see the relationship between the best fitted values and the residuals:
$$
\begin{align*}
\text{Cov}(\boldsymbol{e}, \boldsymbol{\hat{Y}}) &= \text{Cov}((\boldsymbol{I}-\boldsymbol{P})\boldsymbol{Y}, \boldsymbol{P}\boldsymbol{Y}) \\
&= \text{Cov}(\boldsymbol{Y}, \boldsymbol{P}\boldsymbol{Y}) - \text{Cov}(\boldsymbol{P}\boldsymbol{Y}, \boldsymbol{P}\boldsymbol{Y}) \\
&= \text{Var}(\boldsymbol{Y})\boldsymbol{P}^T - \boldsymbol{P} \, \text{Var}(\boldsymbol{Y})\boldsymbol{P}^T \\
&= \sigma^2\boldsymbol{P}^T - \sigma^2\boldsymbol{P}\boldsymbol{P}^T \\
&= 0
\end{align*}
$$

The above calculations required the assumption *Independence of errors* and *Homoscedasticity* because $\text{Var}(\boldsymbol{Y})=\sigma^2\boldsymbol{I}_n$.

Furthermore, given the assumption of *normality of errors*, the fitted values $\boldsymbol{\hat{Y}}$ is Normally distributed because it is a linear transformation of $\boldsymbol{Y}$, and $\boldsymbol{e}$ is also a Normally distributed for the same reason.

Since both $\boldsymbol{e}$ and $\boldsymbol{\hat{Y}}$ are Normal and uncorrelated, this implies that the fitted values and residuals are independent, under the aforementioned assumptions.

If the model assumptions are correct, it is expected that the plot displays an even spread of points with $\text{E}(\boldsymbol{e})=\text{E}(\boldsymbol{Y}) - \text{E}(\boldsymbol{\hat{Y}}) = \boldsymbol{X\beta} - \boldsymbol{PX\beta} = 0$.

Looking at the plot, it seems that it affirms the assumptions made about the model of the data, as it shows an uncorrelated, even spread of points with $y$ = 0. <br>
Moreover, the red line shows that there is no underlying non-linearity hidden within the residuals, therefore supporting the assumption of *linearity*.

***Normal Q-Q Plots***:

The Normal Q-Q plots compares the quantiles of the residuals with the quantiles of an appropriate Normal distribution, thus clearly indicating if the assumption of *normality of errors*.

The plot above is not against the standard $N(0,1)$ distribution but instead against $N(0, \text{Var}(\boldsymbol{e}))$, with $\text{Var}(\boldsymbol{e}) = \boldsymbol{Q}\text{Var}(\boldsymbol{Y})\boldsymbol{Q}^T= \sigma^2\boldsymbol{QQ}^T = \sigma^2\boldsymbol{Q}$, where $\boldsymbol{Q} = \boldsymbol{I} - \boldsymbol{P}$.<br>
One can show that $\boldsymbol{Q}$ is also idempotent and symmetric.

This is because we should expect that the residuals should have mean $0$, however the model assumption is that the errors have constant variance $\sigma^2$ but not necessarily that it is equal to 1.

If the plot was against the standard Normal distribution, then the plot would show the same blue line but at with a different slope, so the scaled distribution is used to clearly show the Normality of the residuals.

***Scale-Location Plot***:

This plot makes it easier to discern if the assumption of *homoscedasticity* is valid.<br>
Similar to the first plot, the square-root standardized residuals (SRSR) are plotted against the fitted values.<br>
This is preferred to check *homoscedasticity* over the first plot because it scales the residuals to have variance $1$ before taking the square-root, and this leads a significantly easier interpretation of the assumption.

The SRSR takes the form:
$$\bar{e}_i = \sqrt{\frac{e_i}{\text{Var}(e_i)}}, \quad i \in \{1,2,...,n\}$$ 

The scale-location plot looks standard given our assumptions. Even spread with a straight red line, showing no underlying non-linearity and just reinforces what was found in the *residual vs fitted values plot*.

***Residuals vs Leverage***:

The leverage of a point $i$ is defined to be $p_{ii}$, the $i^{\text{th}}$ diagonal entry of $\boldsymbol{P}$, the hat matrix. <br>
A high leverage indicates that the data point has a lot influence on the outcome of the coefficients, so removing it from the data will change the value of the weights and bias.

Cook's distance can be used to define what 'high leverage' means. The Cook's distance, $D_i$, for an observation $i$:
$$
D_i = \frac{e_{i}p_{ii}}{r\sigma^2(1-p_{ii})},
$$
where $r$ is the rank of the design matrix. Important to note, $\sigma^2$ can be estimated by taking the average of the variances of the residuals.

Since Cook's distance is a function of the residual and the leverage. As a 
rule of thumb, a Cook's distance of 1 or greater is considered large.

it is possible to plot the contours of each particular value of the Cook's distance. In the plot above, the coloured lines represent the contour plots of $D_i \in \{0.5, 1, 2\}$.

By inspection, no points seem influential as none of the points fall beyond even the first contour line. Therefore, there seems to be no points in the particular dataset that I used that distorts the fit of model.

### Analysis of the coefficient vector, $\boldsymbol{\beta}$

Looking at the coefficient vector $\boldsymbol{\beta}$ that beat scikit-learn's implementation of the model, the output of the parameters came out to be:<br>
`(Array([-8.7041764e-05,  3.8404137e-01,  9.1835845e-01,  1.6454296e-02,
         4.1350484e-02,  2.8759979e-02], dtype=float32))`,<br>
with the very first parameter referring to $\beta_0$, the bias.<br>
For reference, the columns associated with each parameter is:<br>
$\beta_0$, `Hours Studied', 'Previous Scores', 'Extracurricular Activities',
        'Sleep Hours', 'Sample Question Papers Practiced'`.

At first glance, $\beta_0$ is really the only coefficient that is substantially smaller compared to the rest of the parameters $\beta_i$. 

This suggests that the bias is not necessary in the model for this data ([Student Performance (Multiple Linear Regression)](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)), yet the all the columns in said data seems to be important in predicting the label.

Furthermore, using R, we can output a summary of statistics of the coefficients in the model:<br>

<img src="assets/summary_statistic_of_model.jpg" alt="Summary Statistics" style="max-width: 600px; height: auto;">

Looking at the p-values, in the column `Pr(>|t|)`, we can see that every coefficients at each column are statistically significant with very low p-values except for `(Intercept)` and `Extracurricular.Activities_Yes`. 

This corresponds with the findings of the weights through the model proposed using *JAX*, except for `Extracurricular Activites`. The weight associated with the feature was indeed the second smallest, but similar in magnitude with the other statistically significant weights.

This indicates that out of all the features used, the extracurricular activities the students engage in is not influential in achieving good grades. <br>

### Results

<table>
<tr>
        <td><img src='assets/zero.png' alt='Zero-Init.' style = 'max-width:auto; height:auto;'><div align='center'>Zero Initialization</div></td>
        <td><img src='assets/random.png' alt='Zero-Init.' style = 'max-width:auto; height:auto;'><div align='center'>Random Initialization</div></td>
</tr>
        <td><img src='assets/he.png' alt='Zero-Init.' style = 'max-width:auto; height:auto;'><div align='center'>He Initialization</div></td>
        <td><img src='assets/xavier.png' alt='Zero-Init.' style = 'max-width:auto; height:auto;'><div align='center'>Xavier Initialization</div></td>
        <td><img src='assets/lecun.png' alt='Zero-Init.' style = 'max-width:auto; height:auto;'><div align='center'>LeCun Initialization</div></td>
</table>

For each weight initialization, the training and the validation losses decrease with each epoch until convergence, as expected. The random weight initialization did best overall, most likely because of the simplicity of the model and chance. The other plots demonstrate a zig-zag pattern with the training losses, which can be mitigated with a different optimizer algorithm like gradient-descent with momentum, RMS propagation, Adam etc. Issues with exploding gradients did occur with this project and was worked around using normalization of the data. However, techniques like gradient norm clipping can also work around it especially paired with a different optimization algorithm. Other losses can be used as well like RMSE (root mean squared error), SSE (sum of squared errors) and the like, to see any improvements in data you wish to use.
