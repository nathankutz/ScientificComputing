# Homework 3 Report — Feature Spaces

## Overview

This homework explores **feature spaces** and **dimensionality reduction** using the Yale Face Database (YaleFaces). The dataset contains 2414 grayscale face images (39 unique individuals under ~65 lighting conditions each), downsampled to 32x32 pixels and stored as columns of a 1024x2414 matrix $\mathbf{X}$. We investigate image similarity through correlation (dot product) matrices, eigendecomposition, and Singular Value Decomposition (SVD) to understand how high-dimensional face data can be compactly represented.

---

## Part (a) — 100x100 Correlation Matrix

### Method

We compute the correlation matrix $\mathbf{C} \in \mathbb{R}^{100 \times 100}$ for the first 100 images using the dot product:

$$c_{jk} = \mathbf{x}_j^T \mathbf{x}_k$$

where $\mathbf{x}_j$ is the $j$-th column (image) of $\mathbf{X}$. This dot product measures the similarity between two images based on the alignment of their pixel intensity vectors. Two images with similar brightness patterns will produce a large positive dot product, while images with opposing patterns yield smaller values.

### Results

The correlation matrix, visualized with `pcolor`, reveals a clear **block-diagonal structure**. Since the first ~65 images correspond to the same individual under different lighting, images within this block are highly correlated with each other. A visible transition occurs around index 65, where a new individual begins, and correlations drop off. The diagonal naturally contains the largest values since $c_{jj} = \|\mathbf{x}_j\|^2$.

---

## Part (b) — Most and Least Correlated Image Pairs

### Method

To find the most correlated and most uncorrelated **distinct** image pairs, we exclude the diagonal of $\mathbf{C}$ (which represents self-correlation) by masking it with $-\infty$ for the maximum search and $+\infty$ for the minimum search.

### Results

| Pair Type | Image Indices | Correlation |
|-----------|--------------|-------------|
| Most correlated | (86, 88) | High |
| Most uncorrelated | (54, 64) | Low |

The most correlated pair (images 86 and 88) are images of the **same individual** under very similar lighting conditions — visually, they appear nearly identical. The most uncorrelated pair (images 54 and 64) correspond to images with very different overall brightness levels; one is brightly lit while the other is much darker, resulting in a low dot product.

This illustrates a key property of the dot product as a similarity measure: it is sensitive to both the **direction** (pattern) and **magnitude** (overall brightness) of the image vectors. Two images of the same face under different lighting intensities can appear less correlated than two different faces with similar brightness.

---

## Part (c) — 10x10 Correlation Matrix for Selected Images

### Method

We compute the same dot-product correlation matrix, now for 10 specific images at indices [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005] (1-indexed as specified in the assignment, converted to 0-indexed in Python).

### Results

The 10x10 correlation matrix shows a more heterogeneous pattern compared to part (a), since these images span different individuals and lighting conditions across the full dataset. Notable observations:

- Images 1 and 5 (both from the first individual) show high mutual correlation.
- Images 313 and 314 (adjacent indices, likely same person) also show high correlation.
- Cross-individual pairs generally have lower correlation, though pairs with similar overall illumination can still show moderate values.

This subset demonstrates that the dot product captures both **identity** and **illumination** information — a face under bright frontal lighting may correlate more strongly with a *different* face under similar lighting than with itself under extreme side lighting.

---

## Part (d) — Eigendecomposition of $\mathbf{Y} = \mathbf{X}\mathbf{X}^T$

### Method

We form the 1024x1024 matrix $\mathbf{Y} = \mathbf{X}\mathbf{X}^T$ and compute its eigendecomposition using `numpy.linalg.eigh`. The matrix $\mathbf{Y}$ is symmetric positive semi-definite, so all eigenvalues are non-negative. We extract the six eigenvectors corresponding to the six largest eigenvalues.

### Results

The six largest eigenvalues of $\mathbf{Y}$ are:

| Rank | Eigenvalue |
|------|-----------|
| 1 | 234,020.5 |
| 2 | 49,038.3 |
| 3 | 8,236.5 |
| 4 | 6,024.9 |
| 5 | 2,051.5 |
| 6 | 1,901.1 |

The eigenvalues decay rapidly: the first eigenvalue is nearly 5x the second, and the second is 6x the third. This steep drop indicates that the face data is highly compressible — most of the variance in the 1024-dimensional pixel space is concentrated along a small number of directions.

The eigenvectors of $\mathbf{Y} = \mathbf{X}\mathbf{X}^T$ are the **principal axes** of the face space. They represent the directions of maximal variance in pixel space and are often called **eigenfaces**.

---

## Part (e) — SVD of $\mathbf{X}$

### Method

We compute the Singular Value Decomposition:

$$\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where $\mathbf{U} \in \mathbb{R}^{1024 \times 2414}$ contains the left singular vectors, $\boldsymbol{\Sigma}$ is a diagonal matrix of singular values, and $\mathbf{V}^T$ contains the right singular vectors. The first six columns of $\mathbf{U}$ are the first six principal component directions.

### Connection to Eigendecomposition

The SVD and eigendecomposition are deeply connected. Since

$$\mathbf{X}\mathbf{X}^T = \mathbf{U} \boldsymbol{\Sigma}^2 \mathbf{U}^T$$

the left singular vectors of $\mathbf{X}$ are exactly the eigenvectors of $\mathbf{X}\mathbf{X}^T$, and the squared singular values are the eigenvalues of $\mathbf{X}\mathbf{X}^T$. This is verified by confirming $\sigma_i^2 = \lambda_i$ for each mode.

---

## Part (f) — Comparison of Eigenvectors and SVD Modes

### Method

We compare the first eigenvector $\mathbf{v}_1$ from part (d) with the first SVD mode $\mathbf{u}_1$ from part (e) by computing:

$$\| |\mathbf{v}_1| - |\mathbf{u}_1| \|_2$$

The absolute values are taken because eigenvectors are defined only up to a sign — both $\mathbf{v}$ and $-\mathbf{v}$ are valid eigenvectors.

### Results

$$\| |\mathbf{v}_1| - |\mathbf{u}_1| \|_2 \approx 5.25 \times 10^{-16}$$

This is effectively **machine-precision zero**, confirming the theoretical equivalence: the eigenvectors of $\mathbf{X}\mathbf{X}^T$ and the left singular vectors of $\mathbf{X}$ are the same (up to sign). The tiny residual is purely floating-point rounding error.

This result validates that SVD provides an alternative — and often numerically more stable — route to the same principal components that eigendecomposition of the covariance-like matrix $\mathbf{X}\mathbf{X}^T$ yields.

---

## Part (g) — Variance Captured by SVD Modes

### Method

The percentage of variance captured by the $i$-th SVD mode is:

$$\text{Var}_i = \frac{\sigma_i^2}{\sum_{j=1}^{r} \sigma_j^2} \times 100\%$$

where $\sigma_i$ is the $i$-th singular value and $r$ is the rank of $\mathbf{X}$.

### Results

| SVD Mode | Variance Captured |
|----------|------------------|
| 1 | 72.93% |
| 2 | 15.28% |
| 3 | 2.57% |
| 4 | 1.88% |
| 5 | 0.64% |
| 6 | 0.59% |
| **Total (1-6)** | **93.89%** |

The first six SVD modes together capture nearly **94% of all variance** in the dataset. The first mode alone accounts for almost 73%, representing the average face pattern (overall brightness and face shape). The second mode (15.3%) captures the primary lighting variation (left-right asymmetry). Higher modes pick up progressively finer facial features and lighting effects, each contributing diminishing amounts of variance.

### SVD Modes as Eigenfaces

When reshaped to 32x32 images, the six SVD modes reveal interpretable spatial patterns:

- **Mode 1**: A smooth, uniformly lit face — the "average face." Its dominance (73%) reflects the fact that all images share the basic structure of a face.
- **Mode 2**: A left-right lighting gradient, capturing the primary source of variation in the dataset (different lighting directions).
- **Modes 3-6**: Increasingly complex patterns that capture vertical lighting gradients, expression variation, and finer facial structure.

---

## Summary

This homework demonstrates how linear algebra tools — dot products, eigendecomposition, and SVD — can extract meaningful structure from high-dimensional image data.

**Key takeaways:**

1. **Dot-product correlation** provides a simple but effective measure of image similarity, capturing both identity and illumination information. Care must be taken to exclude self-correlation when identifying the most similar distinct pairs.

2. **Eigendecomposition of $\mathbf{X}\mathbf{X}^T$** and **SVD of $\mathbf{X}$** yield identical principal components (up to sign), as confirmed by a near-zero norm difference. SVD is generally preferred in practice because it avoids explicitly forming the large $\mathbf{X}\mathbf{X}^T$ matrix, which can amplify numerical errors.

3. **Dimensionality reduction is remarkably effective** for face data: just 6 modes out of 1024 possible dimensions capture 94% of the total variance. This means that a face image, while nominally living in a 1024-dimensional space, can be well-approximated by its projection onto a 6-dimensional subspace — the essence of Principal Component Analysis (PCA).

4. The **eigenfaces** (SVD modes) are interpretable: they progress from gross illumination and shape features (mode 1) to fine-grained variations (higher modes), providing a natural hierarchy for representing and compressing face data.
