import numpy as np

class Matrix:
    def __init__(self, data):
        """
        Constructor for Matrix class.

        Parameters
        ----------
        data : 2D list or 2D numpy array
            The data to be stored in the matrix.

        Attributes
        ----------
        data : 2D numpy array
            The underlying data of the matrix.
        rows : int
            The number of rows in the matrix.
        cols : int
            The number of columns in the matrix.
        """
        self.data = np.array(data, dtype=float)
        self.rows, self.cols = self.data.shape

    def __matmul__(self, other):
        """
        Perform matrix multiplication between this matrix and another matrix or vector.

        Parameters
        ----------
        other : Matrix or Vector
            The matrix or vector to be multiplied.

        Returns
        -------
        result : Matrix or Vector
            The result of the matrix multiplication.

        Notes
        -----
        If the other object is a Vector, the result is also a Vector.
        If the other object is a Matrix, the result is also a Matrix.
        Otherwise, the result is a numpy array.
        """
        if isinstance(other, Vector):
            return Vector(self.data @ other.data)
        elif isinstance(other, Matrix):
            return Matrix(self.data @ other.data)
        else:
            return self.data @ other

    def eigenvalues(self):
      # Returns eigenvalues and eigenvectors
      return np.linalg.eig(self.data)

    def inverse(self):
        # Returns the inverse of the matrix
        return Matrix(np.linalg.inv(self.data))

    def transpose(self):
        # Returns the transpose of the matrix
        return Matrix(self.data.T)

class Vector:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)

    def __str__(self):
        return str(self.data)

class Transformation:
    def __init__(self, name, matrix, latex, explanation):
        self.name = name
        self.matrix = Matrix(matrix)
        self.latex = latex
        self.explanation = explanation

    def apply(self, vector):
        return self.matrix @ vector

TRANSFORMATIONS_2D = {
    'Identity Matrix': Transformation(
        'Identity Matrix',
        np.eye(2),
        r"\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}",
        "The identity matrix leaves all vectors unchanged."
    ),
    'Scalar Matrix (2I)': Transformation(
        'Scalar Matrix (2I)',
        2 * np.eye(2),
        r"2 \times \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}",
        "This scalar matrix scales all vectors uniformly by a factor of 2."
    ),
    'Identity Matrix with Scaling Factor (1I)': Transformation(
        'Identity Matrix with Scaling Factor (1I)',
        1 * np.eye(2),
        r"1 \times \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}",
        "This is the identity matrix scaled by 1, leaving vectors unchanged."
    ),
    'Negative Identity Matrix (-I)': Transformation(
        'Negative Identity Matrix (-I)',
        -1 * np.eye(2),
        r"-1 \times \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}",
        "This matrix reflects all vectors through the origin (180-degree rotation)."
    ),
    'Diagonal Matrix': Transformation(
        'Diagonal Matrix',
        np.array([[3, 0], [0, 1]]),
        r"\begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}",
        "This diagonal matrix scales the x-component by 3 and leaves the y-component unchanged."
    ),
    'Zero Matrix': Transformation(
        'Zero Matrix',
        np.zeros((2, 2)),
        r"\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}",
        "This matrix maps all vectors to the zero vector."
    ),
    'Shear Matrix': Transformation(
        'Shear Matrix',
        np.array([[1, 1], [0, 1]]),
        r"\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}",
        "This shear matrix moves points in the x-direction proportionally to their y-coordinate."
    ),
    'Orthogonal Matrix': Transformation(
        'Orthogonal Matrix',
        np.array([[0, -1], [1, 0]]),
        r"\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}",
        "This orthogonal matrix rotates vectors by 90 degrees counterclockwise."
    ),
    'Projection Matrix': Transformation(
        'Projection Matrix',
        np.array([[1, 0], [0, 0]]),
        r"\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}",
        "This matrix projects vectors onto the x-axis."
    ),
    'Invertible Matrix': Transformation(
        'Invertible Matrix',
        np.array([[2, 1], [1, 1]]),
        r"\begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}",
        "This matrix is invertible and represents a combination of scaling and shearing."
    ),
}

# Symmetric Matrices for 2D
SYMMETRIC_TRANSFORMATIONS_2D = {
    'Symmetric Matrix': Transformation(
        'Symmetric Matrix',
        np.array([[2, 1], [1, 2]]),
        r"\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}",
        "This symmetric matrix scales and shears vectors symmetrically."
    ),
}

# 3D Transformations
TRANSFORMATIONS_3D = {
    'Identity Matrix': Transformation(
        'Identity Matrix',
        np.eye(3),
        r"\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}",
        "The identity matrix in 3D leaves all vectors unchanged."
    ),
    'Scalar Matrix (2I)': Transformation(
        'Scalar Matrix (2I)',
        2 * np.eye(3),
        r"2 \times \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}",
        "This scalar matrix scales all vectors uniformly by a factor of 2 in 3D."
    ),
    'Identity Matrix with Scaling Factor (1I)': Transformation(
        'Identity Matrix with Scaling Factor (1I)',
        1 * np.eye(3),
        r"1 \times \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}",
        "This is the identity matrix scaled by 1, leaving vectors unchanged in 3D."
    ),
    'Negative Identity Matrix (-I)': Transformation(
        'Negative Identity Matrix (-I)',
        -1 * np.eye(3),
        r"-1 \times \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}",
        "This matrix reflects all vectors through the origin in 3D."
    ),
    'Diagonal Matrix': Transformation(
        'Diagonal Matrix',
        np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]]),
        r"\begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 1 \end{pmatrix}",
        "This diagonal matrix scales x by 2, y by 3, and leaves z unchanged."
    ),
    'Zero Matrix': Transformation(
        'Zero Matrix',
        np.zeros((3, 3)),
        r"\begin{pmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}",
        "This matrix maps all vectors to the zero vector in 3D."
    ),
    'Shear Matrix': Transformation(
        'Shear Matrix',
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]),
        r"\begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix}",
        "This shear matrix shears vectors in 3D space."
    ),
    'Orthogonal Matrix': Transformation(
        'Orthogonal Matrix',
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        r"\begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}",
        "This orthogonal matrix rotates vectors by 90 degrees around the z-axis."
    ),
    'Projection Matrix': Transformation(
        'Projection Matrix',
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        r"\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}",
        "This matrix projects vectors onto the xy-plane."
    ),
    'Invertible Matrix': Transformation(
        'Invertible Matrix',
        np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]]),
        r"\begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}",
        "This invertible matrix combines scaling and shearing in 3D."
    ),
}

# Symmetric Matrices for 3D
SYMMETRIC_TRANSFORMATIONS_3D = {
    'Symmetric Matrix': Transformation(
        'Symmetric Matrix',
        np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]]),
        r"\begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}",
        "This symmetric matrix in 3D scales and shears vectors symmetrically."
    ),
}
