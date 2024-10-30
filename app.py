import streamlit as st
import numpy as np
from utils import (create_vector_animation_frames,
                   create_grid_animation_frames, generate_3d_shape, generate_2d_shape,
                   create_vector_plot, create_grid_plot)
from models import (Matrix, Vector, TRANSFORMATIONS_2D, TRANSFORMATIONS_3D,
                    SYMMETRIC_TRANSFORMATIONS_2D, SYMMETRIC_TRANSFORMATIONS_3D)

# --- Streamlit App Setup ---
st.title("2D and 3D Matrix Transformation Visualization")
st.write(
    "This app visualizes how matrices transform vectors and grids in 2D and 3D space. Adjust the matrix entries and the vector components to see how the transformation affects them."
)

# --- Sidebar ---
# Plot Theme Selection
st.sidebar.header("Plot Theme")
available_themes = [
    "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn",
    "simple_white", "none"
]
selected_theme = st.sidebar.selectbox("Select Plotly Theme",
                                      options=available_themes,
                                      index=available_themes.index(
                                          "plotly_white"))

# --- 2D Grid Setting ---
st.sidebar.header("2D Grid Settings")
grid_shape_2d = st.sidebar.selectbox("Select 2D Shape",
                                     options=["Rectangle", "Triangle", "Octagon"])
point_density = st.sidebar.slider("Point Density",
                                  min_value=5,
                                  max_value=50,
                                  value=21,
                                  step=2)
range_min = st.sidebar.number_input("Range Min", value=-10.0, step=1.0)
range_max = st.sidebar.number_input("Range Max", value=10.0, step=1.0)

# Generate 2D Grid
original_points = generate_2d_shape(grid_shape_2d, int(point_density), range_min, range_max)

# --- 2D Transformation Examples ---
st.header("2D Transformation Examples Visualization")
# Transformation Example Selection
st.sidebar.header("2D Transformation Examples")
selected_example_2d_name = st.sidebar.selectbox(
    'Select a 2D Transformation Example',
    options=['None'] + list(TRANSFORMATIONS_2D.keys()),
    index=0)
selected_example_2d = TRANSFORMATIONS_2D.get(selected_example_2d_name)

# Initialize 2D Matrix
if 'matrix_2d' not in st.session_state:
    st.session_state['matrix_2d'] = np.eye(2)

# Apply or Update 2D Matrix
if selected_example_2d:
    st.session_state['matrix_2d'] = selected_example_2d.matrix.data
else:
    st.sidebar.header("2D Matrix Input")
    matrix_entries_2d = {}
    for i in range(2):
        for j in range(2):
            key = f'a{i+1}{j+1}'
            value = float(st.session_state['matrix_2d'][i, j])
            matrix_entries_2d[key] = st.sidebar.number_input(
                key, value=value, step=0.1, format="%.2f")
    st.session_state['matrix_2d'] = np.array([
        [matrix_entries_2d['a11'], matrix_entries_2d['a12']],
        [matrix_entries_2d['a21'], matrix_entries_2d['a22']]
    ])

# Display 2D Matrix
st.sidebar.write("Current 2D Matrix:")
st.sidebar.latex(
    r'''
    \begin{pmatrix}
    %.2f & %.2f \\
    %.2f & %.2f
    \end{pmatrix}
    ''' % (st.session_state['matrix_2d'][0, 0],
           st.session_state['matrix_2d'][0, 1],
           st.session_state['matrix_2d'][1, 0],
           st.session_state['matrix_2d'][1, 1]))

# 2D Vector Input
st.sidebar.header("2D Vector Input")
vector_2d = np.zeros(2)
for i in range(2):
    key = f'v{i+1}'
    default_value = 1.0 if i == 0 else 0.0
    vector_2d[i] = st.sidebar.number_input(key,
                                           value=float(default_value),
                                           step=0.1, format="%.2f")

# 2D Vector Transformation
matrix_2d = Matrix(st.session_state['matrix_2d'])
vector_2d = Vector(vector_2d)
transformed_vector_2d = matrix_2d @ vector_2d

# 2D Vector Animation
frames_2d = create_vector_animation_frames(matrix_2d, vector_2d)
create_vector_plot(vector_2d, transformed_vector_2d, frames_2d,
                   selected_theme)

# --- 2D Grid Transformation ---
st.header(" Geometric Transformations in ℝ²")
# Apply Transformation
transformed_points = st.session_state['matrix_2d'] @ original_points

# Calculate Axis Limits
all_x = np.concatenate([original_points[0, :], transformed_points[0, :]])
all_y = np.concatenate([original_points[1, :], transformed_points[1, :]])
x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()
padding = max((x_max - x_min), (y_max - y_min)) * 0.1
x_range = [x_min - padding, x_max + padding]
y_range = [y_min - padding, y_max + padding]

# Grid Animation
frames_grid = create_grid_animation_frames(matrix_2d, original_points)
create_grid_plot(original_points, transformed_points, frames_grid,
                 selected_theme, x_range, y_range)

# --- 2D Eigenvalues and Eigenvectors ---
st.header("Eigenvalues and Eigenvectors (2D)")
eigenvalues, eigenvectors = matrix_2d.eigenvalues()
st.subheader("Eigenvalues")
for idx, val in enumerate(eigenvalues):
    st.latex(r"\lambda_%d = %.2f" % (idx + 1, val.real))
st.subheader("Eigenvectors")
for idx, vec in enumerate(eigenvectors.T):
    st.latex(r"""\vec{{v}}_{%d} = \begin{pmatrix} %.2f \\ %.2f \end{pmatrix}""" %
             (idx + 1, vec[0].real, vec[1].real))
st.write(
    r"""An **eigenvector** of a matrix \( A \) is a non-zero vector \( \vec{v} \) that, when \( A \) is applied to it, does not change direction. Instead, it is only scaled by a scalar factor called the **eigenvalue** \( \lambda \). This relationship is expressed as:"""
)
st.latex(r"A \vec{v} = \lambda \vec{v}")

# --- Symmetric Matrices in 2D ---
st.header("Symmetric Matrices in 2D")
# Symmetric Transformation Example Selection
st.sidebar.header("Symmetric Matrices (2D)")
selected_symmetric_example_2d_name = st.sidebar.selectbox(
    'Select a Symmetric 2D Transformation',
    options=['None'] + list(SYMMETRIC_TRANSFORMATIONS_2D.keys()),
    index=0)
selected_symmetric_example_2d = SYMMETRIC_TRANSFORMATIONS_2D.get(selected_symmetric_example_2d_name)

if selected_symmetric_example_2d:
    st.session_state['matrix_2d'] = selected_symmetric_example_2d.matrix.data
    # Display Symmetric Matrix
    st.write(f"**{selected_symmetric_example_2d.name}**:")
    st.latex(selected_symmetric_example_2d.latex)
    st.write(selected_symmetric_example_2d.explanation)
    # Recalculate transformations
    matrix_2d = Matrix(st.session_state['matrix_2d'])
    transformed_vector_2d = matrix_2d @ vector_2d
    transformed_points = st.session_state['matrix_2d'] @ original_points
    # Recalculate animations
    frames_2d = create_vector_animation_frames(matrix_2d, vector_2d)
    create_vector_plot(vector_2d, transformed_vector_2d, frames_2d,
                       selected_theme)
    frames_grid = create_grid_animation_frames(matrix_2d, original_points)
    create_grid_plot(original_points, transformed_points, frames_grid,
                     selected_theme, x_range, y_range)
    # Eigenvalues and Eigenvectors
    st.subheader("Eigenvalues and Eigenvectors of Symmetric Matrix")
    eigenvalues, eigenvectors = matrix_2d.eigenvalues()
    st.subheader("Eigenvalues")
    for idx, val in enumerate(eigenvalues):
        st.latex(r"\lambda_%d = %.2f" % (idx + 1, val.real))
    st.subheader("Eigenvectors")
    for idx, vec in enumerate(eigenvectors.T):
        st.latex(r"""\vec{{v}}_{%d} = \begin{pmatrix} %.2f \\ %.2f \end{pmatrix}""" %
                 (idx + 1, vec[0].real, vec[1].real))
    st.write(
        "Symmetric matrices have real eigenvalues and orthogonal eigenvectors. They represent transformations that are symmetrical with respect to the basis vectors."
    )

# --- Transformation Explanations ---
if selected_example_2d:
    st.header("2D Transformation Example Explanation")
    st.write(f"**{selected_example_2d.name}**:")
    st.latex(selected_example_2d.latex)
    st.write(selected_example_2d.explanation)

if selected_symmetric_example_2d:
    st.header("Symmetric Transformation Explanation")
    st.write(f"**{selected_symmetric_example_2d.name}**:")
    st.latex(selected_symmetric_example_2d.latex)
    st.write(selected_symmetric_example_2d.explanation)

# --- 3D Transformation Examples ---
st.header("3D Transformation Examples")
st.sidebar.header("3D Transformation Examples")
selected_example_3d_name = st.sidebar.selectbox(
    'Select a 3D Transformation Example',
    options=['None'] + list(TRANSFORMATIONS_3D.keys()),
    index=0)
selected_example_3d = TRANSFORMATIONS_3D.get(selected_example_3d_name)

# Initialize 3D Matrix
if 'matrix_3d' not in st.session_state:
    st.session_state['matrix_3d'] = np.eye(3)

# Apply or Update 3D Matrix
if selected_example_3d:
    st.session_state['matrix_3d'] = selected_example_3d.matrix.data
else:
    st.sidebar.header("3D Matrix Input")
    matrix_entries_3d = {}
    for i in range(3):
        for j in range(3):
            key = f'a{i+1}{j+1}_3d'
            value = float(st.session_state['matrix_3d'][i, j])
            matrix_entries_3d[key] = st.sidebar.number_input(
                key, value=value, step=0.1, format="%.2f")
    st.session_state['matrix_3d'] = np.array([
        [
            matrix_entries_3d['a11_3d'], matrix_entries_3d['a12_3d'],
            matrix_entries_3d['a13_3d']
        ],
        [
            matrix_entries_3d['a21_3d'], matrix_entries_3d['a22_3d'],
            matrix_entries_3d['a23_3d']
        ],
        [
            matrix_entries_3d['a31_3d'], matrix_entries_3d['a32_3d'],
            matrix_entries_3d['a33_3d']
        ]
    ])

# Display 3D Matrix
st.sidebar.write("Current 3D Matrix:")
st.sidebar.latex(
    r'''
    \begin{pmatrix}
    %.2f & %.2f & %.2f \\
    %.2f & %.2f & %.2f \\
    %.2f & %.2f & %.2f
    \end{pmatrix}
    ''' % tuple(st.session_state['matrix_3d'].flatten()))

# 3D Vector Input
st.sidebar.header("3D Vector Input")
vector_3d = np.zeros(3)
for i in range(3):
    key = f'v{i+1}_3d'
    default_value = 1.0 if i == 0 else 0.0
    vector_3d[i] = st.sidebar.number_input(key,
                                           value=float(default_value),
                                           step=0.1, format="%.2f")

# 3D Vector Transformation
matrix_3d = Matrix(st.session_state['matrix_3d'])
vector_3d = Vector(vector_3d)
transformed_vector_3d = matrix_3d @ vector_3d

# 3D Vector Animation
frames_3d = create_vector_animation_frames(matrix_3d, vector_3d, dims=3)
create_vector_plot(vector_3d, transformed_vector_3d, frames_3d,
                   selected_theme, dims=3)

# --- 3D Grid Transformation ---
st.header("Geometric Transformations in ℝ³")
# 3D Grid Settings
st.sidebar.header("3D Grid Settings")
grid_shape = st.sidebar.selectbox("Select 3D Shape",
                                  options=["Cube", "Sphere", "Cylinder"])
grid_density = st.sidebar.slider("Grid Density",
                                 min_value=5,
                                 max_value=30,
                                 value=10,
                                 step=1)
# Generate 3D Grid
original_points_3d = generate_3d_shape(grid_shape, grid_density)
transformed_points_3d = st.session_state['matrix_3d'] @ original_points_3d

# 3D Grid Animation
frames_grid_3d = create_grid_animation_frames(matrix_3d,
                                              original_points_3d,
                                              dims=3)
create_grid_plot(original_points_3d,
                 transformed_points_3d,
                 frames_grid_3d,
                 selected_theme,
                 None,
                 None,
                 dims=3)

# --- 3D Eigenvalues and Eigenvectors ---
st.header("Eigenvalues and Eigenvectors (3D)")
eigenvalues_3d, eigenvectors_3d = matrix_3d.eigenvalues()
st.subheader("Eigenvalues")
for idx, val in enumerate(eigenvalues_3d):
    st.latex(r"\lambda_%d = %.2f" % (idx + 1, val.real))
st.subheader("Eigenvectors")
for idx, vec in enumerate(eigenvectors_3d.T):
    st.latex(
        r"""\vec{{v}}_{%d} = \begin{pmatrix} %.2f \\ %.2f \\ %.2f \end{pmatrix}""" %
        (idx + 1, vec[0].real, vec[1].real, vec[2].real))
st.write(
    r"""An **eigenvector** of a matrix \( A \) is a non-zero vector \( \vec{v} \) that, when \( A \) is applied to it, does not change direction. Instead, it is only scaled by a scalar factor called the **eigenvalue** \( \lambda \). This relationship is expressed as:"""
)
st.latex(r"A \vec{v} = \lambda \vec{v}")

# --- Symmetric Matrices in 3D ---
st.header("Symmetric Matrices in 3D")
# Symmetric Transformation Example Selection
st.sidebar.header("Symmetric Matrices (3D)")
selected_symmetric_example_3d_name = st.sidebar.selectbox(
    'Select a Symmetric 3D Transformation',
    options=['None'] + list(SYMMETRIC_TRANSFORMATIONS_3D.keys()),
    index=0)
selected_symmetric_example_3d = SYMMETRIC_TRANSFORMATIONS_3D.get(selected_symmetric_example_3d_name)

if selected_symmetric_example_3d:
    st.session_state['matrix_3d'] = selected_symmetric_example_3d.matrix.data
    # Display Symmetric Matrix
    st.write(f"**{selected_symmetric_example_3d.name}**:")
    st.latex(selected_symmetric_example_3d.latex)
    st.write(selected_symmetric_example_3d.explanation)
    # Recalculate transformations
    matrix_3d = Matrix(st.session_state['matrix_3d'])
    transformed_vector_3d = matrix_3d @ vector_3d
    transformed_points_3d = st.session_state['matrix_3d'] @ original_points_3d
    # Recalculate animations
    frames_3d = create_vector_animation_frames(matrix_3d, vector_3d, dims=3)
    create_vector_plot(vector_3d, transformed_vector_3d, frames_3d,
                       selected_theme, dims=3)
    frames_grid_3d = create_grid_animation_frames(matrix_3d,
                                                  original_points_3d,
                                                  dims=3)
    create_grid_plot(original_points_3d,
                     transformed_points_3d,
                     frames_grid_3d,
                     selected_theme,
                     None,
                     None,
                     dims=3)
    # Eigenvalues and Eigenvectors
    st.subheader("Eigenvalues and Eigenvectors of Symmetric Matrix")
    eigenvalues_3d, eigenvectors_3d = matrix_3d.eigenvalues()
    st.subheader("Eigenvalues")
    for idx, val in enumerate(eigenvalues_3d):
        st.latex(r"\lambda_%d = %.2f" % (idx + 1, val.real))
    st.subheader("Eigenvectors")
    for idx, vec in enumerate(eigenvectors_3d.T):
        st.latex(
            r"""\vec{{v}}_{%d} = \begin{pmatrix} %.2f \\ %.2f \\ %.2f \end{pmatrix}""" %
            (idx + 1, vec[0].real, vec[1].real, vec[2].real))
    st.write(
        "Symmetric matrices in 3D have real eigenvalues and orthogonal eigenvectors. They represent transformations that are symmetrical in all three dimensions."
    )

# --- Transformation Explanations ---
if selected_example_3d:
    st.header("3D Transformation Example Explanation")
    st.write(f"**{selected_example_3d.name}**:")
    st.latex(selected_example_3d.latex)
    st.write(selected_example_3d.explanation)

if selected_symmetric_example_3d:
    st.header("Symmetric Transformation Explanation (3D)")
    st.write(f"**{selected_symmetric_example_3d.name}**:")
    st.latex(selected_symmetric_example_3d.latex)
    st.write(selected_symmetric_example_3d.explanation)
