import plotly.graph_objects as go
import numpy as np
import streamlit as st

def create_vector_animation_frames(matrix, vector, steps=20, dims=2):
    """
    Creates animation frames for the transformation of a vector by a matrix.

    This function generates a sequence of frames that illustrate the linear transformation
    of a given vector by a specified matrix over a number of steps. It supports both 2D and 3D
    transformations, and the generated frames can be used to create an animated visualization
    using Plotly.

    Args:
        matrix (Matrix): The transformation matrix.
        vector (Vector): The vector to be transformed.
        steps (int, optional): The number of intermediate steps for the animation. Default is 20.
        dims (int, optional): The dimensionality of the transformation (2 or 3). Default is 2.

    Returns:
        list: A list of Plotly frames representing the transformation animation.
    """
    frames = []
    for i in range(steps + 1):
        t = i / steps
        intermediate_matrix = np.eye(matrix.rows) * (1 - t) + matrix.data * t
        intermediate_vector = intermediate_matrix @ vector.data
        if dims == 2:
            frame_data = {
                'x': [0, intermediate_vector[0]],
                'y': [0, intermediate_vector[1]],
                'mode': 'lines+markers',
                'line': dict(color='red', width=3),
                'marker': dict(size=8)
            }
            frames.append(go.Frame(data=[go.Scatter(**frame_data)]))
        elif dims == 3:
            frame_data = {
                'x': [0, intermediate_vector[0]],
                'y': [0, intermediate_vector[1]],
                'z': [0, intermediate_vector[2]],
                'mode': 'lines+markers',
                'line': dict(color='red', width=5),
                'marker': dict(size=5)
            }
            frames.append(go.Frame(data=[go.Scatter3d(**frame_data)]))
    return frames

def create_grid_animation_frames(matrix, original_points, steps=20, dims=2):
    """
    Creates animation frames for the transformation of a grid of points by a matrix.

    This function generates a sequence of frames that illustrate the linear transformation
    of a given grid of points by a specified matrix over a number of steps. It supports both
    2D and 3D transformations, and the generated frames can be used to create an animated
    visualization using Plotly.

    Args:
        matrix (Matrix): The transformation matrix.
        original_points (np.ndarray): The grid of points to be transformed.
        steps (int, optional): The number of intermediate steps for the animation. Default is 20.
        dims (int, optional): The dimensionality of the transformation (2 or 3). Default is 2.

    Returns:
        list: A list of Plotly frames representing the transformation animation.
    """
    frames = []
    for i in range(steps + 1):
        t = i / steps
        intermediate_matrix = np.eye(matrix.rows) * (1 - t) + matrix.data * t
        intermediate_points = intermediate_matrix @ original_points
        if dims == 2:
            frame_data = {
                'x': intermediate_points[0, :],
                'y': intermediate_points[1, :],
                'mode': 'markers',
                'marker': dict(color='red', size=4)
            }
            frames.append(go.Frame(data=[go.Scatter(**frame_data)]))
        elif dims == 3:
            frame_data = {
                'x': intermediate_points[0, :],
                'y': intermediate_points[1, :],
                'z': intermediate_points[2, :],
                'mode': 'markers',
                'marker': dict(color='red', size=2)
            }
            frames.append(go.Frame(data=[go.Scatter3d(**frame_data)]))
    return frames

def generate_3d_shape(shape, density):
    """
    Generates a 3D grid of points for a given shape and density.

    Args:
        shape (str): The shape to generate. Supported values are "Cube", "Sphere", and "Cylinder".
        density (int): The number of points on each side of the shape.

    Returns:
        np.ndarray: A 3xN array of 3D points, where N is the total number of points in the grid.
    """
    if shape == "Cube":
        lin = np.linspace(-1, 1, density)
        X, Y, Z = np.meshgrid(lin, lin, lin)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    elif shape == "Sphere":
        phi = np.linspace(0, np.pi, density)
        theta = np.linspace(0, 2 * np.pi, density)
        phi, theta = np.meshgrid(phi, theta)
        X = np.sin(phi) * np.cos(theta)
        Y = np.sin(phi) * np.sin(theta)
        Z = np.cos(phi)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    elif shape == "Cylinder":
        z = np.linspace(-1, 1, density)
        theta = np.linspace(0, 2 * np.pi, density)
        theta, z = np.meshgrid(theta, z)
        X = np.cos(theta)
        Y = np.sin(theta)
        Z = z
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    else:
        points = np.zeros((3, 0))
    return points

def generate_2d_shape(shape, density, range_min=-1, range_max=1):
    """
    Generates a 2D grid of points for a given shape and density.

    Args:
        shape (str): The shape to generate. Supported values are "Rectangle", "Triangle", and "Octagon".
        density (int): The number of points on each side of the shape.
        range_min (float, optional): The minimum value of the range. Defaults to -1.
        range_max (float, optional): The maximum value of the range. Defaults to 1.

    Returns:
        np.ndarray: A 2xN array of 2D points, where N is the total number of points in the grid.
    """
    if shape == "Rectangle":
        x = np.linspace(range_min, range_max, density)
        y = np.linspace(range_min, range_max, density)
        X, Y = np.meshgrid(x, y)
        points = np.vstack([X.ravel(), Y.ravel()])
    elif shape == "Triangle":
        x = np.linspace(range_min, range_max, density)
        y = np.linspace(range_min, range_max, density)
        X, Y = np.meshgrid(x, y)
        mask = Y >= X  # Condition for a right-angled triangle
        points = np.vstack([X[mask].ravel(), Y[mask].ravel()])
    elif shape == "Octagon":
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        radius = (range_max - range_min) / 2
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        points = np.vstack([x, y])
    else:
        points = np.zeros((2, 0))
    return points

def create_vector_plot(vector, transformed_vector, frames, theme, dims=2):
    """
    Creates a Plotly figure to visualize the transformation of a 2D or 3D vector.

    Args:
        vector (Vector): The original vector.
        transformed_vector (Vector): The transformed vector.
        frames (list[go.Frame]): The frames of the animation.
        theme (str): The Plotly theme to use.
        dims (int, optional): The number of dimensions. Defaults to 2.

    Returns:
        None
    """
    
    max_val = max(np.abs(vector.data).max(),
                  np.abs(transformed_vector.data).max(), 1) * 1.2
    if dims == 2:
        fig = go.Figure(
            data=[
                go.Scatter(x=[0, vector.data[0]],
                           y=[0, vector.data[1]],
                           mode='lines+markers',
                           name='Original Vector',
                           line=dict(color='blue', width=3),
                           marker=dict(size=8))
            ],
            frames=frames)
        fig.update_layout(template=theme,
                          title='2D Vector Transformation Animation',
                          xaxis=dict(range=[-max_val, max_val],
                                     zeroline=True,
                                     showgrid=True,
                                     scaleanchor='y',
                                     scaleratio=1),
                          yaxis=dict(range=[-max_val, max_val],
                                     zeroline=True,
                                     showgrid=True),
                          updatemenus=[
                              dict(type='buttons',
                                   showactive=False,
                                   buttons=[
                                       dict(label='Play',
                                            method='animate',
                                            args=[
                                                None, {
                                                    'frame': {
                                                        'duration': 100,
                                                        'redraw': True
                                                    },
                                                    'fromcurrent': True,
                                                    'transition': {
                                                        'duration': 0
                                                    }
                                                }
                                            ]),
                                       dict(label='Pause',
                                            method='animate',
                                            args=[
                                                [None], {
                                                    'frame': {
                                                        'duration': 0,
                                                        'redraw': False
                                                    },
                                                    'mode': 'immediate',
                                                    'transition': {
                                                        'duration': 0
                                                    }
                                                }
                                            ])
                                   ])
                          ])
    elif dims == 3:
        fig = go.Figure(
            data=[
                go.Scatter3d(x=[0, vector.data[0]],
                             y=[0, vector.data[1]],
                             z=[0, vector.data[2]],
                             mode='lines+markers',
                             name='Original Vector',
                             line=dict(color='blue', width=5),
                             marker=dict(size=5))
            ],
            frames=frames)
        fig.update_layout(template=theme,
                          title='3D Vector Transformation Animation',
                          scene=dict(xaxis=dict(range=[-5, 5]),
                                     yaxis=dict(range=[-5, 5]),
                                     zaxis=dict(range=[-5, 5]),
                                     aspectmode='cube'),
                          updatemenus=[
                              dict(type='buttons',
                                   showactive=False,
                                   buttons=[
                                       dict(label='Play',
                                            method='animate',
                                            args=[
                                                None, {
                                                    'frame': {
                                                        'duration': 100,
                                                        'redraw': True
                                                    },
                                                    'fromcurrent': True,
                                                    'transition': {
                                                        'duration': 0
                                                    }
                                                }
                                            ]),
                                       dict(label='Pause',
                                            method='animate',
                                            args=[
                                                [None], {
                                                    'frame': {
                                                        'duration': 0,
                                                        'redraw': False
                                                    },
                                                    'mode': 'immediate',
                                                    'transition': {
                                                        'duration': 0
                                                    }
                                                }
                                            ])
                                   ])
                          ])
    st.plotly_chart(fig, use_container_width=True)

def create_grid_plot(original_points,
                     transformed_points,
                     frames,
                     theme,
                     x_range,
                     y_range,
                     dims=2):
    """
    Create a grid plot for 2D or 3D transformations.

    Parameters
    ----------
    original_points : numpy array
        The original points in 2D or 3D space.
    transformed_points : numpy array
        The points after transformation in 2D or 3D space.
    frames : list
        A list of frames for the animation.
    theme : str
        The theme for the plotly figure.
    x_range : list
        The range of the x-axis.
    y_range : list
        The range of the y-axis.
    dims : int, optional
        The number of dimensions, by default 2.

    Returns
    -------
    None
    """
    
    if dims == 2:
        orig_fig = go.Figure()
        orig_fig.add_trace(go.Scatter(x=original_points[0, :],
                                      y=original_points[1, :],
                                      mode='markers',
                                      marker=dict(color='blue', size=4),
                                      name='Original Grid'))
        orig_fig.update_layout(template=theme,
                               title='Original Grid',
                               xaxis=dict(range=x_range,
                                          zeroline=True,
                                          showgrid=True,
                                          scaleanchor='y',
                                          scaleratio=1),
                               yaxis=dict(range=y_range,
                                          zeroline=True,
                                          showgrid=True),
                               margin=dict(l=50, r=50, t=50, b=50))

        trans_fig = go.Figure(
            data=[
                go.Scatter(x=original_points[0, :],
                           y=original_points[1, :],
                           mode='markers',
                           marker=dict(color='blue', size=4),
                           name='Original Grid')
            ],
            frames=frames)
        trans_fig.update_layout(template=theme,
                                title='Transformed Grid Animation',
                                xaxis=dict(range=x_range,
                                           zeroline=True,
                                           showgrid=True,
                                           scaleanchor='y',
                                           scaleratio=1),
                                yaxis=dict(range=y_range,
                                           zeroline=True,
                                           showgrid=True),
                                updatemenus=[
                                    dict(type='buttons',
                                         showactive=False,
                                         buttons=[
                                             dict(label='Play',
                                                  method='animate',
                                                  args=[
                                                      None, {
                                                          'frame': {
                                                              'duration': 100,
                                                              'redraw': True
                                                          },
                                                          'fromcurrent': True,
                                                          'transition': {
                                                              'duration': 0
                                                          }
                                                      }
                                                  ]),
                                             dict(
                                                 label='Pause',
                                                 method='animate',
                                                 args=[
                                                     [None], {
                                                         'frame': {
                                                             'duration': 0,
                                                             'redraw': False
                                                         },
                                                         'mode': 'immediate',
                                                         'transition': {
                                                             'duration': 0
                                                         }
                                                     }
                                                 ])
                                         ])
                                ],
                                margin=dict(l=50, r=50, t=50, b=50))
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(orig_fig, use_container_width=True)
            st.caption("Original Grid")
        with col2:
            st.plotly_chart(trans_fig, use_container_width=True)
            st.caption("Transformed Grid Animation")

    elif dims == 3:
        orig_fig_3d = go.Figure()
        orig_fig_3d.add_trace(go.Scatter3d(x=original_points[0, :],
                                           y=original_points[1, :],
                                           z=original_points[2, :],
                                           mode='markers',
                                           marker=dict(color='blue', size=2),
                                           name='Original Shape'))
        orig_fig_3d.update_layout(template=theme,
                                  title='Original 3D Shape',
                                  scene=dict(xaxis=dict(range=[-2, 2]),
                                             yaxis=dict(range=[-2, 2]),
                                             zaxis=dict(range=[-2, 2]),
                                             aspectmode='cube'),
                                  margin=dict(l=50, r=50, t=50, b=50))

        trans_fig_3d = go.Figure(
            data=[
                go.Scatter3d(x=original_points[0, :],
                             y=original_points[1, :],
                             z=original_points[2, :],
                             mode='markers',
                             marker=dict(color='blue', size=2),
                             name='Original Shape')
            ],
            frames=frames)
        trans_fig_3d.update_layout(template=theme,
                                   title='Transformed 3D Shape Animation',
                                   scene=dict(xaxis=dict(range=[-2, 2]),
                                              yaxis=dict(range=[-2, 2]),
                                              zaxis=dict(range=[-2, 2]),
                                              aspectmode='cube'),
                                   updatemenus=[
                                       dict(type='buttons',
                                            showactive=False,
                                            buttons=[
                                                dict(label='Play',
                                                     method='animate',
                                                     args=[
                                                         None, {
                                                             'frame': {
                                                                 'duration':
                                                                 100,
                                                                 'redraw': True
                                                             },
                                                             'fromcurrent':
                                                             True,
                                                             'transition': {
                                                                 'duration': 0
                                                             }
                                                         }
                                                     ]),
                                                dict(
                                                    label='Pause',
                                                    method='animate',
                                                    args=[
                                                        [None], {
                                                            'frame': {
                                                                'duration': 0,
                                                                'redraw': False
                                                            },
                                                            'mode': 'immediate',
                                                            'transition': {
                                                                'duration': 0
                                                            }
                                                        }
                                                    ])
                                            ])
                                   ],
                                   margin=dict(l=50, r=50, t=50, b=50))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(orig_fig_3d, use_container_width=True)
            st.caption("Original 3D Shape")
        with col2:
            st.plotly_chart(trans_fig_3d, use_container_width=True)
            st.caption("Transformed 3D Shape Animation")
