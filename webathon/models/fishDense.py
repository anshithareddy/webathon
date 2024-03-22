
import streamlit as st

def create_fish_density_plot():
    # Your existing code to generate the dataset, train the model, and create the figure
    # ...
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor

    # Generate sample dataset
    num_rows = 20  # Increase number of rows
    num_cols = 20  # Increase number of columns

    # Generate random fish density values for each region
    fish_density = np.random.randint(0, 100, size=(num_rows, num_cols))

    # Generate random depth values for each region
    depth = np.random.randint(10, 100, size=(num_rows, num_cols))

    # Flatten the grid into a 1D array
    fish_density_flat = fish_density.flatten()
    depth_flat = depth.flatten()

    # Generate coordinates for each region
    coordinates = [(i, j) for i in range(num_rows) for j in range(num_cols)]

    # Create a DataFrame to hold the dataset
    data = pd.DataFrame({'Latitude': [coord[0] for coord in coordinates],
                        'Longitude': [coord[1] for coord in coordinates],
                        'Depth': depth_flat,
                        'Fish_Density': fish_density_flat})

    # Split the dataset into training and testing sets
    X = data[['Latitude', 'Longitude', 'Depth']]
    y = data['Fish_Density']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor model with hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf_regressor = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_

    # Predict fish density for each region
    data['Predicted_Fish_Density'] = best_rf_model.predict(X)

    # Create a scatter plot of the predicted fish density using Plotly
    fig = go.Figure()

    # Add water animation
    water_animation = go.Scatter3d(
        x=data['Longitude'],
        y=data['Latitude'],
        z=data['Depth'] * -1,  # Flip depth for visualization
        mode='markers',
        marker=dict(
            size=8,
            color='blue',  # Blue color for water
            opacity=0.5,
        ),
        showlegend=False
    )

    # Add special character animation (e.g., fish)
    fish_animation = go.Scatter3d(
        x=[10, 15, 8, 12],  # Example fish positions
        y=[5, 10, 8, 14],    # Example fish positions
        z=[-30, -40, -25, -35],  # Example fish depth positions
        mode='markers+text',
        marker=dict(
            size=30,
            color='orange',  # Orange color for fish
            opacity=0.8,
        ),
        text=['🐟', '🐠', '🐡', '🦈'],  # Fish emojis
        showlegend=False
    )

    # Add predicted fish density scatter plot
    predicted_density_scatter = go.Scatter3d(
        x=data['Longitude'],
        y=data['Latitude'],
        z=data['Depth'] * -1,  # Flip depth for visualization
        mode='markers',
        marker=dict(
            size=12,
            color=data['Predicted_Fish_Density'],
            colorscale='Viridis',  # Choose a visually appealing color scale
            opacity=0.8,
            colorbar=dict(title='Fish Density')
        ),
        showlegend=False
    )

    # Add all traces to the figure
    fig.add_trace(water_animation)
    fig.add_trace(fish_animation)
    fig.add_trace(predicted_density_scatter)

    # Update layout for better aesthetics
    fig.update_layout(
        title='Predicted Fish Density in the Arabian Sea',
        scene=dict(
            xaxis=dict(title='Longitude'),
            yaxis=dict(title='Latitude'),
            zaxis=dict(title='Depth (m)'),
        ),
        plot_bgcolor='rgb(255,255,255)'  # Set background color to white
    )

    # fig.show()



    return fig
# Main part of your Streamlit app
st.title('Fish Density Prediction Visualization')
st.write('This visualization predicts fish density based on depth and coordinates in the Arabian Sea.')

# When a button is clicked, the plot will be displayed
if st.button('Show Predicted Fish Density'):
    fig = create_fish_density_plot()
    st.plotly_chart(fig, use_container_width=True)
