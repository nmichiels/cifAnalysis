import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import pandas as pd 

df = pd.read_csv("./example/models/CD8-CD4-NK_C0C1C2C3C4C5_valid20_gridSearch/gridsearchResults.csv")


data = [
    go.Parcoords(
         line = dict(color = df['mean_test_f1'],
                   colorscale = 'Jet',
                   showscale = True,
                   reversescale = False,
                   cmin = 0.92,
                   cmax = 0.95),
        dimensions = list([
            dict(tickvals = [1,2,3],
                 label = 'conv_layer_sizes', values = df['conv_layer_sizes'],
                 ticktext = ['[32, 32]', '[32, 64]', '[64, 64]']),
            dict(
                 tickvals = [1,2],
                 label = 'dense_layer_sizes', values = df['dense_layer_sizes'],
                 ticktext = ['[64]', '[128]']),
            dict(
                 tickvals = [0.25,0.5],
                 label = 'dropout', values = df['param_dropout'],
                 ticktext = ['0.25', '0.5']),
            dict(
                 tickvals = [3,5],
                 label = 'kernel_size', values = df['param_kernel_size'],
                 ticktext = ['3', '5']),
            dict(
                 tickvals = [2],
                 label = 'pool_size', values = df['param_pool_size'],
                 ticktext = ['2']),    
            dict(range=[0.87,1],
                 label = 'mean_test_accuracy', values = df['mean_test_accuracy']),
            dict(range=[0.87,1],
                 label = 'mean_train_f1', values = df['mean_train_f1']),
            dict(range=[0,1],
                 label = 'std_test_f1', values = df['std_test_f1']),
            dict(range=[0.87,1],
                 label = 'mean_test_f1', values = df['mean_test_f1']),


        ])
    )
]


# data = [
#     go.Parcoords(
#         line = dict(color = df['species_id'],
#                    colorscale = [[0,'#D7C16B'],[0.5,'#23D8C3'],[1,'#F3F10F']]),
#         dimensions = list([
#             dict(range = [0,8],
#                 constraintrange = [4,8],
#                 label = 'Sepal Length', values = df['sepal_length']),
#             dict(range = [0,8],
#                 label = 'Sepal Width', values = df['sepal_width']),
#             dict(range = [0,8],
#                 label = 'Petal Length', values = df['petal_length']),
#             dict(range = [0,8],
#                 label = 'Petal Width', values = df['petal_width'])
#         ])
#     )
# ]

layout = go.Layout(
    plot_bgcolor = '#E5E5E5',
    paper_bgcolor = '#E5E5E5'
)

fig = go.Figure(data = data, layout = layout)
plot(fig, filename = 'parcoords-basic.html')