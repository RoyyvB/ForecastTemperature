from preprocessing import read_ahu_one, MergeData, PrepareData
from calculations import CalculateEnergy

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

data = PrepareData()
data = CalculateEnergy(data)

# import plotly.offline as py
# import plotly.graph_objs as go
# import plotly.tools as tls

# trace = go.Scatter(
#     x = data['date'].values,
#     y = data['energy'].values,
#     mode='markers',
#     marker=dict(
#         sizemode='diameter',
#         sizeref=1,
#         size=25,
#         color=data['energy'].values,
#         colorscale='Portland',
#         showscale=True
#     ),
#     text=data['energy'].values
# )

# viz = [trace]

# layout = go.Layout(
#     autosize=True,
#     title='Energy Consumption',
#     hovermode='closest',

#     yaxis=dict(
#         title='Energy in Joule',
#         ticklen=5,
#         gridwidth=2
#     ),
#     showlegend=False
# )

# fig = go.Figure(data=viz, layout=layout)
# py.iplot(fig,filename='attempt')

