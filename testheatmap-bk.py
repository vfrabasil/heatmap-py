import numpy as np
import pandas as pd
import panel as pn
import pathlib
import time
import base64
from pylab import *
from os import truncate, write

import streamlit as st
import streamlit.components.v1 as components
from streamlit.report_thread import add_report_ctx

import altair as alt
from altair.vegalite.v4.schema.channels import Tooltip
from altair import Row, Column, Chart, Text, Scale, Color

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
from plotly import tools

import seaborn as sns
import plotnine as p9
import bqplot

from ipywidgets import Layout
from ipywidgets import widgets
from IPython.display import display

import cufflinks as cf
import chart_studio
import holoviews as hv 
from holoviews import opts

from bokeh.io import show, output_notebook, output_file
from bokeh.plotting import figure, show, save
from bokeh.palettes import BuPu
from bokeh.layouts import row
from bokeh.models import CustomJS
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    FactorRange
)

# Modo de Ejecucion:
# streamlit run testheatmap.py
debugInfo = True

'''
Para mi muestra genero un dataset con valores semanales entre 1920 y 2020 (48 semanas x 100 años) y lo populo con valores
aleatorios interos.
retorno el dataset original y la version matricial para las librerias que utilian una matriz para general el grafico. Esto
lo hago con un simple pivot sobre las dos primeras columnas dejando la tercera como el dato a mostrar en cada celda del mapa de calor
'''
def generateDf():
    df = pd.DataFrame()
    rows = []
    for year in range(1920, 2021):
        for week in range(1, 49):
            rows.append([year, week, np.random.normal(loc=150,scale=np.random.randint(50,100)) ])
    df = pd.DataFrame(rows, columns=['year','week','ratio'])
    dfmatrix = df.pivot("week", "year", "ratio")
    return df, dfmatrix


def displayMatplotLib(df):
    global mycmaps
    global colName0
    global colName1

    fig = plt.figure()
    fig, ax = plt.subplots(1,1, figsize=(20,20))
    heatplot = ax.imshow(df, cmap=mycmaps)
    plt.xticks(rotation=90)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(heatplot, cax=cax)

    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_title("MATPLOTLIB")
    ax.set_xlabel(colName0)
    ax.set_ylabel(colName1)
    st.pyplot(fig)


def displaySeaborn(df):
    global mycmaps
    fig = plt.figure(figsize=(16,8))
    r = sns.heatmap(df, cmap=mycmaps)
    r.set_title("SEABORN")
    st.pyplot(fig)


def displayGgplot(df):
    global myscheme

    fig = plt.figure()
    p = p9.ggplot(df, p9.aes(df.columns[0],df.columns[1])) + p9.geom_tile(p9.aes(fill=df.columns[2]))\
        + p9.scale_fill_gradientn(colors=myscheme) \
        + p9.ggtitle("GGPLOT") \
        + p9.theme(figure_size = (12, 6))
    st.pyplot(p9.ggplot.draw(p))


def displayBqPlot(df):
    col_sc = bqplot.ColorScale()
    grid_map = bqplot.GridHeatMap(color=df, scales={'color': col_sc})
    bqplot.Figure(marks=[grid_map], padding_y=0.0)
    st.write( grid_map )


def displayPloty(df):
    global myscheme
    fig = go.Figure( data=go.Heatmap(z=df, x=df.columns, y=df.index, colorscale=myscheme))
    fig.update_layout(title="PLOTY") 
    st.plotly_chart(fig)


def displayCufflinks( df):
    global mycmaps
    fig = df.T.iplot(title="CUFFLINKS", asFigure=True, kind='heatmap', colorscale=mycmaps )
    st.plotly_chart(fig)


def displayBokeh(df1, df):
    ta0 = time.time()
    global mycmaps
    global colName0
    global colName1
    global colName2

    hv.extension('bokeh', 'matplotlib')
    colormap =cm.get_cmap(mycmaps)
    bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    mapper = LinearColorMapper(palette=bokehpalette, low=df.min().min(), high=df.max().max())
    x_range = FactorRange(factors=df.columns.astype('str'))

    years = list(df.columns.astype('str'))
    weeks = list(df.index.astype('str'))
    df1[colName0] = df1[colName0].astype('str')

    source = ColumnDataSource(df1)
    z = figure(title="BOKEH", x_range=years, y_range=weeks, # tools="hover",
            toolbar_location='above',  toolbar_sticky=False) #below

    z.rect(x=colName0, y=colName1, width=1, height=1, source=source
        , fill_color={'field': colName2, 'transform': mapper}, line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                        ticker=BasicTicker(desired_num_ticks=8),
                        formatter=PrintfTickFormatter(format="%d%%"),
                        label_standoff=6, border_line_color=None, location=(0, 0))
    z.add_layout(color_bar, 'right')
    z.xaxis.axis_label = colName0
    z.xaxis.major_label_orientation = "vertical"
    z.xaxis.major_label_text_font_size = "4pt"
    z.yaxis.major_label_text_font_size = "6pt"
    z.yaxis.axis_label = colName1
    plot = pn.panel(z)
    path = pathlib.Path(__file__).parent/'testB.html'
    plot.save(path, embed=True, max_states=100)
    html=path.read_text()
    components.html(html, height=640)

    ta1 = time.time()
    if debugInfo:
        st.write(f'Size: **{df.shape}** - Time: **{ta1 - ta0:.5f}s**')
        st.write('___')
        st.write('##')


def displayHoloviews(df):
    hv.extension('bokeh', 'matplotlib')
    global mycmaps
    heatmap = hv.HeatMap(df, label="HOLOVIEWS" )
    overlay = (heatmap)
    overlay.opts(
    opts.HeatMap(width=680, height=400, tools=['hover'], colorbar=True, logz=True, cmap=mycmaps, 
                 invert_yaxis=True, labelled=[], toolbar='above', xrotation= 90, 
                 clim=(1, np.nan)),
    opts.VLine(line_color='black'),
    )
    plot = pn.panel(heatmap)
    path = pathlib.Path(__file__).parent/'test.html'
    plot.save(path, embed=True, max_states=100)
    html=path.read_text()
    components.html(html, height=640)


def displayAltair(df):
    xx = df[df.columns[0]].nunique()
    yy = df[df.columns[1]].nunique()
    myfill='#ebf7f1'

    col0 = df.columns[0]
    col1 = df.columns[1]
    col2 = df.columns[2] 

    myTooltip=[col0, col1, col2+':Q']
    global myscheme
    factor = min(xx, yy)/6
    chart = alt.Chart(df, title='ALTAIR').mark_rect( #stroke='black', strokeWidth=0.5
    ).encode(
        x=col0 + ':O',
        y=col1 + ':O',
        color=alt.Color(col2+':Q' , scale=alt.Scale(range=[myscheme[0], myscheme[len(myscheme) // 2], myscheme[-1]]) ),
        tooltip=myTooltip
    ).configure_scale(
        bandPaddingInner=0.01
    ).configure_legend(
        gradientLength= max(200, yy*4),
        gradientThickness=15
    ).configure_view(
        fill=myfill,
    ).properties(
        width = max( 200, xx*factor),
        height = max( 200, yy*factor),
        autosize=alt.AutoSizeParams(
            type='fit',
            contains='padding'
        )
    ).configure_axis(
        labelFontSize=(factor+2)/2,
        titleFontSize=15
    )
    st.altair_chart(chart)


'''
Uso un "decorator" para mostrar en cada funcion de graficacion el tamaño del dataset y el tiempo que tarda en generarse
'''
def timing(func, df):
    global debugInfo
    ta0 = time.time()
    func(df)
    ta1 = time.time()
    if debugInfo:
        st.write(f'Size: **{df.shape}** - Time: **{ta1 - ta0:.5f}s**')
        st.write('___')
        st.write('##')


'''
Esta funcion convierte una paleta de colores de matplotlib (CMAP) en una lista de colores RGB que utilizan algunas de las librerias
'''
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def cmap2rgb(mycmap):
    sourcecmap = cm.get_cmap(mycmap, 5) 
    scheme = []
    for i in range(sourcecmap.N):
        rgba = sourcecmap(i)
        # rgb2hex accepts rgb or rgba
        scheme.append(matplotlib.colors.rgb2hex(rgba))
    return scheme


'''
Dos custom made separadores, uno para el frame principal y otro para el sidebar
'''
def sep():
    st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)

def sepS():
    st.sidebar.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)


'''
Setting Cache for dataset:
lo uso para cargar el dataset desde un archivo .CSV
'''
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_dataset(myfile ):
    df = pd.read_csv(myfile, sep=';')
    return df

def getFile():
    sepS()
    myfile = None
    myfile  = st.sidebar.file_uploader("📂 open file:", 
                                accept_multiple_files=False,
                                type='csv')
    sepS()   
    return myfile


# ------------------------------------------------------------------------------
def main():

    st.title("Heatmaps for 🐍Python:")
    st.subheader("Try different libraries and palettes")
    st.write('Some libraries use a matrix to generate the heatmap, therefore the original dataset is pivoted to generate it. \
                As a preview, we will show the original version of the dataset, and the version converted into a matrix.')
    st.write('In case of using an input .CSV file, it must respect the order **Data X**; **Data Y**; **Value**.\
                The name of the columns does not matter.')


    '''
    Estas son las principales paletas de colores obtenidas desde matplotlib, las guardo en una lista para poder
    mostrar y seleccionar desde un listbox:
    '''
    cmaps = [
            'Blues', 'Oranges', 'Purples', 'Greys', 'Greens', 'Reds',           #Sequential
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',         #Sequential (2)
            'viridis', 'plasma', 'inferno', 'magma', 'cividis',                 #Perceptually Uniform Sequential
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',                     #Diverging
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'twilight', 'twilight_shifted', 'hsv',                              #Cyclic
            'Pastel1', 'Pastel2', 'Paired', 'Accent',                           #Qualitative
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',    #Miscellaneous
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar']


    '''
    Algunas variables globales (horror!), pero necesaria para que todas las funciones que grafican mantengan la misma estructura
    '''
    global mycmaps 
    global myscheme
    global colName0
    global colName1
    global colName2

    '''
    Genero el dataset aleatorio:
    '''
    df, dfmat = generateDf()


    '''
    Aca posiciono el selectbox para elegir la paleta a usar en las graficas, y la convierto a un  "scheme" que
    utilizan algunas de las librerias. Un "scheme" es basicamente una lista de colores RGB.
    '''
    mycmaps = st.sidebar.selectbox('🎨 Select Palette (Matplotlib):', cmaps)
    link = "[matplotlib](https://matplotlib.org/stable/tutorials/colors/colormaps.html)"
    st.sidebar.markdown(link, unsafe_allow_html=True)
    myscheme = cmap2rgb(mycmaps)
    myfile = getFile()


    '''
    Si el dataset proviene desde un archivo, lo cargo y genero la forma matricial del mismo:
    '''
    if myfile is not None:
        df = load_dataset(myfile)
        dfmat = df.pivot(df.columns[1], df.columns[0], df.columns[2])
    

    '''
    Boton para generar un nuevo dataset de forma random
    '''
    if st.sidebar.button('🎲 Generate Random'):
        if myfile is None:
            df, dfmat = generateDf()
    
    '''
    En caso de que el dataset lo obtenga desde un archivo, guardo el nombre de cada columna para las etiquetas de los graficos:
    '''
    colName0 = df.columns[0]
    colName1 = df.columns[1]
    colName2 = df.columns[2] 

    ''' 
    Genero un "preview" del dataset original y de la matriz
    '''
    sep()
    col1, col2 = st.beta_columns(2)
    col1.write("*Original* Dataset preview:")
    col1.write(df.head(15))
    col2.write("*Matrix* Dataset preview:")
    col2.write(dfmat.head(15))
    sep()


    '''
    Cada checkbox activa el heatmap de cada libreria grafica
    '''
    if st.checkbox("display Altair"):
        timing(displayAltair, df)
    if st.checkbox("display Holoviews"):
        timing(displayHoloviews, df)
    if st.checkbox("display Bokeh"):
        displayBokeh(df, dfmat) 
    if st.checkbox("display MatplotLib"):
        timing(displayMatplotLib, dfmat)
    if st.checkbox("display Seaborn"):
        timing(displaySeaborn, dfmat)
    if st.checkbox("display Ggplot"):
        timing(displayGgplot, df)
    if st.checkbox("display Ploty"):
        timing(displayPloty, dfmat)
    if st.checkbox("display Cufflinks"):
        timing(displayCufflinks, dfmat)

# -----------------------------------------------------------------------------
if __name__ == '__main__':
	main()
# -----------------------------------------------------------------------------



