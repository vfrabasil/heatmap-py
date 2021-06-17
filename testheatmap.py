from os import truncate, write
import pathlib
import panel as pn

from altair.vegalite.v4.schema.channels import Tooltip
import numpy as np
import pandas as pd
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import altair as alt
from altair import Row, Column, Chart, Text, Scale, Color
from streamlit.report_thread import add_report_ctx
import streamlit.components.v1 as components
import base64

import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
#from plotnine import * 
import plotnine as p9

import bqplot
from ipywidgets import Layout
from ipywidgets import widgets
from IPython.display import display

from plotly import tools
import plotly.offline as py

import cufflinks as cf
import chart_studio


from bokeh.io import show, output_notebook, output_file
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    FactorRange
)
#from bokeh.charts import HeatMap, bins, output_file, show, vplot
from bokeh.plotting import figure, show, save
from bokeh.palettes import BuPu
from bokeh.layouts import row

from bokeh.models import CustomJS
#from streamlit_bokeh_events import streamlit_bokeh_events


import holoviews as hv 
from holoviews import opts
#from holoviews import opts, dim

#from lightning import Lightning
import time

debugInfo = True




#Modo de Ejecucion:
# streamlit run testheatmap.py
def generateDf():
    df = pd.DataFrame()
    rows = []

    for year in range(1920, 2021):
        for week in range(1, 49):
            #if week <= 12:
            #    rows.append([year, week, np.random.normal(100, 50) ])
            #if 13 < week <= 24:
            #    rows.append([year, week, np.random.normal(150, 75) ])
            #if 25 < week <= 36:
            #    rows.append([year, week, np.random.normal(200, 100) ])
            #if week > 37:
            #    rows.append([year, week, np.random.normal(250, 150) ])
            rows.append([year, week, np.random.normal(loc=150,scale=np.random.randint(50,100)) ])
            
    df = pd.DataFrame(rows, columns=['year','week','ratio'])
    #df['ratio'] = df['ratio'].apply(np.int64)

    dfmatrix = df.pivot("week", "year", "ratio")
    return df, dfmatrix

    #mysize = 200
    #min_max = ([1960, 2020], [1, 12], [0, 430])
    #def make_series(low, high, name):
    #    if any(isinstance(_, float) for _ in (low, high)):
    #        func = np.random.uniform
    #    else:
    #        func = np.random.randint
    #    return pd.Series(func(low, high, size=(mysize,)), name=name)
    #df = pd.concat([make_series(lo, hi, name) for (lo, hi), name in zip(min_max, "ABC")], axis=1)
    #df.columns = ['year','month','passengers']
    #df = df.groupby(['year','month'])['passengers'].sum()
    #print(df.head(20))


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

    #myscheme  = [ '#edf8fb', '#b3cde3', '#8856a7', '#810f7c']
    #myscheme2 = [ '#9ebcda', '#8c6bb1', '#88419d', '#6e016b']
    global myscheme

    fig = plt.figure()
    p = p9.ggplot(df, p9.aes(df.columns[0],df.columns[1])) + p9.geom_tile(p9.aes(fill=df.columns[2]))\
        + p9.scale_fill_gradientn(colors=myscheme) \
        + p9.ggtitle("GGPLOT") \
        + p9.theme(figure_size = (12, 6))
    st.pyplot(p9.ggplot.draw(p))


def displayBqPlot(df):
    #x = df.index
    #y = df.columns
    #x_sc, y_sc, col_sc = bqplot.OrdinalScale(), bqplot.OrdinalScale(), bqplot.ColorScale(scheme='BuPu')
    #ax_x = bqplot.Axis(scale=x_sc, label='year')
    #ax_y = bqplot.Axis(scale=y_sc, orientation='vertical', label='week')
    #ax_c = bqplot.ColorAxis(scale=col_sc)
    ##row must be an array of size color.shape[0]
    #heat = bqplot.marks.HeatMap(color=df, \
    #            scales={'row': x_sc, 'column': y_sc, 'color': col_sc}, stroke='white', row=y.tolist(), column=x.tolist())
    #
    #fig = bqplot.Figure(marks=[heat], axes=[ax_x, ax_y, ax_c],
    #            title="displayBqPlot", layout=Layout(width='800px', height='800px'))
    #st.write(heat)

    col_sc = bqplot.ColorScale()
    grid_map = bqplot.GridHeatMap(color=df, scales={'color': col_sc})
    bqplot.Figure(marks=[grid_map], padding_y=0.0)
    st.write( grid_map )


def displayPloty(df):
    #colorscale = [[0, '#edf8fb'], [.3, '#b3cde3'],  [.6, '#8856a7'],  [1, '#810f7c']]
    global myscheme
    fig = go.Figure( data=go.Heatmap(z=df, x=df.columns, y=df.index, colorscale=myscheme))
    fig.update_layout(title="PLOTY") 
    st.plotly_chart(fig)


def displayCufflinks( df):
    #Cufflinks is a third-party wrapper library around Plotly, inspired by the Pandas .plot() API.
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
    # Bokeh doesn't have its own gradient color maps supported but you can easily use on from matplotlib.
    colormap =cm.get_cmap(mycmaps)
    bokehpalette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]

    #this mapper is what transposes a numerical value to a color. 
    mapper = LinearColorMapper(palette=bokehpalette, low=df.min().min(), high=df.max().max())
    x_range = FactorRange(factors=df.columns.astype('str'))


    years = list(df.columns.astype('str'))
    weeks = list(df.index.astype('str'))

    #df1[df.columns[0]] = df1[df1.columns[0]].astype('str')
    #df1['year'] = df1.year.astype('str')
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

    #output_file("testB.html")
    #save(z)
    #show(z)
    #HtmlFile = open("testB.html", 'r', encoding='utf-8')
    #html = HtmlFile.read() 
    #components.html(html, height = 640) #Default height is 150


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
    #opts.Overlay(width=900, height=300, show_title=False)
    )
    #st.write(hv.render(heatmap, backend='bokeh')) #bokeh_chart?

    plot = pn.panel(heatmap)

    path = pathlib.Path(__file__).parent/'test.html'
    plot.save(path, embed=True, max_states=100)
    html=path.read_text()
    components.html(html, height=640)


def displayAltair(df):
    #chart = alt.Chart(df).mark_text(
    #        ).encode(
    #            x='year:O',
    #            y='week:O',
    #            #row=Row('week', sort='none'),
    #            #column='year',
    #            #text=Text(value=' '),
    #            color=alt.Color('ratio:Q', scale=alt.Scale(type='linear', range=['#bfd3e6', '#6e016b'], scheme='greenblue'))
    #        ).configure_scale(
    #            bandPaddingInner=0.01
    #            #textBandWidth=55,
    #            #bandSize=60
    #        )
    #st.altair_chart(chart)

    xx = df[df.columns[0]].nunique()
    yy = df[df.columns[1]].nunique()
    #myscheme = 'greenblue'
    myfill='#ebf7f1'


    col0 = df.columns[0]
    col1 = df.columns[1]
    col2 = df.columns[2] 

    myTooltip=[col0, col1, col2+':Q']

    #df.sort(by=['year', 'week'], ascending=[True, False], inplace= True)
    global myscheme
    #myscheme  = [ '#edf8fb', '#b3cde3', '#8856a7', '#810f7c']
    #myscheme2 = [ '#9ebcda', '#8c6bb1', '#88419d', '#6e016b']
    factor = min(xx, yy)/6
    chart = alt.Chart(df, title='ALTAIR').mark_rect( #stroke='black', strokeWidth=0.5
    ).encode(
        x=col0 + ':O',
        y=col1 + ':O',
        color=alt.Color(col2+':Q' , scale=alt.Scale(range=[myscheme[0], myscheme[len(myscheme) // 2], myscheme[-1]]) ),  #'ratio:Q'
        tooltip=myTooltip
        #stroke='black', strokeWidth=2
        #strokeWidth=alt.StrokeWidthValue(0, condition=alt.StrokeWidthValue(3, selection=highlight.name))
    ).configure_scale(
        bandPaddingInner=0.01
    ).configure_legend(
        gradientLength= max(200, yy*4),
        gradientThickness=15
    ).configure_view(
        fill=myfill,
    ).properties(
        #width='container',
        #height='container'
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



#def displayLightning(df):
    #lgn = Lightning(ipython=True, host='http://public.lightning-viz.org')
    #lgn = Lightning(host='https://herokuappname.herokuapp.com')
    
    #lgn = Lightning()

    ##lgn = Lightning(local=True) #this is the server we are going to use.
    ##st.write(lgn.matrix(df, colormap='BuPu', 
    ##    row_labels=list(df.index.values), 
    ##    column_labels=list(df.columns.values), width=1000, description="displayLightning") )

    #data = np.random.random(100)
    ## typical plot
    #lgn.line(data)
    ## custom plot
    #lgn.plot(data, type='my-custom-lightning-viz')

def timing(func, df):
    global debugInfo
    ta0 = time.time()
    func(df)
    ta1 = time.time()
    if debugInfo:
        st.write(f'Size: **{df.shape}** - Time: **{ta1 - ta0:.5f}s**')
        st.write('___')
        st.write('##')


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def cmap2rgb(mycmap):
    sourcecmap = cm.get_cmap(mycmap, 5) 
    scheme = []
    for i in range(sourcecmap.N):
        rgba = sourcecmap(i)
        # rgb2hex accepts rgb or rgba
        scheme.append(matplotlib.colors.rgb2hex(rgba))
    return scheme


# Setting Cache for dataset:
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_dataset(myfile ):
    df = pd.read_csv(myfile, sep=';')
    return df

def sep():
    st.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)

def sepS():
    st.sidebar.markdown('<hr style="height:2px;border-width:0;color:gray;background-color:blue">', unsafe_allow_html=True)


#def getFile():
#    sepS()
#    myfile = None
#    myfile  = st.sidebar.file_uploader("open file:", 
#                                accept_multiple_files=False,
#                                type='csv')
#    if st.sidebar.checkbox('Random Dataset' ):
#        myfile  = None
#    sepS()
#    return myfile

def getFile():
    sepS()
    myfile = None
    myfile  = st.sidebar.file_uploader("📂 open file:", 
                                accept_multiple_files=False,
                                type='csv')
    sepS()   
    return myfile

#@st.cache(suppress_st_warning=True)
#def onStart():
#    # This function will only be run the first time it's called
#    return generateDf()


# ------------------------------------------------------------------------------
def main():

    st.title("Heatmaps for 🐍Python:")
    st.subheader("Try different libraries and palettes")
    st.write('Some libraries use a matrix to generate the heatmap, therefore the original dataset is pivoted to generate it. \
                As a preview, we will show the original version of the dataset, and the version converted into a matrix.')
    st.write('In case of using an input .CSV file, it must respect the order **Data X**; **Data Y**; **Value**.\
                The name of the columns does not matter.')


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

    global mycmaps 
    global myscheme
    global colName0
    global colName1
    global colName2

    df, dfmat = generateDf()

    mycmaps = st.sidebar.selectbox('🎨 Select Palette (Matplotlib):', cmaps)
    link = "[matplotlib](https://matplotlib.org/stable/tutorials/colors/colormaps.html)"
    st.sidebar.markdown(link, unsafe_allow_html=True)
    myscheme = cmap2rgb(mycmaps)
    myfile = getFile()



    if myfile is not None:

        df = load_dataset(myfile)
        dfmat = df.pivot(df.columns[1], df.columns[0], df.columns[2])

    
    if st.sidebar.button('🎲 Generate Random'):
        if myfile is None:
            df, dfmat = generateDf()
            
    #def download_csv(name, df):
    #    #csv = df.to_csv(index=False)
    #    csv = df.to_csv(sep=";", index=False)
    #    base = base64.b64encode(csv.encode()).decode()
    #    file = (f'<a href="data:file/csv;base64,{base}" download="%s.csv">Download file</a>' % (name))
    #    return file
    #st.markdown(download_csv('Data Frame',df), unsafe_allow_html=True)
    
    colName0 = df.columns[0]
    colName1 = df.columns[1]
    colName2 = df.columns[2] 

    sep()
    col1, col2 = st.beta_columns(2)
    col1.write("*Original* Dataset preview:")
    col1.write(df.head(15))
    col2.write("*Matrix* Dataset preview:")
    col2.write(dfmat.head(15))
    sep()

    #displayLightning(dfmat) #doesnt work
    #displayBqPlot(dfmat) #doesnt work
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



