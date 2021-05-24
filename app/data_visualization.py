import plotly
import pandas as pd
from plotly.graph_objs import Bar

def return_figures(df):
    
    """
    Function creates visualization out of DisasterResponse database using plotly and return figures
    in the form of list.
    
    Parameters:
    -----------
        df : dataframe out of which plotly visualizations will be created
    
    Returns:
    --------
        figures : list of plotly visualizations .
    """
    
    #create empty list to store list of figures
    figures = []
    
    # extract data needed for visuals..
    
    #1. Data Visualization for number of message count against each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    genre_dict = {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
             }
        }
    
    figures.append(genre_dict)
    
    #2. Data Visualization for categories count
    
    #creating dataframe for count of categories
    category_colnames = df.drop(['id','genre','message','original'],axis=1).columns
    df_new=df[df[category_colnames]==1][category_colnames].count().to_frame().reset_index()
    df_new.columns = ['category','count']
    category = df_new['category']
    category_count = df_new['count']
    
    category_dict = {
            'data': [
                Bar(
                    x=category,
                    y=category_count,
                    textposition='outside',
                    marker=dict(color='green')
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category_Name",
                    'automargin':True
                },

            }
        }
    figures.append(category_dict)
    
    return figures