""" Database actions """
import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine


def get_db_table(table: str, index_col='id'):
    """ Read SQL table from database and return as dataframe.

    :param table: Name of the table in database.
    :type table: str
    :param index_col: Column which should be set as the dataframe's index.
    :type index_col: str
    :return: All data as dataframe.
    :rtype: pandas.DataFrame
    """
    # Read url from secret environment variable. Set this in your CI environment.
    url = os.getenv('DATABASE_URL')
    if url is None:
        print("ERROR: Environment variable DATABASE_URL not set.")
        return pd.DataFrame()
        
    # Create an engine instance.
    engine = create_engine(url, pool_recycle=3600)

    # Connect to PostgreSQL server.
    conn = engine.connect()

    # Read data from PostgreSQL database table and load into a DataFrame instance.
    dataFrame = pd.read_sql(f"select * from \"{table}\"", conn, index_col=index_col)
    
    # Close the database connection.
    conn.close()
    return dataFrame
