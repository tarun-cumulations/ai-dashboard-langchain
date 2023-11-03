from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv 
import json

load_dotenv()

def create_db_engine(dbtype, host, database, user, password):
    db_string = ""
    if "postgres" in dbtype.lower():
        db_string = f"postgresql://{user}:{password}@{host}/{database}"
    elif "mysql" in dbtype.lower():
        db_string = f"mysql://{user}:{password}@{host}/{database}"
    # Add more elif conditions for other databases

    try:
        engine = create_engine(db_string)
        return engine
    except Exception as e:
        print("Error while connecting to the database:", e)
        return None

def fetch_dataframe_from_table(engine, table_name):
    query = f"SELECT * FROM {table_name};"
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def get_common_column(engine, table_names):
    common_columns = None
    with engine.connect() as conn:
        for table in table_names:
            query = text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}';")
            result = conn.execute(query)
            columns = {row[0] for row in result}
            if common_columns is None:
                common_columns = columns
            else:
                common_columns &= columns
    return list(common_columns)[0] if common_columns else None

def merge_dataframes(engine, table_names):
    common_column = get_common_column(engine, table_names)
    if common_column:
        df_list = [fetch_dataframe_from_table(engine, table_name) for table_name in table_names]
        df_final = df_list[0]
        for df in df_list[1:]:
            df_final = pd.merge(df_final, df, on=common_column, how='inner')
        return df_final
    else:
        print("No common column found to merge on.")
        return None

if __name__ == "__main__":
    
    host = "ai-dashboard.cox4boq5aldo.ap-south-1.rds.amazonaws.com"
    database = "postgres"
    user = "postgres"
    password = "postgres"
    dbtype = "postgres"

    engine = create_db_engine(dbtype, host, database, user, password)

    if engine:
        conn = engine.connect()
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';"))

        table_names = [row[0] for row in result]
        merged_df = merge_dataframes(engine, table_names)

        if merged_df is not None:
            print("Merged Dataframe:")
            print(merged_df)
        else:
            print("Could not merge dataframes.")
