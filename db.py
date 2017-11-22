import sqlite3 as lite
import pandas as pd
import load_functions as lfunctions
import numpy as np
import json

con = lite.connect("states.db")
cursor = con.cursor()
#delete
cursor.execute("""DROP TABLE pairwise_comparisons;""")

sql_command = """
CREATE TABLE pairwise_comparisons (
stateaction_1 VARCHAR(300),
stateaction_2 VARCHAR(300),
value DOUBLE);"""

con.execute(sql_command)

with con:
    con.row_factory = lite.Row

    cur = con.cursor()

    cur.execute("SELECT DISTINCT step FROM game1")
    rows = cur.fetchall()

    for row in rows:
        cur.execute("SELECT * FROM game1 WHERE step =?", (row["step"],))
        placements = cur.fetchall()

        resultList = pd.DataFrame(columns=["step", "stateaction", "value"])
        for placement in placements:
            resultList = resultList.append({
                "step" : placement["step"],
                "stateaction": json.dumps(np.array(lfunctions.parseState(placement["stateaction"]).reshape(-1)).tolist()),
                "value": placement["value"]}, ignore_index=True)

        pw = lfunctions.makePairwiseComparisons(resultList)
        for p in pw:
            cur.execute("INSERT INTO pairwise_comparisons VALUES(?,?,?)", (p[0], p[1], p[2]))

        con.commit()


