import pandas as pd
import numpy as np

df_input = pd.read_excel('./input.xlsx')
df_input.index = df_input.columns

def agglomerative(sim: pd.DataFrame, R: float):
    sim = sim.copy()
    sim[np.isnan(sim)] = 0
    sim = sim + sim.T
    steps = 1 
    print(f"Исходная матрица:\n{sim}")


    while len(sim) > 1:

        stack_sim = sim.stack()

        max_value = stack_sim.max()
        a, b  = stack_sim.idxmax()


        if max_value < R:
            break

        new_row = sim.loc[[a, b]].max().drop([a, b])

        sim = sim.drop(index=[a, b], columns=[a, b])

        name = f"{a}, {b}"
        sim[name] = new_row
        sim.loc[name] = new_row
        sim.loc[name, name] = 0

        print(f"Matrix step№{steps}:")
        print(f"\n{sim}")

        steps +=1 

    return list(sim.index)


result = agglomerative(df_input, 0.85)

print(result)