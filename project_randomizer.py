import pandas as pd
import numpy as np
def randomizer(xls_file,no_group):
    file=pd.read_excel(xls_file)
    shuffled_xls=file.sample(frac=1,random_state=42).reset_index(drop=True)
    file= np.array_split(shuffled_xls,no_group) ##did not use np.spitt() because it raises an error if the array can not be divided in to equal parts
    return file
