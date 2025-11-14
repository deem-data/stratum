from time import time

import stratum as skrub
from sklearn.preprocessing import OneHotEncoder
from skrub.datasets import fetch_employee_salaries
from skrub import TableVectorizer, StringEncoder
import pandas as pd
from systemds.context import SystemDSContext
from systemds.operator.algorithm import imputeByMean, na_locf
from joblib import parallel_backend

# Load dataset
dataset = fetch_employee_salaries()
employees, salaries = dataset.X, dataset.y

# Append dataset n times to have a larger dataset
employees = pd.concat([employees] * 10, ignore_index=True)
print(employees.info())
employees = employees.dropna() #necessary for rusty one-hot encoder

# Use skrub's vanilla TableVectorizer
skrub.set_config(rust_backend=False, debug_timing=False)
t1 = time()
vectorizer = TableVectorizer(n_jobs=-1)
employees_enc = vectorizer.fit_transform(employees)
t2 = time()
print(f"skrub - Encoding time: {t2 - t1:.3f} seconds\n")
print(f"Encoded data shape: {employees_enc.shape}")


# Use stratum's TableVectorizer
t1 = time()
skrub.set_config(rust_backend=True, debug_timing=False)
with parallel_backend('threading'):
    vectorizer = TableVectorizer(high_cardinality=StringEncoder(), low_cardinality=OneHotEncoder(), n_jobs=-1) #default setup
    #vectorizer = TableVectorizer(high_cardinality=StringEncoder(), low_cardinality=StringEncoder(), n_jobs=-1)
    employees_enc = vectorizer.fit_transform(employees)
t2 = time()
print(f"stratum - Encoding time: {t2 - t1:.3f} seconds\n")
print(f"Encoded data shape: {employees_enc.shape}")

# Explore the encodings
print(vectorizer.kind_to_columns_)
print("Fitted transformers to department column")
print(vectorizer.transformers_["department"]) #low_cardinality
print("Fitted transformers to division column")
print(vectorizer.transformers_["division"]) #high_cardinality
