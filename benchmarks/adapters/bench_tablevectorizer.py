import time
import stratum as skrub
from sklearn.preprocessing import OneHotEncoder
from skrub.datasets import fetch_employee_salaries
from skrub import TableVectorizer, StringEncoder
import pandas as pd
from joblib import parallel_backend

def main():
    # Load dataset
    dataset = fetch_employee_salaries()
    employees, salaries = dataset.X, dataset.y

    # Append dataset n times to have a larger dataset
    employees = pd.concat([employees] * 10, ignore_index=True)
    print(employees.info())
    employees = employees.dropna() #necessary for rusty one-hot encoder

    # Use skrub's vanilla TableVectorizer
    skrub.set_config(rust_backend=False, debug_timing=False)
    t0 = time.perf_counter()
    vectorizer = TableVectorizer(n_jobs=-1)
    employees_enc = vectorizer.fit_transform(employees)
    t1 = time.perf_counter()
    exec_time = t1 - t0
    print(f"skrub - Encoding time: {exec_time:8.3f}s\n")
    print(f"Encoded data shape: {employees_enc.shape}")


    # Use stratum's TableVectorizer
    t0 = time.perf_counter()
    skrub.set_config(rust_backend=True, debug_timing=False, scheduler=True, stats=True)
    with parallel_backend('threading'):
        vectorizer = TableVectorizer(high_cardinality=StringEncoder(), low_cardinality=OneHotEncoder(), n_jobs=-1) #default setup
        employees_enc = vectorizer.fit_transform(employees)
    t1 = time.perf_counter()
    exec_time = t1 - t0
    print(f"stratum - Encoding time: {exec_time:8.3f}s\n")
    print(f"Encoded data shape: {employees_enc.shape}")

    # Explore the encodings
    print(vectorizer.kind_to_columns_)
    print("Fitted transformers to department column")
    print(vectorizer.transformers_["department"]) #low_cardinality
    print("Fitted transformers to division column")
    print(vectorizer.transformers_["division"]) #high_cardinality

if __name__ == "__main__":
    main()
