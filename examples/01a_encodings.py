from time import time
from skrub.datasets import fetch_employee_salaries
from skrub import TableVectorizer
import pandas as pd
from systemds.context import SystemDSContext
from systemds.operator.algorithm import imputeByMean, na_locf
import json
import cProfile
import pstats
from joblib import parallel_backend

# Load dataset
dataset = fetch_employee_salaries()
employees, salaries = dataset.X, dataset.y

# Append dataset n times to have a larger dataset
#employees = pd.concat([employees] * 10, ignore_index=True)
print(employees.info())

# Encode the data using the TableVectorizer
t1 = time()
pr = cProfile.Profile()
# pr.enable()
vectorizer = TableVectorizer(n_jobs=1)
employees_enc = vectorizer.fit_transform(employees)
# pr.disable()
print(f"Encoded data shape: {employees_enc.shape}")
t2 = time()
print(f"Encoding time: {t2 - t1:.3f} seconds")
print(employees_enc.columns)

# Explore the encodings
print(vectorizer.kind_to_columns_)
print("Fitted transformers to department column")
print(vectorizer.transformers_["department"])
print("Fitted transformers to division column")
print(vectorizer.transformers_["division"])
print(vectorizer.input_to_outputs_["date_first_hired"])
# print(employees_enc[vectorizer.input_to_outputs_["date_first_hired"]])
print(vectorizer.output_to_input_["department_BOA"])
# print(vectorizer.all_processing_steps_["date_first_hired"])
print(vectorizer.all_processing_steps_["department"])

# stats = pstats.Stats(pr).sort_stats("cumtime")
# stats.print_stats(30)


# Using SystemDS to encode the data
sds = SystemDSContext(logging_level=10, capture_stdout=True)

DATA_PATH = "employees.csv"
DATA_SCHEMA = '"string,string,string,string,string,string,string,string"'
JSPEC_PATH = "employees_spec1.json"

# pr.enable()
F1 = sds.read(
    DATA_PATH,
    data_type="frame",
    schema=DATA_SCHEMA,
    format="csv",
    header=False,
)
F1 = F1.rbind(F1).rbind(F1).rbind(F1).rbind(F1)
F1 = F1.rbind(F1)

t1 = time()
jspec = sds.read(JSPEC_PATH, data_type="scalar", value_type="string")
X, M = F1.transform_encode(spec=jspec)
X_imputed = na_locf(X)
X_cl = X_imputed.compute()
t2 = time()
print(f"Encoding time (SystemDS): {t2 - t1:.3f} seconds")
# pr.disable()
print(X_cl.shape)

# stats = pstats.Stats(pr).sort_stats("cumtime")
# stats.print_stats(30)
