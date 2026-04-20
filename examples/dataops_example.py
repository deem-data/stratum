import stratum as skrub #drop-in replacement
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

def main():
    dataset = skrub.datasets.fetch_employee_salaries()
    df = skrub.as_data_op(dataset.employee_salaries).skb.subsample()
    df_clean = df.dropna()
    y = df_clean["current_annual_salary"].skb.mark_as_y()
    X = df_clean.drop(columns=["current_annual_salary"]).skb.mark_as_X()

    skrub.set_config(rust_backend=True, debug_timing=True, scheduler=True, stats=True)
    tv = skrub.TableVectorizer(high_cardinality=skrub.StringEncoder(), low_cardinality=OneHotEncoder())
    X_enc = X.skb.apply(tv)
    print(f"Encoded data shape: {X_enc.shape.skb.preview()}")

    pred = X_enc.skb.apply(LinearRegression(), y=y)
    search = pred.skb.make_grid_search(cv=3, fitted=True, scoring="r2", refit=False)
    print(search.results_)

if __name__ == "__main__":
    main()