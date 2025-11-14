import sys
import pandas as pd
import stratum as skrub
from sklearn.preprocessing import OneHotEncoder
from skrub import TableVectorizer, StringEncoder

def capture_std_out(capfd):
    # Capture timing output
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + (captured.err or "")
    return combined_output

def test_tablevectorizer_stringencoder(capfd):
    df = pd.DataFrame({
        'A': ['one', 'two', 'two', 'three'],
        'B': ['02/02/2024', '23/02/2024', '12/03/2024', '13/03/2024'],
        'C': ['1.5', 'N/A', '12.2', 'N/A'],
    })
    skrub.set_config(rust_backend=True, debug_timing=True)
    vectorizer = TableVectorizer(low_cardinality=StringEncoder())
    _ = vectorizer.fit_transform(df)
    # Assert if the fitted transformers is RustyStringEncoder
    assert repr(vectorizer.transformers_['A']).startswith('RustyStringEncoder')
    # Assert if the Rust code is executed
    assert "[rust]" in capture_std_out(capfd)

def test_tablevectorizer_onehotencoder(capfd):
    df = pd.DataFrame({
        'A': ['one', 'two', 'two', 'three'],
        'B': ['02/02/2024', '23/02/2024', '12/03/2024', '13/03/2024'],
        'C': ['1.5', 'N/A', '12.2', 'N/A'],
    })
    skrub.set_config(rust_backend=True, debug_timing=True)
    vectorizer = TableVectorizer(low_cardinality=OneHotEncoder())
    _ = vectorizer.fit_transform(df)
    # Assert if the fitted transformers is RustyStringEncoder
    assert repr(vectorizer.transformers_['A']).startswith('RustyOneHotEncoder')
    # Assert if the Rust code is executed
    assert "[rust]" in capture_std_out(capfd)
