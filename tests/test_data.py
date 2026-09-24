import pandas as pd
import pytest

from ealz.data import make_folds, read_split


def test_read_split_resolves_paths_and_rejects_missing_images(tmp_path):
    (tmp_path / "cn").mkdir()
    (tmp_path / "cn" / "a.png").write_bytes(b"x")
    csv = tmp_path / "split.csv"
    pd.DataFrame({"filepath": ["cn/a.png"], "label": ["cn"]}).to_csv(csv, index=False)
    assert read_split(csv, str(tmp_path))["filepath"].tolist() == [str(tmp_path / "cn" / "a.png")]

    pd.DataFrame({"filepath": ["cn/a.png", "cn/b.png"], "label": ["cn", "cn"]}).to_csv(csv, index=False)
    with pytest.raises(FileNotFoundError, match="1 of 2 images"):
        read_split(csv, str(tmp_path))


def test_folds_match_the_protocol():
    # 3,000 slices, 1,000 per class, as in the paper.
    slices = pd.DataFrame({"filepath": [f"s{i}.png" for i in range(3000)], "label": ["cn", "emci", "lmci"] * 1000})
    tested = []
    for _, train, val, test in make_folds(slices, n_folds=5, seed=0):
        assert (len(train), len(val), len(test)) == (1920, 480, 600)
        assert set(test["label"].value_counts()) == {200}
        assert set(val["label"].value_counts()) == {160}
        assert not set(train.index) & set(val.index) and not set(train.index) & set(test.index)
        assert not set(val.index) & set(test.index)
        tested += test.index.tolist()
    assert sorted(tested) == list(range(3000))  # every slice is tested exactly once
