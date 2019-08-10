import subprocess
from typing import BinaryIO, Sequence, Iterable, Mapping, Callable
import numpy as np


class Vector(np.array, Sequence):
    """1D array of floats"""

    def __init__(self, values: Iterable[float]) -> None:
        super(Vector, self).__init__(values)


W2V = Mapping[str, Vector]


class Unsafe:
    def load_glove(self) -> BinaryIO:
        try:
            f = open("glove.6B.50d.txt", "rb")
        except FileNotFoundError:
            subprocess.call(["wget", "http://nlp.stanford.edu/data/glove.6B.zip"])
            subprocess.call(["unzip", "glove.6B.zip"])
            f = open("love.6B.zip", "rb")
        return f

    default_data_files = ("train.json", "user.json")

    def load_data(
        self, fnames: Optional[Iterable[str]]
    ) -> Union[Mapping[Vector, str], Mapping[str, str]]:
        if not fnames:
            fnames = self.default_data_files
        data = {
            key: value
            for key, value in json.load(open(fname))
            for fname in fnames
            if os.path.isfile(fname)
        }
        return data


    def vectorize_data(self, w2v) -> None:
        data = self.load_data(("raw.json",))
        self.save_data(vectorized, "train.json")
        # can be rewritten as a streaming function with tuples of data instead of dict

    def main(self):
        w2v = self.load_glove()
        try:
            train = json.load("train.json")
        except FileNotFoundError:
            raw = json.load(open("raw.json"))
            train = {vectorize(w2v, text): tags for text, tags in raw}
            json.dump(train, open("train.json", "w"))
        try:
            user = json.load(open("user.json"))
        except FileNotFoundError:
            user = {}
        data = {**train, **user}
        try:
            while True:
                text = input("Text to be tagged: ")
                vector = vectorize(w2v, text)
                suggested_tags = knn(data, vecotr)
                print(f"Suggested tags: {suggested}")
                real_tags = input("Your tags (blank to accept suggestions as-is): ")
                if not real_tags:
                    real_tags = suggested
                user[vector] = data[vector] = real_tags
        finally:
            json.dump(user, open("user.json", "w"))


def parse_glove(f: BinaryIO) -> Mapping[str, Vector]:
    with f as lines:
        w2v = {
            line.split()[0].decode("utf-8"): Vector(map(float, (line.split()[1:])))
            for line in lines
        }
        return w2v


def vectorize(w2v: Mapping[str, Vector], document: str) -> Vector:
    vectors = [w2v[word] for word in document.split() if word in w2v]
    mean = Vector(np.mean(vectors or np.zeros(len(next(w2v.values()))), axis=0))
    return mean


def knn(data: Mapping[Vector, str], vector: Vector) -> str:
    def dist(entry: Tuple[Vector, str], vector=vector) -> float:
        return np.linalg.norm(entry[0] - vector)

    nearest = min(data, key=dist)
    return nearest[1]


def suggest_tags(w2v: W2V, data: Mapping[Vector, str], text: str) -> str:
    vector = vectorize(w2v, text)
    return knn(data, vector)
