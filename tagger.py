import json
import subprocess
from typing import BinaryIO, Iterable, Mapping, Tuple, Callable
import numpy as np

class Vector(np.ndarray):
    """1D array of floats"""
    __init__: Callable[[Iterable[float]], None]


W2V = Mapping[str, Vector]
RawData = Mapping[str, str]
Data = Mapping[Vector, str]


def parse_glove(f: BinaryIO) -> Mapping[str, Vector]:
    with f as lines:
        w2v = {
            line.split()[0].decode("utf-8"): Vector(map(float, (line.split()[1:])))
            for line in lines
        }
        return w2v


def vectorize(w2v: Mapping[str, Vector], document: str) -> Vector:
    vectors = [w2v[word] for word in document.split() if word in w2v]
    mean = Vector(np.mean(vectors or np.zeros(len(next(iter(w2v.values())))), axis=0))
    return mean


def knn(data: Data, vector: Vector) -> str:
    def dist(entry: Tuple[Vector, str], vector: Vector = vector) -> float:
        return np.linalg.norm(entry[0] - vector)

    nearest = min(data, key=dist)
    return nearest[1]


def suggest_tags(w2v: W2V, data: Data, text: str) -> str:
    vector = vectorize(w2v, text)
    return knn(data, vector)


def main() -> None:
    try:
        f = open("glove.6B.50d.txt", "rb")
    except FileNotFoundError:
        subprocess.call(["wget", "http://nlp.stanford.edu/data/glove.6B.zip"])
        subprocess.call(["unzip", "glove.6B.zip"])
        f = open("love.6B.50d.zip", "rb")
    w2v = parse_glove(f)
    try:
        train = json.load(open("train.json"))
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
            suggested_tags = knn(data, vector)
            print(f"Suggested tags: {suggested_tags}")
            real_tags = input("Your tags (blank to accept suggestions as-is): ")
            if not real_tags:
                real_tags = suggested_tags
            user[vector] = data[vector] = real_tags
    finally:
        json.dump(user, open("user.json", "w"))
