"""OpenCV'nin örnek takip verisini ve modelini doğrulanmış hash ile indir."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
MODEL_BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_tracking_vittrack/"
DATA_BASE = "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/tracking/"
FILES = {
    "assets/vittrack.onnx": (MODEL_BASE + "object_tracking_vittrack_2023sep.onnx", "2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d"),
    "data/david.webm": (DATA_BASE + "david/data/david.webm", "4bbe69b7d097379af4d7ad128f02692820dc57a6d8275ea2b9e7eef4399a3e61"),
    "data/dudek.webm": (DATA_BASE + "dudek/data/dudek.webm", "679c95a22c71e259971ccaaabf805fdba55c70e224b28eafa7536b7989c722e1"),
    "data/faceocc2.webm": (DATA_BASE + "faceocc2/data/faceocc2.webm", "5ffedeca4fb3acfd65e6f6aa71f016f748628727be1e24914fde2198b42089c5"),
    "data/david_gt.txt": (DATA_BASE + "david/gt.txt", "8e46f079b39f7877f9650bcf2fc42879b03f22d77ef832ca1d3e1ed258269588"),
    "data/dudek_gt.txt": (DATA_BASE + "dudek/gt.txt", "6645d041ab014a2b3b7f34ca271c1ac42d3d72a9546443c7cc9c542f772f56c4"),
    "data/faceocc2_gt.txt": (DATA_BASE + "faceocc2/gt.txt", "7d7b2dc7251e20fbb2fedfc36356ffbc7146ce59f49c21df e251b8714b87ab88".replace(" ", "")),
    "data/david.yml": (DATA_BASE + "david/david.yml", None),
    "data/dudek.yml": (DATA_BASE + "dudek/dudek.yml", None),
    "data/faceocc2.yml": (DATA_BASE + "faceocc2/faceocc2.yml", None),
}


def valid(path: Path, expected: str | None) -> bool:
    return path.is_file() and (expected is None or sha256(path.read_bytes()).hexdigest() == expected)


def main() -> None:
    for relative, (url, expected) in FILES.items():
        path = ROOT / relative
        path.parent.mkdir(exist_ok=True)
        if valid(path, expected):
            print(f"OK: {relative}")
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if not valid(temporary, expected):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Dosya doğrulanamadı: {relative}")
        temporary.replace(path)
        print(f"İndirildi: {relative}")


if __name__ == "__main__":
    main()
