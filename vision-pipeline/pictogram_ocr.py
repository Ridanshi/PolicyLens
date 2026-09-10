"""
Reads printed handling pictograms + text off a carton crop and turns them
into a structured policy dict.

Icon matching: ORB feature matching against reference icon crops (no training
needed, works with a handful of reference images per icon — good enough for
a hackathon timeline; swap for a fine-tuned classifier later if time allows).
Drop reference crops into assets/icons/<icon_name>/*.png before running —
a few real photos or clean symbol renders per icon, 3-5 each is enough.

Text: EasyOCR reads printed numbers (weight limit, stack limit) near the icons.
"""
import os
import re
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import easyocr

ICON_DIR = os.path.join(os.path.dirname(__file__), "assets", "icons")
MATCH_THRESHOLD = 15  # min good ORB matches to accept an icon as present

DEFAULT_POLICY = {
    "fragile": False,
    "orientation": "any",
    "max_stack": None,
    "max_drop_height_m": 0.3,
    "keep_dry": False,
    "source": "default_unknown",
}


@dataclass
class Policy:
    fragile: bool
    orientation: str
    max_stack: int | None
    max_drop_height_m: float
    keep_dry: bool
    source: str

    def to_dict(self):
        return asdict(self)


class PictogramReader:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=800)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ocr_reader = easyocr.Reader(["en"], gpu=False)
        self.references = self._load_references()

    def _load_references(self) -> dict:
        refs = {}
        if not os.path.isdir(ICON_DIR):
            return refs
        for icon_name in os.listdir(ICON_DIR):
            icon_path = os.path.join(ICON_DIR, icon_name)
            if not os.path.isdir(icon_path):
                continue
            descriptors = []
            for fname in os.listdir(icon_path):
                img = cv2.imread(os.path.join(icon_path, fname), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                _, des = self.orb.detectAndCompute(img, None)
                if des is not None:
                    descriptors.append(des)
            if descriptors:
                refs[icon_name] = descriptors
        return refs

    def _icon_present(self, crop_gray: np.ndarray, icon_name: str) -> bool:
        _, des_crop = self.orb.detectAndCompute(crop_gray, None)
        if des_crop is None:
            return False
        for ref_des in self.references.get(icon_name, []):
            matches = self.matcher.match(ref_des, des_crop)
            good = [m for m in matches if m.distance < 40]
            if len(good) >= MATCH_THRESHOLD:
                return True
        return False

    def _extract_numbers(self, crop_bgr: np.ndarray) -> list[str]:
        result = self.ocr_reader.readtext(crop_bgr, detail=0)
        return [t for t in result if re.search(r"\d", t)]

    def read(self, carton_crop_bgr: np.ndarray) -> Policy:
        if carton_crop_bgr is None or carton_crop_bgr.size == 0 or not self.references:
            return Policy(**DEFAULT_POLICY)

        gray = cv2.cvtColor(carton_crop_bgr, cv2.COLOR_BGR2GRAY)

        fragile = self._icon_present(gray, "fragile")
        upright_required = self._icon_present(gray, "this_side_up")
        keep_dry = self._icon_present(gray, "keep_dry")

        max_stack = None
        if self._icon_present(gray, "stack_limit"):
            for text in self._extract_numbers(carton_crop_bgr):
                digits = re.findall(r"\d+", text)
                if digits:
                    max_stack = int(digits[0])
                    break

        found_anything = fragile or upright_required or keep_dry or max_stack is not None
        return Policy(
            fragile=fragile,
            orientation="upright" if upright_required else "any",
            max_stack=max_stack,
            max_drop_height_m=0.0 if fragile else 0.3,
            keep_dry=keep_dry,
            source="label_extracted" if found_anything else "default_unknown",
        )
