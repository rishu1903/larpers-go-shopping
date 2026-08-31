from __future__ import annotations

from dataclasses import dataclass, field
import re
from collections.abc import Iterable

from src.hard_constraints import coerce_price
from src.shopping_intent import canonicalize_attribute_value
from src.slots import detect_explicit_slots


PROFILE_ATTRIBUTES = (
    "material",
    "color",
    "size",
    "style",
    "feature",
    "use_case",
)


# Catalogue fields are deliberately not treated as interchangeable.
#
# The V15B audit showed that title/features/description carry most
# material, colour, style and feature evidence, while categories are
# especially useful for use-case semantics. Store names are explicitly
# excluded from attribute extraction because a store called "Black
# Leather Co" is not evidence that every product is black or leather.
_ATTRIBUTE_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "material": ("title", "features", "description"),
    "color": ("title", "features", "description"),
    "size": ("title", "features", "description"),
    "style": ("title", "features", "description"),
    "feature": ("title", "features", "description"),
    "use_case": ("title", "features", "description", "categories"),
}


_FIELD_ALLOWED_ATTRIBUTES: dict[str, tuple[str, ...]] = {}
for _attribute, _field_names in _ATTRIBUTE_TEXT_FIELDS.items():
    for _field_name in _field_names:
        _FIELD_ALLOWED_ATTRIBUTES[_field_name] = (
            *_FIELD_ALLOWED_ATTRIBUTES.get(_field_name, ()),
            _attribute,
        )


_DETAIL_KEY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "material": (
        re.compile(r"\bmaterial\b", re.I),
        re.compile(r"\bfabric\b", re.I),
    ),
    "color": (
        re.compile(r"^colou?r$", re.I),
    ),
    "size": (
        re.compile(r"^size$", re.I),
    ),
    "style": (
        re.compile(r"\bstyle\b", re.I),
        re.compile(r"^fit type$", re.I),
    ),
    "feature": (
        re.compile(r"\bspecial feature\b", re.I),
    ),
    "use_case": (
        re.compile(r"\bsport type\b", re.I),
        re.compile(r"\brecommended uses?\b", re.I),
        re.compile(r"\bactivity\b", re.I),
        re.compile(r"\boccasion\b", re.I),
    ),
}


_DEPARTMENT_ALIASES = {
    "women": "women",
    "womens": "women",
    "women's": "women",
    "woman": "women",
    "men": "men",
    "mens": "men",
    "men's": "men",
    "man": "men",
    "girls": "girls",
    "girl": "girls",
    "boys": "boys",
    "boy": "boys",
    "unisex-adult": "unisex_adult",
    "unisex adult": "unisex_adult",
    "unisex": "unisex",
    "baby": "baby",
    "baby-boys": "baby_boys",
    "baby boys": "baby_boys",
    "baby-girls": "baby_girls",
    "baby girls": "baby_girls",
}


@dataclass(frozen=True)
class ProductSignal:
    """One normalized product-side attribute with provenance."""

    value: str
    sources: tuple[str, ...]
    native: bool = False

    @property
    def support_count(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class ProductProfile:
    """
    Read-only, provenance-aware catalogue interpretation.

    The profile is the product-side counterpart of CompiledShoppingIntent
    and is used conservatively for observed negative-constraint filtering:

        shopper language -> normalized intent values
        catalogue text  -> normalized product values

    Both sides share canonicalize_attribute_value().
    """

    parent_asin: str
    title: str
    category_path: tuple[str, ...]
    leaf_category: str | None

    department: str | None
    store: str | None
    manufacturer: str | None
    native_brand: str | None
    price: float | None

    attributes: dict[str, tuple[ProductSignal, ...]] = field(
        default_factory=dict
    )

    def values(self, attribute: str) -> tuple[str, ...]:
        return tuple(
            signal.value
            for signal in self.attributes.get(attribute, ())
        )

    def to_dict(self) -> dict:
        return {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "category_path": list(self.category_path),
            "leaf_category": self.leaf_category,
            "department": self.department,
            "store": self.store,
            "manufacturer": self.manufacturer,
            "native_brand": self.native_brand,
            "price": self.price,
            "attributes": {
                attribute: [
                    {
                        "value": signal.value,
                        "sources": list(signal.sources),
                        "native": signal.native,
                        "support_count": signal.support_count,
                    }
                    for signal in signals
                ]
                for attribute, signals in self.attributes.items()
            },
        }


def _flatten_strings(value: object) -> list[str]:
    if value is None:
        return []

    if isinstance(value, dict):
        flattened: list[str] = []
        for key, item in value.items():
            flattened.append(str(key))
            flattened.extend(_flatten_strings(item))
        return flattened

    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_strings(item))
        return flattened

    text = re.sub(r"\s+", " ", str(value)).strip()
    return [text] if text else []


def _field_text(product: dict, field_name: str) -> str:
    return " ".join(_flatten_strings(product.get(field_name)))


def _category_path(product: dict) -> tuple[str, ...]:
    return tuple(_flatten_strings(product.get("categories")))


def _normalize_department(value: object) -> str | None:
    values = _flatten_strings(value)
    if not values:
        return None

    normalized = re.sub(r"\s+", " ", values[0]).strip().casefold()
    return _DEPARTMENT_ALIASES.get(
        normalized,
        normalized.replace("-", "_").replace(" ", "_"),
    )


def _first_detail(details: dict, *keys: str) -> str | None:
    wanted = {key.casefold() for key in keys}

    for key, value in details.items():
        if str(key).strip().casefold() not in wanted:
            continue

        values = _flatten_strings(value)
        if values:
            return values[0]

    return None


def _merge_signal(
    bucket: dict[tuple[str, str], dict],
    *,
    attribute: str,
    value: str,
    source: str,
    native: bool,
) -> None:
    canonical = canonicalize_attribute_value(value)
    if not canonical:
        return

    key = (attribute, canonical)
    entry = bucket.setdefault(
        key,
        {
            "sources": [],
            "native": False,
        },
    )

    if source not in entry["sources"]:
        entry["sources"].append(source)

    entry["native"] = bool(entry["native"] or native)


def _detect_from_text(
    bucket: dict[tuple[str, str], dict],
    *,
    text: str,
    source: str,
    allowed_attributes: Iterable[str],
    native: bool,
) -> None:
    if not text:
        return

    allowed = set(allowed_attributes)
    if not allowed:
        return


    for detected in detect_explicit_slots(text):
        if detected.attribute not in allowed:
            continue

        _merge_signal(
            bucket,
            attribute=detected.attribute,
            value=detected.value,
            source=source,
            native=native,
        )


def build_product_profile(product: dict) -> ProductProfile:
    """
    Compile one catalogue row into the controlled product taxonomy.

    This function is deterministic and label-free. It does not infer
    hidden attributes that are absent from catalogue evidence.
    """

    details = product.get("details")
    if not isinstance(details, dict):
        details = {}

    merged: dict[tuple[str, str], dict] = {}

    # Scan each shared catalogue field once, then route detections only to
    # dimensions that trust that field. This preserves V15C semantics while
    # avoiding repeated full-vocabulary regex passes over identical text.
    for field_name, allowed_attributes in _FIELD_ALLOWED_ATTRIBUTES.items():
        _detect_from_text(
            merged,
            text=_field_text(product, field_name),
            source=field_name,
            allowed_attributes=allowed_attributes,
            native=False,
        )

    # A detail key can support more than one dimension. Determine all allowed
    # dimensions first, then scan its value once.
    for key, value in details.items():
        key_text = str(key).strip()
        if not key_text:
            continue

        allowed_attributes = [
            attribute
            for attribute, patterns in _DETAIL_KEY_PATTERNS.items()
            if any(pattern.search(key_text) is not None for pattern in patterns)
        ]
        if not allowed_attributes:
            continue

        text = " ".join(_flatten_strings(value))
        if not text:
            continue

        _detect_from_text(
            merged,
            text=text,
            source=f"details:{key_text}",
            allowed_attributes=allowed_attributes,
            native=True,
        )

    grouped: dict[str, list[ProductSignal]] = {}

    for (attribute, value), metadata in merged.items():
        grouped.setdefault(attribute, []).append(
            ProductSignal(
                value=value,
                sources=tuple(metadata["sources"]),
                native=bool(metadata["native"]),
            )
        )

    for signals in grouped.values():
        signals.sort(key=lambda signal: signal.value)

    categories = _category_path(product)

    department = _normalize_department(
        _first_detail(details, "Department", "Suggested Users")
    )

    manufacturer = _first_detail(details, "Manufacturer")
    native_brand = _first_detail(details, "Brand", "Brand Name")

    store_values = _flatten_strings(product.get("store"))
    store = store_values[0] if store_values else None

    title_values = _flatten_strings(product.get("title"))
    title = title_values[0] if title_values else ""

    return ProductProfile(
        parent_asin=str(product.get("parent_asin") or ""),
        title=title,
        category_path=categories,
        leaf_category=(categories[-1] if categories else None),
        department=department,
        store=store,
        manufacturer=manufacturer,
        native_brand=native_brand,
        price=coerce_price(product.get("price")),
        attributes={
            attribute: tuple(signals)
            for attribute, signals in grouped.items()
            if signals
        },
    )
