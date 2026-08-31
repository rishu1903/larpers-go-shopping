from __future__ import annotations

from dataclasses import dataclass, field
import re


STRUCTURED_ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
)


# Attributes where several simultaneous values
# are naturally valid.
MULTI_VALUE_ATTRIBUTES = {
    "style",
    "feature",
    "use_case",
}


_ATTRIBUTE_PATTERNS: dict[
    str,
    tuple[
        re.Pattern[str],
        ...,
    ],
] = {
    "material": (
        re.compile(
            r"\bcotton\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpolyester\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bnylon\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bleather\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwool\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bspandex\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bsilk\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\brayon\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdenim\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\blinen\b",
            re.IGNORECASE,
        ),
    ),

    "color": (
        re.compile(
            r"\bblack\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhite\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bblue\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bred\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpink\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bgreen\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bgrey\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bgray\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbrown\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbeige\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\byellow\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpurple\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\borange\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bnavy\b",
            re.IGNORECASE,
        ),
    ),

    "size": (
        re.compile(
            (
                r"\bsize\s+"
                r"(?:xxs|xs|small|medium|large|"
                r"xl|xxl|xxxl|\d+(?:\.\d+)?)\b"
            ),
            re.IGNORECASE,
        ),
        re.compile(
            r"\bextra[- ]small\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bextra[- ]large\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bplus[- ]size\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpetite\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwide\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bnarrow\b",
            re.IGNORECASE,
        ),
    ),

    "style": (
        re.compile(
            r"\bslim(?:[- ]fit)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\brelaxed(?:[- ]fit)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bloose(?:[- ]fit)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bregular[- ]fit\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bstraight[- ]leg\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bv[- ]neck\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bcrew[- ]neck\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\blong[- ]sleeve(?:d)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bshort[- ]sleeve(?:d)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bsleeveless\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhigh[- ]waist(?:ed)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\blow[- ]rise\b",
            re.IGNORECASE,
        ),
    ),

    "use_case": (
        re.compile(
            r"\bhiking\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\brunning\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bjogging\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bjogs?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\btrail(?:s)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bgym\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bworkout(?:s)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwinter\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\boutdoor(?:s)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwork\b",
            re.IGNORECASE,
        ),
    ),

    "feature": (
        re.compile(
            r"\bwaterproof\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwater[- ]resistant\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbreathable\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpockets?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhooded\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhood\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\binsulat(?:ed|ion)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\blightweight\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:non[- ]?slip|slip[- ]resistant|anti[- ]slip)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bstretch(?:y)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\badjustable\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdurable\b",
            re.IGNORECASE,
        ),
    ),
}


_BRAND_RE = re.compile(
    (
        r"\bbrand"
        r"(?:\s+is|:)?\s+"
        r"("
        r"[a-z0-9][a-z0-9&.'-]*"
        r"(?:\s+[a-z0-9][a-z0-9&.'-]*)?"
        r")\b"
    ),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DetectedSlot:
    attribute: str
    value: str
    confidence: float


@dataclass
class SlotObservation:
    """
    One structured conversational fact.

    status:
        active
        superseded
        cleared
    """

    attribute: str
    value: str | None

    source_turn: int
    intent_epoch: int

    confidence: float

    strength: str
    provenance: str

    status: str = "active"

    resolved_turn: int | None = None
    resolution_reason: str | None = None


@dataclass
class SlotState:
    """
    Structured conversational memory.

    Unlike the first V14A implementation, this
    supports multiple simultaneously active values
    for attributes such as:

        feature
        style
        use_case

    Example:

        feature:
            waterproof
            lightweight
            breathable

    Explicit overrides can still replace all active
    values for the affected attribute.
    """

    history: list[
        SlotObservation
    ] = field(
        default_factory=list
    )

    def _resolve_active(
        self,
        *,
        attribute: str,
        source_turn: int,
        provenance: str,
        status: str,
    ) -> None:

        for observation in self.history:

            if (
                observation.attribute
                != attribute
            ):
                continue

            if (
                observation.status
                != "active"
            ):
                continue

            observation.status = status

            observation.resolved_turn = (
                source_turn
            )

            observation.resolution_reason = (
                provenance
            )

    def set_slot(
        self,
        *,
        attribute: str,
        value: str,
        source_turn: int,
        intent_epoch: int,
        confidence: float,
        strength: str,
        provenance: str,
        mode: str = "replace",
    ) -> SlotObservation:

        if (
            attribute
            not in STRUCTURED_ATTRIBUTES
        ):
            raise ValueError(
                (
                    "Unsupported structured "
                    f"attribute: {attribute}"
                )
            )

        if mode not in {
            "replace",
            "append",
        }:
            raise ValueError(
                (
                    "slot mode must be "
                    "'replace' or 'append'"
                )
            )

        normalized = (
            normalize_slot_value(
                value
            )
        )

        if not normalized:
            raise ValueError(
                "slot value cannot be blank"
            )

        # Reuse an identical active value rather
        # than duplicating it.
        for observation in reversed(
            self.history
        ):

            if (
                observation.attribute
                == attribute

                and

                observation.status
                == "active"

                and

                observation.value
                is not None

                and

                observation.value.casefold()
                == normalized.casefold()
            ):

                observation.confidence = max(
                    observation.confidence,
                    float(
                        confidence
                    ),
                )

                return observation

        if mode == "replace":

            self._resolve_active(
                attribute=attribute,
                source_turn=source_turn,
                provenance=provenance,
                status="superseded",
            )

        observation = (
            SlotObservation(
                attribute=attribute,
                value=normalized,
                source_turn=source_turn,
                intent_epoch=intent_epoch,
                confidence=float(
                    confidence
                ),
                strength=strength,
                provenance=provenance,
            )
        )

        self.history.append(
            observation
        )

        return observation

    def replace_values(
        self,
        *,
        attribute: str,
        values: list[str],
        source_turn: int,
        intent_epoch: int,
        confidence: float,
        strength: str,
        provenance: str,
    ) -> list[
        SlotObservation
    ]:
        """
        Atomically replace one attribute with one
        or more new active values.

        This is useful for explicit override:

            feature:
                waterproof
                lightweight

        becomes:

            feature:
                breathable
                hooded
        """

        self._resolve_active(
            attribute=attribute,
            source_turn=source_turn,
            provenance=provenance,
            status="superseded",
        )

        created: list[
            SlotObservation
        ] = []

        seen: set[
            str
        ] = set()

        for value in values:

            normalized = (
                normalize_slot_value(
                    value
                )
            )

            key = (
                normalized.casefold()
            )

            if (
                not normalized
                or
                key in seen
            ):
                continue

            seen.add(
                key
            )

            created.append(
                self.set_slot(
                    attribute=attribute,
                    value=normalized,
                    source_turn=source_turn,
                    intent_epoch=intent_epoch,
                    confidence=confidence,
                    strength=strength,
                    provenance=provenance,
                    mode="append",
                )
            )

        return created

    def clear_slot(
        self,
        *,
        attribute: str,
        source_turn: int,
        provenance: str,
    ) -> None:
        """
        Clear every active value for an attribute.
        """

        if (
            attribute
            not in STRUCTURED_ATTRIBUTES
        ):
            return

        self._resolve_active(
            attribute=attribute,
            source_turn=source_turn,
            provenance=provenance,
            status="cleared",
        )

    def active_slot(
        self,
        attribute: str,
    ) -> SlotObservation | None:
        """
        Return the most recent active value.

        Kept for backward compatibility with
        V14A diagnostics.
        """

        for observation in reversed(
            self.history
        ):

            if (
                observation.attribute
                == attribute

                and

                observation.status
                == "active"
            ):
                return observation

        return None

    def active_values(
        self,
        attribute: str,
    ) -> list[
        SlotObservation
    ]:
        """
        Return every active value for one
        attribute in insertion order.
        """

        return [
            observation

            for observation
            in self.history

            if (
                observation.attribute
                == attribute

                and

                observation.status
                == "active"
            )
        ]

    def active_snapshot(
        self,
    ) -> dict[str, str]:
        """
        Compatibility snapshot.

        For each attribute this returns the most
        recently active value.

        Use active_value_snapshot() when all
        simultaneous values are needed.
        """

        snapshot: dict[
            str,
            str,
        ] = {}

        for attribute in (
            STRUCTURED_ATTRIBUTES
        ):

            observation = (
                self.active_slot(
                    attribute
                )
            )

            if (
                observation is not None

                and

                observation.value
                is not None
            ):
                snapshot[
                    attribute
                ] = (
                    observation.value
                )

        return snapshot

    def active_value_snapshot(
        self,
    ) -> dict[
        str,
        list[str],
    ]:
        """
        Full structured context including all
        simultaneous active values.
        """

        snapshot: dict[
            str,
            list[str],
        ] = {}

        for attribute in (
            STRUCTURED_ATTRIBUTES
        ):

            values = [
                observation.value

                for observation
                in self.active_values(
                    attribute
                )

                if (
                    observation.value
                    is not None
                )
            ]

            if values:

                snapshot[
                    attribute
                ] = [
                    str(
                        value
                    )

                    for value
                    in values
                ]

        return snapshot

    def history_for(
        self,
        attribute: str,
    ) -> list[
        SlotObservation
    ]:

        return [
            observation

            for observation
            in self.history

            if (
                observation.attribute
                == attribute
            )
        ]

    def effective_confidence(
        self,
        observation: SlotObservation,
        current_turn: int,
    ) -> float:
        """
        Explicit current-session evidence does not
        decay.

        Only weak/inferred context may decay.
        """

        base = max(
            0.0,
            min(
                1.0,
                float(
                    observation.confidence
                ),
            ),
        )

        if observation.provenance in {
            "category",
            "question_answer",
            "explicit_keyword",
            "override",
            "budget_parser",
        }:
            return base

        age = max(
            0,
            int(
                current_turn
            )
            -
            int(
                observation.source_turn
            ),
        )

        if (
            observation.provenance
            == "profile_prior"
        ):
            return (
                base
                *
                (
                    0.70
                    ** age
                )
            )

        if (
            observation.provenance
            == "weak_inference"
        ):
            return (
                base
                *
                (
                    0.85
                    ** age
                )
            )

        return base


def normalize_slot_value(
    text: str,
) -> str:

    value = re.sub(
        r"\s+",
        " ",
        str(
            text
        ),
    ).strip()

    value = re.sub(
        (
            r"^For that, "
            r"what matters is:\s*"
        ),
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        (
            r"^Actually, "
            r"ignore my earlier preference\.\s*"
            r"What I need is:\s*"
        ),
        "",
        value,
        flags=re.IGNORECASE,
    )

    return value.strip(
        " .;,"
    )


def detect_explicit_slots(
    text: str,
) -> list[
    DetectedSlot
]:
    """
    Conservatively identify explicit slot values.

    Multiple values for the same attribute are
    retained.

    Example:

        "waterproof and lightweight"

    produces:

        feature=waterproof
        feature=lightweight
    """

    normalized = (
        normalize_slot_value(
            text
        )
    )

    if not normalized:
        return []

    found: list[
        tuple[
            int,
            DetectedSlot,
        ]
    ] = []

    seen: set[
        tuple[
            str,
            str,
        ]
    ] = set()

    for (
        attribute,
        patterns,
    ) in _ATTRIBUTE_PATTERNS.items():

        for pattern in patterns:

            for match in pattern.finditer(
                normalized
            ):

                value = (
                    match.group(0)
                    .strip()
                )

                key = (
                    attribute,
                    value.casefold(),
                )

                if key in seen:
                    continue

                seen.add(
                    key
                )

                found.append(
                    (
                        match.start(),

                        DetectedSlot(
                            attribute=attribute,
                            value=value,
                            confidence=0.95,
                        ),
                    )
                )

    brand_match = (
        _BRAND_RE.search(
            normalized
        )
    )

    if brand_match is not None:

        value = (
            brand_match
            .group(1)
            .strip()
        )

        key = (
            "brand",
            value.casefold(),
        )

        if key not in seen:

            found.append(
                (
                    brand_match.start(),

                    DetectedSlot(
                        attribute="brand",
                        value=value,
                        confidence=0.98,
                    ),
                )
            )

    found.sort(
        key=lambda item: (
            item[0],
            item[1].attribute,
            item[1].value.casefold(),
        )
    )

    return [
        detected

        for (
            _,
            detected,
        ) in found
    ]